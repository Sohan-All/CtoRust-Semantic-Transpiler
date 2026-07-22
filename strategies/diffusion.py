"""Strategy A — diffusion merge.

Pass 0 describes seed blocks leaves-first (callee descriptions substituted into
caller prompts). Merge passes then combine open explanations along adjacency and
call-DAG edges until everything is locked or irreducible.

Invariants are combined in code (verbatim union), never rewritten by the merge
call — the model only writes the combined description.
"""

from __future__ import annotations

import asyncio

from chunker import SeedGraph, SeedBlock, chunk
from config import Config
from llm import LLM
from lock_check import LockChecker, LockResult
from state import Explanation, Store, OPEN, LOCKED, IRREDUCIBLE
from validator import check_coverage, blank_lines

DESCRIBE_PROMPT = """\
You are documenting a C source file one block at a time, to enable a later
reimplementation in another language.

Describe what the following block of C code does, in 1-5 sentences, at the level
of purpose and behavior (not line-by-line mechanics). Then list the semantic
invariants a reimplementer must preserve: integer overflow/wraparound behavior,
ownership and lifetime, aliasing assumptions, boundary conditions, error paths
and the state left behind on failure, evaluation-order dependence — and, when
the code parses or serializes data, the EXACT format: byte layout, field order
and widths, terminators, encodings, alignment/padding. Format facts must be in
the invariants; the prose may summarize but the invariants carry the spec.
Never quote C code, expressions, or identifiers in an invariant — express
formats in words and numbers ("the name is followed by one zero byte", "the
count is decimal ASCII digits"), and refer to inputs by role ("the search
key", "the output record"), never by C variable syntax. State each
invariant as OBSERVABLE BEHAVIOR, not implementation mechanism (write "on
growth, prior references to the contents are not preserved", not "realloc
invalidates the old pointer"). Only list invariants that are real and
non-obvious; an empty list is fine for trivial blocks.

Never use C vocabulary anywhere in your reply: no "pointer", "malloc"/"free"/
"realloc", "struct", "null-terminated", "macro"/"#define"/"preprocessor", and no
C library function names. Say "a growable byte sequence" not "a malloc'd char
buffer"; "terminated by a zero byte" not "null-terminated"; "a record with
fields X, Y" not "a struct".

{callee_context}Reply with ONLY a JSON object:
{{"description": "...", "invariants": ["...", ...]}}

CODE (lines {start}-{end}):
```c
{code}
```
"""

CALLEE_CONTEXT = """\
This block calls the following functions, already documented — rely on these
summaries instead of re-deriving them:
{summaries}

"""

MERGE_PROMPT = """\
Two adjacent/related parts of a program have been documented separately. Combine
the two descriptions into ONE description (1-5 sentences) of the combined
behavior, at a level slightly more general than either input — describe what the
whole accomplishes, not the two parts glued together. Do not invent behavior not
implied by the inputs. Do not mention programming-language specifics, and never
use C vocabulary (pointer, malloc/free/realloc, struct, null-terminated, macro,
C library function names) — describe observable behavior instead.

PART A:
{text_a}

PART B:
{text_b}

Reply with ONLY a JSON object: {{"description": "..."}}
"""


class DiffusionStrategy:
    name = "diffusion"

    def __init__(self, cfg: Config, llm: LLM, store: Store):
        self.cfg = cfg
        self.llm = llm
        self.store = store
        self.checker = LockChecker(cfg, llm)
        self.unit_blocks: dict[str, list[str]] = {}   # explanation id -> member seed block ids
        self.block_calls: dict[str, set[str]] = {}    # block id -> callee block ids

    async def run(self, source: str, source_name: str) -> None:
        graph = chunk(source, self.cfg.split_function_over_lines)
        self.store.write_record(graph.to_record())
        self.block_calls = {b.id: set() for b in graph.blocks}
        for a, b in graph.call_edges:
            self.block_calls[a].add(b)

        source_lines = source.split("\n")
        self.source_lines = source_lines
        await self._describe_pass(graph, source_lines)

        for pass_num in range(1, self.cfg.max_merge_passes + 1):
            locked = await self._lock_pass(pass_num)
            merged = await self._merge_pass(pass_num, graph)
            self.store.event("pass_end", pass_num=pass_num, locked=locked, merged=merged,
                             open=len(self.store.open_units()))
            if not self.store.open_units():
                break
            if locked == 0 and merged == 0:
                # fixpoint: nothing can change; remaining open units are irreducible
                for exp in self.store.open_units():
                    exp.status = IRREDUCIBLE
                    exp.lock_failures.append({"stage": "fixpoint", "detail": "no merges or locks possible"})
                    self.store.put(exp)
                break
        else:
            self.store.event("pass_cap_hit", cap=self.cfg.max_merge_passes)
            for exp in self.store.open_units():
                exp.status = IRREDUCIBLE
                exp.lock_failures.append({"stage": "pass_cap", "detail": "merge pass cap reached"})
                self.store.put(exp)

        # terminal invariant: full coverage by locked/irreducible units
        ranges = [r for e in self.store.live_units() for r in e.ranges]
        cov = check_coverage(ranges, len(source_lines), ignorable=blank_lines(source))
        self.store.event("terminal_coverage", ok=cov.ok, detail=cov.describe())
        self.store.render_markdown(source_name)

    # --- pass 0 ---------------------------------------------------------------

    async def _describe_pass(self, graph: SeedGraph, source_lines: list[str]) -> None:
        blocks_by_id = {b.id: b for b in graph.blocks}
        in_scc = {bid: tuple(sorted(g)) for g in graph.scc_groups for bid in g}

        # describe units: SCC groups (>1) are one unit; everything else single
        units: list[list[SeedBlock]] = []
        seen_groups: set[tuple] = set()
        for bid in graph.topo_order:
            if bid in in_scc:
                g = in_scc[bid]
                if g in seen_groups:
                    continue
                seen_groups.add(g)
                units.append([blocks_by_id[b] for b in g])
            else:
                units.append([blocks_by_id[bid]])
        # topo_order may omit gap blocks added after graph construction; append them
        ordered_ids = {b.id for u in units for b in u}
        units.extend([[b] for b in graph.blocks if b.id not in ordered_ids])

        summaries: dict[str, str] = {}  # block id -> description (for callee substitution)

        # process in dependency waves so independent describes run in parallel
        described: set[str] = set()
        pending = units
        while pending:
            ready = [u for u in pending
                     if all(c in described or c in {b.id for b in u}
                            for b in u for c in self.block_calls.get(b.id, ()))]
            if not ready:  # cycle remnants or missing deps; take everything left
                ready = pending
            pending = [u for u in pending if u not in ready]

            async def describe(unit: list[SeedBlock]) -> tuple[list[SeedBlock], dict]:
                start = min(b.start for b in unit)
                end = max(b.end for b in unit)
                code = "\n".join(source_lines[start - 1:end])
                callee_ids = {c for b in unit for c in self.block_calls.get(b.id, ())} - {b.id for b in unit}
                ctx = ""
                lines = [f"- {blocks_by_id[c].function or c}: {summaries[c]}"
                         for c in sorted(callee_ids) if c in summaries]
                if lines:
                    ctx = CALLEE_CONTEXT.format(summaries="\n".join(lines))
                resp = await self.llm.ask_json(DESCRIBE_PROMPT.format(
                    callee_context=ctx, start=start, end=end, code=code))
                return unit, resp

            results = await asyncio.gather(*(describe(u) for u in ready))
            for unit, resp in results:
                exp = Explanation(
                    id=self.store.new_id(),
                    ranges=_merge_ranges([[b.start, b.end] for b in unit]),
                    text=str(resp.get("description", "")).strip(),
                    invariants=[str(i) for i in resp.get("invariants", [])],
                    external_deps=sorted({c for b in unit for c in b.calls_external}),
                    model=self.cfg.worker_model,
                    strategy=self.name,
                    pass_num=0,
                )
                self.store.put(exp)
                self.unit_blocks[exp.id] = [b.id for b in unit]
                for b in unit:
                    summaries[b.id] = exp.text
                    described.add(b.id)
        self.store.event("describe_pass_done", units=len(self.store.explanations))

    # --- lock pass --------------------------------------------------------------

    async def _lock_pass(self, pass_num: int) -> int:
        candidates = [e for e in self.store.open_units() if e.changed_since_lock_attempt]
        # sanitize first: batch-launder regex-detectable C phrasing so it never
        # costs a lock attempt
        await asyncio.gather(*(self.checker.sanitize(e) for e in candidates))
        live = self.store.live_units()

        def siblings_of(exp: Explanation) -> list[str]:
            return [u.text for u in live if u.id != exp.id and u.text]

        results = await asyncio.gather(*(self.checker.check(e, siblings_of(e))
                                         for e in candidates))
        locked = 0
        for exp, res in zip(candidates, results):
            if res.passed:
                exp.status = LOCKED
                exp.pass_num = pass_num
                locked += 1
                self.store.put(exp)
                continue
            exp.lock_attempts += 1
            exp.changed_since_lock_attempt = False
            exp.lock_failures.append({"stage": res.stage, "detail": res.detail})

            # localized failure -> try a behavioral rephrase (counts as a change)
            if res.offending_invariant is not None:
                inv = res.offending_invariant
                used = exp.invariant_rephrases.get(inv, 0)
                if used < self.cfg.max_invariant_rephrases:
                    rephrased = await self.checker.rephrase_invariant(inv)
                    exp.invariant_rephrases[inv] = used + 1
                    if rephrased is not None:
                        exp.invariants = [rephrased if i == inv else i for i in exp.invariants]
                        exp.changed_since_lock_attempt = True  # earns a fresh attempt
                        exp.lock_attempts = 0  # change resets staleness
                        self.store.put(exp)
                        continue
            elif res.offending_description:
                used = exp.invariant_rephrases.get("__description__", 0)
                if used < self.cfg.max_invariant_rephrases:
                    rephrased = await self.checker.rephrase_description(exp.text)
                    exp.invariant_rephrases["__description__"] = used + 1
                    if rephrased is not None:
                        exp.text = rephrased
                        exp.changed_since_lock_attempt = True
                        exp.lock_attempts = 0  # change resets staleness
                        self.store.put(exp)
                        continue
            elif res.questions:
                # underspecified: answer the probe's questions from the source
                used = exp.invariant_rephrases.get("__repair__", 0)
                if used < self.cfg.max_repairs:
                    new_invs = await self.checker.repair(exp, self.source_lines, res.questions)
                    exp.invariant_rephrases["__repair__"] = used + 1
                    if new_invs:
                        exp.invariants = exp.invariants + new_invs
                        await self.checker.sanitize(exp)
                        exp.changed_since_lock_attempt = True
                        exp.lock_attempts = 0
                        self.store.put(exp)
                        continue

            # a failed attempt whose failure handling produced no change leaves the
            # unit progressable only by merge; after staleness_limit total failed
            # attempts we stop waiting for merges to save it
            stale = (not exp.changed_since_lock_attempt
                     and exp.lock_attempts >= self.cfg.staleness_limit)
            oversized = exp.total_lines() > self.cfg.size_guard_lines
            if stale or oversized:
                exp.status = IRREDUCIBLE
                exp.lock_failures.append({
                    "stage": "policy",
                    "detail": "size guard" if oversized else "staleness limit",
                })
            self.store.put(exp)
        return locked

    # --- merge pass ---------------------------------------------------------------

    async def _merge_pass(self, pass_num: int, graph: SeedGraph) -> int:
        # units awaiting a lock re-check (just described or rephrased) are not
        # merge candidates — every change earns its lock attempt before the
        # unit can be swallowed by a merge
        open_units = [e for e in self.store.open_units()
                      if not e.changed_since_lock_attempt]
        if len(open_units) < 2:
            return 0

        block_owner = {bid: eid for eid, bids in self.unit_blocks.items()
                       for bid in bids if eid in {e.id for e in open_units}}

        # call-DAG candidates first, then adjacency
        pairs: list[tuple[str, str]] = []
        seen_pairs: set[frozenset] = set()
        for a, b in graph.call_edges:
            ea, eb = block_owner.get(a), block_owner.get(b)
            if ea and eb and ea != eb and frozenset((ea, eb)) not in seen_pairs:
                seen_pairs.add(frozenset((ea, eb)))
                pairs.append((ea, eb))
        by_start = sorted(open_units, key=Explanation.sort_key)
        for u, v in zip(by_start, by_start[1:]):
            if _ranges_adjacent(u.ranges, v.ranges) and frozenset((u.id, v.id)) not in seen_pairs:
                seen_pairs.add(frozenset((u.id, v.id)))
                pairs.append((u.id, v.id))

        # greedy: each unit participates in at most one merge per pass;
        # skip pairs whose combined size would blow the size guard
        exps = {e.id: e for e in open_units}
        taken: set[str] = set()
        selected: list[tuple[Explanation, Explanation]] = []
        for ida, idb in pairs:
            if ida in taken or idb in taken:
                continue
            a, b = exps[ida], exps[idb]
            if a.total_lines() + b.total_lines() > self.cfg.merge_soft_cap_lines:
                continue
            taken.update((ida, idb))
            selected.append((a, b))

        if not selected:
            return 0

        async def merge(a: Explanation, b: Explanation) -> tuple[Explanation, Explanation, dict]:
            resp = await self.llm.ask_json(MERGE_PROMPT.format(
                text_a=_render_unit(a), text_b=_render_unit(b)))
            return a, b, resp

        results = await asyncio.gather(*(merge(a, b) for a, b in selected))
        merged = 0
        for a, b, resp in results:
            invariants = list(dict.fromkeys(a.invariants + b.invariants))  # verbatim union, deduped
            exp = Explanation(
                id=self.store.new_id(),
                ranges=_merge_ranges(a.ranges + b.ranges),
                text=str(resp.get("description", "")).strip(),
                invariants=invariants,
                external_deps=sorted(set(a.external_deps + b.external_deps)),
                model=self.cfg.worker_model,
                strategy=self.name,
                pass_num=pass_num,
                parent_ids=[a.id, b.id],
            )
            self.unit_blocks[exp.id] = self.unit_blocks.get(a.id, []) + self.unit_blocks.get(b.id, [])
            # parents leave the working set (their status stays as-is in history;
            # live_units() excludes anything referenced as a parent)
            a.status = b.status = "merged"
            self.store.put(a)
            self.store.put(b)
            self.store.put(exp)
            merged += 1
        return merged


def _render_unit(e: Explanation) -> str:
    inv = "\n".join(f"- {i}" for i in e.invariants) or "(none)"
    return f"{e.text}\nInvariants:\n{inv}"


def _merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    """Union of inclusive ranges, coalescing overlapping/adjacent ones."""
    out: list[list[int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def _ranges_adjacent(a: list[list[int]], b: list[list[int]]) -> bool:
    return any(abs_gap(ra, rb) <= 1 for ra in a for rb in b)


def abs_gap(ra: list[int], rb: list[int]) -> int:
    if ra[1] < rb[0]:
        return rb[0] - ra[1]
    if rb[1] < ra[0]:
        return ra[0] - rb[1]
    return 0
