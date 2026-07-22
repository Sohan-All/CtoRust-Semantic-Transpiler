"""Strategy B — whole-file direct MTU extraction.

One call proposes the full partition; the coverage validator converts silent
omission into detectable, retryable failure. Each proposed MTU then goes through
the same lock-check stack as Strategy A (failures -> irreducible; no merging
machinery here).
"""

from __future__ import annotations

import asyncio

from chunker import chunk
from config import Config
from llm import LLM
from lock_check import LockChecker
from state import Explanation, Store, LOCKED, IRREDUCIBLE
from validator import check_coverage, blank_lines

EXTRACT_PROMPT = """\
Partition the following C source file into Minimum Translation Units (MTUs):
contiguous chunks of code, each encompassing one operation or task that can be
described WITHOUT reference to C-specific constructs (no pointers, manual
allocation, preprocessor, or C standard library specifics — describe observable
behavior instead).

Rules:
- Every line of the file must belong to exactly one MTU: no gaps, no overlaps.
- Each MTU gets a 1-5 sentence language-agnostic description of its behavior.
- Each MTU gets a list of semantic invariants a reimplementer must preserve
  (overflow behavior, ownership/lifetime, aliasing, boundary conditions, error
  paths), each stated as observable behavior, not mechanism.
- Invariants describe OBSERVABLE BEHAVIOR and CONTRACTS, never the C
  implementation's mechanism. Memory management (allocate, free, copy,
  capacity growth, refcounts), and the idioms that implement it (manual
  buffer sizing, string duplication, ensure-capacity helpers), are
  MECHANISM — do not state them as invariants. State what must be TRUE
  instead: what is stored, what is owned, what remains valid, what the
  caller may rely on afterwards.
- When the code
  parses or serializes data, the invariants must carry the EXACT format: byte
  layout, field order and widths, terminators, encodings — expressed in words
  and numbers, never by quoting C code, expressions, or variable names. Empty
  list is fine for trivial units.
- Prefer semantically coherent units (a data structure plus its operations can
  be one MTU) over mechanical per-function splits, but never let one MTU exceed
  ~{size_guard} lines.
- Fold file-header comments and include directives into the first behavioral
  MTU — they are not MTUs of their own.
- Never use C vocabulary in descriptions or invariants: no "pointer",
  "malloc"/"free"/"realloc", "struct", "null-terminated", "macro", or C library
  function names. Say "a growable byte sequence" not "a malloc'd char buffer";
  "terminated by a zero byte" not "null-terminated".

Reply with ONLY a JSON object:
{{"mtus": [{{"start": <int>, "end": <int>, "description": "...", "invariants": ["..."]}}, ...]}}

The file has {n_lines} lines. SOURCE (line numbers prefixed):
```c
{numbered_source}
```
"""

RETRY_SUFFIX = """

Your previous partition was invalid: {violations}.
Produce a corrected partition. Every line from 1 to {n_lines} must be covered
exactly once (blank separator lines may be attached to either neighbor).
"""


class WholeFileStrategy:
    name = "whole_file"

    def __init__(self, cfg: Config, llm: LLM, store: Store):
        self.cfg = cfg
        self.llm = llm
        self.store = store
        self.checker = LockChecker(cfg, llm)

    async def run(self, source: str, source_name: str) -> None:
        source_lines = source.split("\n")
        n_lines = len(source_lines)
        if n_lines > self.cfg.whole_file_max_lines:
            self.store.event("file_too_large", n_lines=n_lines,
                             limit=self.cfg.whole_file_max_lines)
            raise SystemExit(
                f"{source_name}: {n_lines} lines exceeds whole_file limit "
                f"({self.cfg.whole_file_max_lines}); use --strategy diffusion")

        numbered = "\n".join(f"{i + 1:4d}| {line}" for i, line in enumerate(source_lines))
        prompt = EXTRACT_PROMPT.format(size_guard=self.cfg.size_guard_lines,
                                       n_lines=n_lines, numbered_source=numbered)
        ignorable = blank_lines(source)

        extract_budget = 16000 if n_lines > 500 else 8000
        mtus = None
        for attempt in range(self.cfg.whole_file_max_retries + 1):
            resp = await self.llm.ask_json(prompt, max_tokens=extract_budget)
            proposed = resp.get("mtus", []) if isinstance(resp, dict) else []
            ranges = [[int(m.get("start", 0)), int(m.get("end", -1))] for m in proposed]
            cov = check_coverage(ranges, n_lines, ignorable=ignorable)
            self.store.event("extract_attempt", attempt=attempt, ok=cov.ok,
                             n_mtus=len(proposed), detail=cov.describe())
            if cov.ok:
                mtus = proposed
                break
            prompt = EXTRACT_PROMPT.format(size_guard=self.cfg.size_guard_lines,
                                           n_lines=n_lines, numbered_source=numbered)
            prompt += RETRY_SUFFIX.format(violations=cov.describe(), n_lines=n_lines)

        if mtus is None:
            self.store.event("extract_failed", retries=self.cfg.whole_file_max_retries)
            raise SystemExit(f"{source_name}: no valid partition after "
                             f"{self.cfg.whole_file_max_retries} retries (A/B data point; "
                             "see state.jsonl)")

        # deterministic chunker pass purely to annotate external dependencies
        # (calls to functions not defined in this file) per line range
        seed_blocks = chunk(source, self.cfg.split_function_over_lines).blocks

        def deps_for(start: int, end: int) -> list[str]:
            return sorted({c for b in seed_blocks
                           if b.start <= end and b.end >= start
                           for c in b.calls_external})

        exps = []
        for m in sorted(mtus, key=lambda m: int(m["start"])):
            start, end = int(m["start"]), int(m["end"])
            exp = Explanation(
                id=self.store.new_id(),
                ranges=[[start, end]],
                text=str(m.get("description", "")).strip(),
                invariants=[str(i) for i in m.get("invariants", [])],
                external_deps=deps_for(start, end),
                model=self.cfg.worker_model,
                strategy=self.name,
            )
            exps.append(exp)
            self.store.put(exp)

        # same lock-check stack as strategy A; failures are irreducible directly
        self.source_lines = source_lines
        all_texts = {e.id: e.text for e in exps}
        results = await asyncio.gather(*(
            self._lock_with_rephrase(e, [t for i, t in all_texts.items() if i != e.id])
            for e in exps))
        for exp, passed in zip(exps, results):
            exp.status = LOCKED if passed else IRREDUCIBLE
            self.store.put(exp)

        self.store.event("done", locked=sum(results), irreducible=len(exps) - sum(results))
        self.store.render_markdown(source_name)

    async def _lock_with_rephrase(self, exp: Explanation, siblings: list[str]) -> bool:
        """Lock check with the invariant-rephrase loop (no merging in this strategy)."""
        await self.checker.sanitize(exp)  # batch-launder regex-detectable C phrasing
        for _ in range(1 + self.cfg.max_invariant_rephrases + self.cfg.max_repairs):
            res = await self.checker.check(exp, siblings)
            exp.lock_attempts += 1
            if res.passed:
                return True
            exp.lock_failures.append({"stage": res.stage, "detail": res.detail})
            if res.offending_invariant is not None:
                inv = res.offending_invariant
                used = exp.invariant_rephrases.get(inv, 0)
                if used >= self.cfg.max_invariant_rephrases:
                    return False
                rephrased = await self.checker.rephrase_invariant(inv)
                exp.invariant_rephrases[inv] = used + 1
                if rephrased is None:
                    return False
                exp.invariants = [rephrased if i == inv else i for i in exp.invariants]
            elif res.offending_description:
                used = exp.invariant_rephrases.get("__description__", 0)
                if used >= self.cfg.max_invariant_rephrases:
                    return False
                rephrased = await self.checker.rephrase_description(exp.text)
                exp.invariant_rephrases["__description__"] = used + 1
                if rephrased is None:
                    return False
                exp.text = rephrased
            elif res.questions:
                # underspecified: answer the probe's questions from the source
                used = exp.invariant_rephrases.get("__repair__", 0)
                if used >= self.cfg.max_repairs:
                    return False
                new_invs = await self.checker.repair(exp, self.source_lines, res.questions)
                exp.invariant_rephrases["__repair__"] = used + 1
                if not new_invs:
                    return False
                exp.invariants = exp.invariants + new_invs
                await self.checker.sanitize(exp)
            else:
                return False
        return False
