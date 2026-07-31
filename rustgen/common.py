"""Shared helpers for the rustgen stages."""

from __future__ import annotations

import asyncio
import re
import traceback
from typing import Awaitable, Callable

from state import Explanation

# Never swallowed by gather_units: these mean "stop", not "this unit failed".
# CancelledError is a BaseException in 3.12 and `return_exceptions=True` would
# happily capture it, turning Ctrl-C into a silently degraded run.
_FATAL = (asyncio.CancelledError, KeyboardInterrupt, SystemExit)


async def gather_units(
    stage: str,
    units: list[Explanation],
    run: Callable[[Explanation], Awaitable[tuple[str, object]]],
    failures: list[dict] | None = None,
    on_result: Callable[[str, object], None] | None = None,
) -> dict:
    """Run `run(unit)` for every unit in parallel, surviving individual failures.

    Replaces a bare `asyncio.gather`, which is fail-fast: one unit raising
    (a malformed reply that outlived its retries, a connection reset, an empty
    choices list) aborted the whole stage and threw away every sibling that had
    already succeeded. One observed run died at stage S because a single MTU's
    JSON would not parse, discarding seventeen good specs.

    Two guarantees:

    - a unit that raises is recorded in `failures` and omitted from the result,
      leaving the rest of the stage intact;
    - `on_result` fires the moment a unit lands, not after the barrier, so a
      later crash cannot un-persist work that already completed.

    A degraded stage is NOT a normal one: `failures` is what run_project turns
    into a `degraded` record, and a run carrying one scores nothing. Silently
    returning a short dict here would recreate the `todo!()` hole — a crate
    that builds, looks clean, and is missing behaviour.
    """

    async def one(u: Explanation):
        try:
            uid, value = await run(u)
        except _FATAL:
            raise
        except Exception as e:
            if failures is not None:
                failures.append({
                    "stage": stage,
                    "unit": u.id,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=8),
                })
            print(f"[{stage}] {u.id}: FAILED ({type(e).__name__}: {e}) "
                  f"— unit dropped, run marked degraded")
            return None
        if on_result is not None:
            on_result(uid, value)
        return uid, value

    done = await asyncio.gather(*(one(u) for u in units))
    return {uid: value for uid, value in (d for d in done if d is not None)}


def unbalanced_delimiters(code: str) -> str:
    """"" if the code's delimiters balance, else a description of the problem.

    Cheap structural sanity, not a parser. A section whose braces do not balance
    is a PARSE error, and rustc reports exactly one of those for the whole crate
    no matter how much else is wrong — so the compile loop sees "1 error", tries
    a repair, sees "1 error" again, scores it as no improvement and reverts.
    Observed pinning runs at `final: 1` for five rounds (an `impl Scheduler {}`
    followed by orphan doc comments and a stray `}`), and masking a dozen real
    type errors behind it. Catch it before it reaches the loop.

    String and char literals, comments, and lifetimes (`&'a T`) are skipped, so
    a brace inside a string does not count.
    """
    stack, i, n = [], 0, len(code)
    pairs = {")": "(", "]": "[", "}": "{"}
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            i = code.find("\n", i)
            if i == -1:
                break
        elif c == "/" and nxt == "*":
            j = code.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        elif c == '"':
            # r"..." / r#"..."# raw strings end at the matching hash count
            hashes = 0
            if i and code[i - 1] == "r":
                k = i - 2
                while k >= 0 and code[k] == "#":
                    hashes += 1
                    k -= 1
            if hashes:
                j = code.find('"' + "#" * hashes, i + 1)
                i = n if j == -1 else j + 1 + hashes
                continue
            i += 1
            while i < n and code[i] != '"':
                i += 2 if code[i] == "\\" else 1
        elif c == "'":
            # char literal vs lifetime (&'a T). Escaped first ('\n', '\''),
            # then any single character ('}' must not count as a closer), then
            # the identifier scan — which lands on a closing quote for 'ab' and
            # on something else for a lifetime, where there is nothing to skip.
            if code[i + 1:i + 2] == "\\":
                j = i + 2
                while j < n and code[j] != "'":
                    j += 1
                i = j
            elif code[i + 2:i + 3] == "'":
                i += 2
            else:
                j = i + 1
                while j < n and (code[j].isalnum() or code[j] == "_"):
                    j += 1
                i = j if (j < n and code[j] == "'") else j - 1
        elif c in "([{":
            stack.append((c, i))
        elif c in ")]}":
            if not stack:
                line = code.count("\n", 0, i) + 1
                return f"stray closing {c!r} at line {line} with nothing open"
            if stack[-1][0] != pairs[c]:
                line = code.count("\n", 0, i) + 1
                return (f"mismatched {c!r} at line {line}, expected closer for "
                        f"{stack[-1][0]!r}")
            stack.pop()
        i += 1
    if stack:
        opener, pos = stack[0]
        return (f"{len(stack)} unclosed {opener!r} (first at line "
                f"{code.count(chr(10), 0, pos) + 1})")
    return ""


def parse_regression(old: str, new: str) -> str:
    """"" if `new` is an acceptable replacement for `old`, else the problem.

    The rule a repair must satisfy: do not INTRODUCE a parse error. An
    unbalanced section is not an ordinary bad edit — rustc reports exactly one
    parse error for the whole crate however much else is wrong, so the compile
    loop reads "1 error", repairs, reads "1 error", scores no improvement and
    reverts, forever. Trial 4's `base_srvA` pinned at `final: 1` this way.

    Deliberately asymmetric: a section that is ALREADY unbalanced stays
    writable. The guard exists to stop a good section being broken, not to
    freeze a broken one out of reach of the repair that would fix it.
    """
    problem = unbalanced_delimiters(new)
    if problem and not unbalanced_delimiters(old):
        return problem
    return ""


_BLOCK = re.compile(r"\b(impl|trait)\b[^{;]*\{(.*?)\n\}", re.S)
_ITEM = re.compile(r"\b(fn|const|type|struct|enum)\b")
_FN_NAME = re.compile(r"(?:^|[\s;{}])fn\s+(\w+)", re.M)


def _empty_blocks(code: str) -> list[str]:
    """`impl`/`trait` blocks whose body holds no item — only comments."""
    out = []
    for kw, body in _BLOCK.findall(_blank_literals(code)):
        live = "\n".join(l for l in body.split("\n")
                         if l.strip() and not l.strip().startswith("//"))
        if not _ITEM.search(live):
            out.append(kw)
    return out


def emptied_blocks(old: str, new: str) -> str:
    """"" unless `new` gutted an `impl`/`trait` block that `old` had populated.

    The second repair-damage mode, found by sweeping every recorded repair
    after the 2026-07-30 batch. A repair deletes the method bodies from an
    `impl` block and leaves their doc comments behind:

        impl Scheduler {
            /// Spawns a new task into the scheduler.
            /// Dispatches the next available task ...
            /// Peeks at the next available task ...
        }

    Braces still balance, so the parse guard accepts it — `binary_heap
    base_srvB_t1` went 1431 bytes -> 491 and lost `spawn`, `dispatch` and
    `peek` this way, then BUILD_FAILED on the orphaned docs. It is only that
    loud by luck: without the stray docs the same deletion surfaces as
    unresolved names, or as a crate that builds with the behaviour missing.

    Counts blocks rather than comparing function names, because the obvious
    name-based rules do not survive contact with the data. "Lost a fn" fires on
    9.4% of all repairs and includes legitimate `#[derive]` substitutions;
    "lost a fn defined nowhere else" misses this very case, because an
    unresolved `*_deps` stub elsewhere counts as a definition. Emptying a block
    that had items is unambiguous: 18 of 502 recorded transitions, every one of
    them real damage.
    """
    before, after = _empty_blocks(old), _empty_blocks(new)
    if len(after) > len(before):
        dropped = (set(_FN_NAME.findall(_blank_literals(old)))
                   - set(_FN_NAME.findall(_blank_literals(new))))
        return (f"repair emptied {len(after) - len(before)} impl/trait "
                f"block(s) of every item, keeping only comments"
                + (f" (lost: {', '.join(sorted(dropped))})" if dropped else ""))
    return ""


# A deduplication exemption was tried here and REVERTED — recorded because the
# reasoning looked sound and was wrong twice over.
#
# `binary_heap base_srvB_t2` stalled with nine rejections and zero accepted
# repairs, which read like this guard blocking the only correct fix for E0592
# ("duplicate definitions for `compare`"). The proposed exemption: permit the
# emptying when every dropped fn still exists elsewhere in the crate.
#
# Wrong on the facts. The two `compare`s were not peers — the shared-types copy
# was `todo!()` and the unit's copy was the real implementation. The model was
# proposing to delete the REAL one and keep the stub; the guard was right to
# refuse. The stall's cause was upstream (stage T shipping a stubbed `impl`
# block after exhausting its retries), not this check.
#
# Wrong on the mechanics too, and independently: `spawn`/`dispatch`/`peek` in
# the base_srvB_t1 gutting all "existed elsewhere" as deps stubs, so the
# exemption permitted a genuine gutting. Restricting it to non-stub definitions
# then flipped the other case, because the surviving copy there WAS the stub.
#
# Both directions were caught only by testing against both fixtures at once.
# Do not reintroduce this without doing the same.


def remaining_stubs(code: str) -> str:
    """"" unless the code contains ANY `todo!`/`unimplemented!`.

    Stricter than `illegal_stubs`, and for a different question. Mid-pipeline
    the documented form `todo!("<why>")` is a deliberate escape valve, so the
    stage gates permit it. In a crate about to be SCORED it is not an escape
    valve, it is a runtime panic: `array_list base_srvA_t1` shipped
    `todo!("<populate_atlas>")` — which satisfies the letter of the rule while
    naming no reason at all — and panicked 5 of its 26 cases. `illegal_stubs`
    called that crate clean.

    A panic is not a translation-quality signal, so a crate carrying one must
    not enter the record as an ordinary verdict. Same rule as the original
    `todo!()` hole, applied to the form that hole's fix deliberately exempted.
    """
    hits = []
    for i, line in enumerate(_blank_literals(code).split("\n"), 1):
        if re.search(r"\b(todo|unimplemented)!", line):
            hits.append(f"line {i}")
    if not hits:
        return ""
    shown = ", ".join(hits[:6]) + (" ..." if len(hits) > 6 else "")
    return f"{len(hits)} stub(s) reachable at run time: {shown}"


def _blank_literals(code: str) -> str:
    """`code` with comment bodies and string/char literal *interiors* replaced
    by spaces. Length, newlines and therefore every offset and line number are
    preserved; the delimiters themselves stay put.

    Lets brace matching and item searches run over the code's real structure:
    a `{` inside a string, or a `'}'` char literal, must not count as a brace,
    and a commented-out `todo!()` must not count as a stub.
    """
    out = list(code)
    i, n = 0, len(code)

    def blank(a: int, b: int) -> None:
        """Spaces over [a, b), newlines left alone so line numbers hold."""
        for k in range(max(a, 0), min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = code.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
        elif c == "/" and nxt == "*":
            j = code.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
        elif c == '"':
            # r"..." / r#"..."# end at the matching hash count
            hashes = 0
            if i and code[i - 1] == "r":
                k = i - 2
                while k >= 0 and code[k] == "#":
                    hashes += 1
                    k -= 1
            if hashes:
                j = code.find('"' + "#" * hashes, i + 1)
                if j == -1:
                    blank(i + 1, n)
                    i = n
                else:
                    blank(i + 1, j)
                    i = j + 1 + hashes
                continue
            j = i + 1
            while j < n and code[j] != '"':
                j += 2 if code[j] == "\\" else 1
            blank(i + 1, j)
            i = min(j + 1, n)
        elif c == "'":
            # char literal vs lifetime (&'a T) — same disambiguation as
            # unbalanced_delimiters: escaped first, then any single character,
            # then the identifier scan, which lands on a closing quote for a
            # literal and on something else for a lifetime.
            if code[i + 1:i + 2] == "\\":
                j = i + 2
                while j < n and code[j] != "'":
                    j += 1
                blank(i + 1, j)
                i = min(j + 1, n)
            elif code[i + 2:i + 3] == "'":
                blank(i + 1, i + 2)
                i += 3
            else:
                j = i + 1
                while j < n and (code[j].isalnum() or code[j] == "_"):
                    j += 1
                if j < n and code[j] == "'":
                    blank(i + 1, j)
                    i = j + 1
                else:
                    i = j                      # lifetime: nothing to blank
        else:
            i += 1
    return "".join(out)


_STUB_CALL = re.compile(r"\b(todo|unimplemented)\s*!\s*\(\s*\)")
_DEPS_MOD = re.compile(r"\bmod\s+(\w*_deps)\s*\{")


def illegal_stubs(code: str) -> str:
    """"" if the code carries no unexplained stub body, else a description.

    `todo!()` and `unimplemented!()` TYPE-CHECK — they coerce to `!`, so cargo
    check stays green and every error-driven repair round sees a clean crate.
    They fail at RUN time, as a panic, which is how a stubbed crate passes a
    build gate and then diverges on every case that reaches it. One observed
    run shipped a stage-T `impl DraftEditor` whose seven methods were all
    `todo!()`: the crate built, reported "MTU sections clean: 17/17", and 20 of
    26 differential cases panicked at the same line.

    Two forms are legitimate and are NOT reported:

      - anything inside a `*_deps` module — stage T is told to stub external
        domain functions there, and sibling_deps resolves them afterwards;
      - `todo!("<why>")` carrying a message — an honest, documented gap, which
        the stub-repair prompt explicitly permits to survive.

    A bare `todo!()` anywhere else is neither: it is a unit (or, worse, the
    shared types section) silently deferring to a sibling that never
    implements the capability. Nothing downstream fills it in.
    """
    blanked = _blank_literals(code)

    # brace-matched extent of every deps module — the one legal home for stubs
    spans: list[tuple[int, int]] = []
    for m in _DEPS_MOD.finditer(blanked):
        depth, k, n = 0, m.end() - 1, len(blanked)
        while k < n:
            if blanked[k] == "{":
                depth += 1
            elif blanked[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        spans.append((m.start(), k))

    hits = [f"{m.group(1)}!() at line {blanked.count(chr(10), 0, m.start()) + 1}"
            for m in _STUB_CALL.finditer(blanked)
            if not any(a <= m.start() <= b for a, b in spans)]
    if not hits:
        return ""
    shown = ", ".join(hits[:6]) + (" ..." if len(hits) > 6 else "")
    return f"{len(hits)} unexplained stub body/bodies: {shown}"


def extract_rust(text: str) -> str:
    """Pull Rust source out of a model reply (```rust fence, or the whole
    reply if unfenced). When the model emits several fences (draft + revised
    version), the LAST one is taken — concatenating them duplicates items.
    Translator narration comments are stripped from the result."""
    fences = re.findall(r"```(?:rust)?\s*\n(.*?)```", text, re.DOTALL)
    if fences:
        return strip_narration(fences[-1].strip())
    # unterminated fence (reply truncated at max_tokens): take what follows the
    # last opener rather than leaking the ``` marker into the source file
    openers = list(re.finditer(r"```(?:rust)?\s*\n", text))
    if openers:
        return strip_narration(text[openers[-1].end():].strip())
    return strip_narration(text.strip())


# Comment blocks where any line contains one of these are translator
# narration (the model talking about its prompt), not code documentation.
_NARRATION_PHRASES = (
    "the spec", "design spec", "spec says", "per instructions",
    "the instructions", "as requested", "compiler error", "authority rule",
    "sibling function", "sibling module", "sibling signature", "siblings",
    "must also implement", "must not implement", "owned by another unit",
    "owned by other units", "behavioral unit", "the mtu", "based on the",
    "sibling list", "compilation error", "external calls",
    "behavior description", "in idiomatic rust", "in rust,", "raii",
    "to adhere to", "to satisfy the", "resolve the conflict",
    "resolve the collision", "the implementer", "however,", "the contract",
    "the invariant mapping", "the design", "call sites (like",
)

# Lines/blocks starting with these survive untouched (assembly markers,
# invariant citations, contract headers, unsafe-justification docs).
_KEEP_PREFIXES = ("// invariant", "// =====", "// -----", "// low confidence",
                  "// missing error variant", "// api contract", "// safety")


def demote_dangling_docs(rust: str) -> str:
    """Turn `///` comments that document nothing into plain `//` comments.

    Stage T likes to answer "no external functions to stub here" as a doc
    comment inside an otherwise empty `pub mod <stem>_deps { }`, which is a
    hard rustc error (E0585, "expected item after doc comment"). The compile
    loop cannot repair it — its writable set is MTU sections plus the types
    block, and a deps module is neither, so the run reports `MTU sections
    clean: 14/14` and `final: 1` forever and the crate never builds. This was
    the failure mode of both arms of the first ablation attempt.

    Only `///` is affected. `//!` is an inner doc comment and is legal with no
    item after it, so it is left alone.
    """
    lines = rust.split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("///"):
            i += 1
            continue
        run_end = i                     # contiguous run of /// lines
        while (run_end + 1 < len(lines)
               and lines[run_end + 1].lstrip().startswith("///")):
            run_end += 1
        nxt = next((l for l in lines[run_end + 1:] if l.strip()), None)
        # nothing after it, or only a closing brace: it documents nothing
        if nxt is None or nxt.lstrip().startswith("}"):
            for j in range(i, run_end + 1):
                lines[j] = lines[j].replace("///", "//", 1)
        i = run_end + 1
    return "\n".join(lines)


def strip_narration(code: str) -> str:
    """Drop plain `//` comment blocks that narrate the translation (references
    to the spec, siblings, instructions, compiler errors, Rust tutorials)
    instead of documenting the code. `///`/`//!` docs, `// invariant:` lines,
    and assembly/contract markers are always kept. Deterministic, no LLM."""
    out: list[str] = []
    block: list[str] = []  # pending run of plain // comment lines

    def flush() -> None:
        if not block:
            return
        if block[0].strip().lower().startswith("// api contract"):
            out.extend(block)  # contract block: numbered lines are content
        else:
            plain = [l for l in block
                     if not l.strip().lower().startswith(_KEEP_PREFIXES)]
            text = " ".join(l.strip().lower() for l in plain)
            if any(p in text for p in _NARRATION_PHRASES):
                # narration: keep only the marker lines (invariants etc.)
                out.extend(l for l in block if l not in plain)
            else:
                out.extend(block)
        block.clear()

    for line in code.split("\n"):
        s = line.strip()
        if (s.startswith("//") and not s.startswith("///")
                and not s.startswith("//!")):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    # collapse blank runs the removals leave behind
    collapsed: list[str] = []
    for line in out:
        if line.strip() == "" and collapsed and collapsed[-1].strip() == "":
            continue
        collapsed.append(line)
    return "\n".join(collapsed)


_SPEC_SECTIONS = [
    ("signatures", "Signature(s) — implement exactly these"),
    ("ownership", "Ownership / borrowing decisions"),
    ("error_mapping", "Error mapping"),
    ("invariant_obligations", "Per-invariant obligations"),
    ("idioms", "Idioms to use"),
    ("owns", "This unit MUST also implement"),
    ("must_not_implement", "NEVER implement (owned by other units — call them)"),
    # the implementer must honour the mapping the spec committed to: an entry
    # saying a C function was inlined somewhere means that destination has to
    # carry its behaviour and its output text.
    ("symbol_map", "C symbol -> Rust destination (account for every one)"),
]


def render_spec(spec: dict) -> str:
    """Render a stage-S spec (thin or rich) for the codegen/repair prompts."""
    parts: list[str] = []
    for key, title in _SPEC_SECTIONS:
        val = spec.get(key)
        if not val:
            continue
        parts.append(f"{title}:")
        if isinstance(val, list):
            parts.extend(f"  - {v}" for v in val)
        else:
            parts.append(f"  {val}")
    if spec.get("behavior_note"):
        parts.append(f"Note: {spec['behavior_note']}")
    return "\n".join(parts) or "(no spec — choose an idiomatic design)"


def unit_block(exp: Explanation, extra: str = "") -> str:
    """Render one MTU's description + invariants for inclusion in a prompt.
    `extra` (from unit_extras) appends caller/C-source context."""
    inv = "\n".join(f"  - {i}" for i in exp.invariants) or "  (none)"
    ranges = ", ".join(f"{s}-{e}" for s, e in exp.ranges)
    deps = f"\nExternal calls: {', '.join(exp.external_deps)}" if exp.external_deps else ""
    return (f"[{exp.id}] (C lines {ranges}, {exp.status})\n"
            f"Behavior: {exp.text}\nInvariants:\n{inv}{deps}{extra}")


def unit_extras(units: list[Explanation], source: str, split_over: int,
                c_context: str = "off",
                external_callers: dict[str, list[str]] | None = None,
                external_call_texts: dict[str, list[str]] | None = None,
                exit_status: dict | None = None,
                call_sites: bool = True, symbol_map: bool = True,
                output_formats: bool = True) -> dict[str, str]:
    """Per-unit prompt additions, keyed by unit id.

    Always: the REVERSE call graph — for each C function a unit defines,
    which other units (and, in project mode, which sibling files) call it.
    Callers are the reason a unit's output must be reachable: a unit that
    knows who calls it designs signatures they can call and doesn't get
    orphaned as dead code.

    `c_context` (cfg.rustgen_c_source_context): "off" — nothing further;
    "literals" — just the string/char/numeric literals from the unit's C
    lines (data fidelity, no control flow or API shape); "full" — the raw
    C lines, labeled reference-only.

    `external_callers`: C function name -> sibling files that reference it
    (from the project index).

    `external_call_texts`: C function name -> the call expressions SIBLING
    FILES use to call it (project_call_texts in run_project). Without these a
    cross-file callee sees only "(sibling file cli.c)" — the caller's name but
    not its arguments — which is half the call graph in a multi-file project.

    `exit_status`: C function name -> ExitStatus, for functions whose return
    value becomes the process exit status (project_exit_status in run_project).
    A unit cannot derive this: `run_script`'s C returns to `cli_run`, not to
    `main`, so without being told it has no way to know its return value is
    observable output — which is exactly how a distinct exit code 2 got
    flattened to 1 while the same rule landed fine in `cli_run`.

    `call_sites`/`symbol_map`/`output_formats` are ablation switches (see
    Config); False drops that block from every unit's prompt."""
    from chunker import chunk

    graph = chunk(source, split_over)
    lines = source.split("\n")

    def unit_of(start: int, end: int) -> Explanation | None:
        """Unit with the largest line overlap with [start, end] — chunker
        blocks absorb adjacent blank lines, so exact range match is too
        strict."""
        best, best_ov = None, 0
        for u in units:
            ov = sum(max(0, min(e, end) - max(s, start) + 1)
                     for s, e in u.ranges)
            if ov > best_ov:
                best, best_ov = u, ov
        return best

    defined_in: dict[str, str] = {}      # C function -> unit id
    fn_unit: dict[str, str] = {}
    for b in graph.blocks:
        if b.function and b.function not in defined_in:
            u = unit_of(b.start, b.end)
            if u:
                defined_in[b.function] = u.id
                fn_unit[b.function] = u.id

    # callee unit -> {callee fn -> set of caller descriptions}
    callers: dict[str, dict[str, set[str]]] = {}
    # callee unit -> {callee fn -> set of distinct call expressions}
    call_texts: dict[str, dict[str, set[str]]] = {}
    for b in graph.blocks:
        if not b.function:
            continue
        caller_unit = fn_unit.get(b.function)
        for callee in getattr(b, "calls_internal", []) or []:
            target = defined_in.get(callee)
            if target and target != caller_unit:
                callers.setdefault(target, {}).setdefault(callee, set()).add(
                    f"{b.function} [{caller_unit}]" if caller_unit else b.function)
                for text in (getattr(b, "call_sites", {}) or {}).get(callee, []):
                    call_texts.setdefault(target, {}).setdefault(
                        callee, set()).add(text)
    # sibling-file call sites for the functions THIS file defines. Keyed by
    # C name, and defined_in only holds names defined here, so a same-named
    # static in another file cannot attach its call texts to this unit.
    for fn, texts in (external_call_texts or {}).items():
        target = defined_in.get(fn)
        if target:
            call_texts.setdefault(target, {}).setdefault(fn, set()).update(texts)
    for fn, files in (external_callers or {}).items():
        target = defined_in.get(fn)
        if target:
            for f in files:
                callers.setdefault(target, {}).setdefault(fn, set()).add(
                    f"(sibling file {f})")

    # unit id -> C functions it defines (the domain of its symbol_map)
    owned: dict[str, list[str]] = {}
    for fn, uid in defined_in.items():
        owned.setdefault(uid, []).append(fn)

    extras: dict[str, str] = {}
    for u in units:
        parts = []
        if symbol_map and owned.get(u.id):
            parts.append(
                "\nC FUNCTIONS DEFINED IN THIS UNIT (your `symbol_map` must"
                " account for EVERY one — where it went, or why it is gone):"
                f" {', '.join(sorted(owned[u.id]))}")
        if u.id in callers:
            edges = "; ".join(
                f"{fn} <- {', '.join(sorted(who))}"
                for fn, who in sorted(callers[u.id].items()))
            parts.append(f"\nCALLED BY (C call graph — these callers will "
                         f"invoke what this unit defines; design for them, "
                         f"they must be able to reach it): {edges}")
        if call_sites and u.id in call_texts:
            # cap PER CALLEE at render time: the per-block cap in the chunker
            # bounds one caller, but the union over many callers can run to
            # dozens of near-identical lines (text_equals(argv[1], "demo") and
            # friends), which buys nothing and crowds the prompt. The first few
            # distinct forms already show whether the arguments vary.
            sites = "\n".join(
                f"  {t}" for _fn, texts in sorted(call_texts[u.id].items())
                for t in sorted(texts)[:4])
            parts.append(
                "\nCALL SITES (how those callers actually invoke it). Every"
                " argument here is part of the contract. If an argument's"
                " VALUE DIFFERS between two call sites, it is a real input the"
                " body cannot recover on its own — it MUST survive as a"
                " parameter of your Rust signature, even when it looks like an"
                " implementation detail:\n" + sites)
        for fn in sorted(owned.get(u.id, [])):
            info = (exit_status or {}).get(fn)
            if not info:
                continue
            codes = ", ".join(str(c) for c in info.codes)
            chain = " -> ".join(info.path)
            caveat = ("" if info.exhaustive else
                      " This list may be INCOMPLETE — some return value is"
                      " computed or comes from a macro"
                      + (f" ({', '.join(info.symbolic)})" if info.symbolic else "")
                      + ", so read the C for the full set.")
            parts.append(
                f"\nEXIT STATUS — `{fn}`'s return value BECOMES THE PROCESS EXIT"
                f" STATUS, via {chain}. The C can return these distinct values:"
                f" {codes}. Each one is observable output, exactly like printed"
                " text: a differential test compares the process exit code."
                " Your signature must let the caller recover EVERY one of them"
                " — return the status directly, or use one error variant per"
                " code. A `Result<(), E>` that the entry point collapses to"
                " `Err(_) => 1` reports every failure as 1, and no compiler"
                f" error will ever flag it.{caveat}")

        snippet = "\n".join("\n".join(lines[s - 1:e]) for s, e in u.ranges)
        fmts = c_format_strings(snippet) if output_formats else ""
        if fmts:
            parts.append(
                "\nOUTPUT FORMATS — this unit's C emits exactly these. They are"
                " the program's observable output and must be reproduced"
                " BYTE-FOR-BYTE: same spacing, punctuation, quoting and field"
                " order. Rust's `{:?}`/`Display` defaults do NOT match C's"
                " printf; build the string explicitly to match:\n" + fmts)
        if c_context in ("literals", "full"):
            if c_context == "full":
                parts.append(
                    "\nORIGINAL C SOURCE (reference for literals, output"
                    " formats, constants, and data values ONLY — the behavior"
                    " description and invariants above remain the contract;"
                    " write idiomatic Rust, do not transliterate):"
                    f"\n```c\n{snippet}\n```")
            else:
                lits = c_literals(snippet)
                if lits:
                    parts.append(
                        "\nLITERAL VALUES from this unit's original C source"
                        " (exact strings/numbers the behavior and output must"
                        " preserve — they say nothing about API shape):\n"
                        + lits)
        if parts:
            extras[u.id] = "".join(parts)
    return extras


_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_C_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_C_CHAR = re.compile(r"'(?:\\.|[^'\\])'")
_C_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


_PRINTF_FAMILY = ("printf", "fprintf", "sprintf", "snprintf", "vprintf",
                  "vfprintf", "puts", "fputs")
_PRINTF_CALL = re.compile(
    r"\b(" + "|".join(_PRINTF_FAMILY) + r")\s*\(", re.MULTILINE)


def c_format_strings(c_snippet: str, max_items: int = 24) -> str:
    """`printf`-family format strings in this C, as `fn("...")` lines.

    These are the program's observable output, verbatim. They are surfaced
    SEPARATELY from c_literals: a format string is not merely a literal to
    preserve somewhere in the crate, it is the exact byte layout one function
    must emit — and it is the thing a model most readily replaces with
    `{:?}`, which never matches."""
    text = _C_COMMENT.sub("", c_snippet)
    out: list[str] = []
    for m in _PRINTF_CALL.finditer(text):
        # first string literal inside this call's argument list; for fprintf
        # that correctly skips the stream argument (it is not a literal).
        tail = text[m.end():m.end() + 600]
        depth_end = tail.find(";")
        lit = _C_STRING.search(tail if depth_end < 0 else tail[:depth_end])
        if not lit:
            continue
        entry = f'{m.group(1)}({lit.group(0)})'
        if entry not in out:
            out.append(entry)
        if len(out) >= max_items:
            break
    return "\n".join(f"  {e}" for e in out)


def c_literals(c_snippet: str, max_items: int = 60) -> str:
    """String/char/numeric literals from a C snippet (comments stripped),
    deduped in first-appearance order — the data a translation must carry
    even when it never sees the code."""
    src = _C_COMMENT.sub(" ", c_snippet)
    seen: list[str] = []
    for pat in (_C_STRING, _C_CHAR, _C_NUMBER):
        for m in pat.finditer(src):
            v = m.group(0)
            if v not in seen:
                seen.append(v)
    if not seen:
        return ""
    shown = seen[:max_items]
    tail = f"\n  ... and {len(seen) - max_items} more" if len(seen) > max_items else ""
    return "  " + ", ".join(shown) + tail
