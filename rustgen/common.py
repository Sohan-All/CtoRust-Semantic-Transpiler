"""Shared helpers for the rustgen stages."""

from __future__ import annotations

import re

from state import Explanation


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
                external_callers: dict[str, list[str]] | None = None
                ) -> dict[str, str]:
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
    (from the project index)."""
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
    for b in graph.blocks:
        if not b.function:
            continue
        caller_unit = fn_unit.get(b.function)
        for callee in getattr(b, "calls_internal", []) or []:
            target = defined_in.get(callee)
            if target and target != caller_unit:
                callers.setdefault(target, {}).setdefault(callee, set()).add(
                    f"{b.function} [{caller_unit}]" if caller_unit else b.function)
    for fn, files in (external_callers or {}).items():
        target = defined_in.get(fn)
        if target:
            for f in files:
                callers.setdefault(target, {}).setdefault(fn, set()).add(
                    f"(sibling file {f})")

    extras: dict[str, str] = {}
    for u in units:
        parts = []
        if u.id in callers:
            edges = "; ".join(
                f"{fn} <- {', '.join(sorted(who))}"
                for fn, who in sorted(callers[u.id].items()))
            parts.append(f"\nCALLED BY (C call graph — these callers will "
                         f"invoke what this unit defines; design for them, "
                         f"they must be able to reach it): {edges}")
        if c_context in ("literals", "full"):
            snippet = "\n".join("\n".join(lines[s - 1:e])
                                for s, e in u.ranges)
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
