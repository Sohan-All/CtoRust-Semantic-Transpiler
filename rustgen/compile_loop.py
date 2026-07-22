"""Compile loop with per-error routed repairs.

cargo check diagnostics are attributed to the MTU sections of lib.rs (marker
comments delimit sections; the shared type layer and the FFI shim module are
sections too). Each error is then ROUTED by what it actually implicates:

- all of a diagnostic's spans (primary, secondary, notes) land in ONE section
  -> cheap single-section repair, as before;
- spans touch TWO OR MORE sections (the E0308 caller/callee class), or a
  single-span name error ("cannot find function `x`") whose identifier is
  defined in another section -> the sections are joined into a repair CLUSTER
  and fixed in one call that sees both sides plus the authority rule: the
  stage-S spec is the contract — fix whichever side deviates from its spec;
  if both conform, change the caller (callee signatures are load-bearing).

Clusters are connected components (union-find) over the cross-error edges, so
a section never receives two conflicting repairs in one round; a section's
single-section errors ride along into its cluster's call. A round that makes
NO progress escalates once: every error-bearing section is bundled into one
capped joint call. A second stall ends the loop. The best-state regression
guard wraps every tier.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from config import Config
from llm import LLM
from state import Explanation
from rustgen.common import extract_rust, render_spec, unit_block

SHARED = "__shared__"   # the types/deps region
FFI = "__ffi__"         # the extern "C" shim layer

MAX_CLUSTER = 4         # most-implicated sections kept when a cluster is bigger

# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

REPAIR_CODE_PROMPT = """\
The following Rust code implements one behavioral unit of a program, but it
fails to compile. Fix it. The errors below are ONLY those attributed to this
unit's code — do not modify or redefine the shared types or other units'
functions (their ACTUAL current signatures are listed; call them exactly as
listed). Keep this unit's own public function signatures exactly as they are —
other code calls them. Preserve the unit's behavior and its `// invariant:`
comments. If an error indicates a needed capability that genuinely cannot be
expressed with the given types and signatures, replace only that part with
`todo!("<what is missing>")`.
Comment discipline: inside the code block, the ONLY comments allowed are
`///` docs and `// invariant:` citations. Never write comments about the
errors, the fix, the spec, other units, or your reasoning — explain nothing
inside the code.

SHARED TYPES (context, do not re-emit):
```rust
{types_rs}
```

OTHER UNITS' SIGNATURES (context, do not re-emit):
{sibling_sigs}

UNIT BEHAVIOR (must be preserved):
{unit}

ITS RUST DESIGN SPEC (still the contract):
{spec}

CURRENT CODE OF THIS UNIT:
```rust
{code}
```

COMPILER ERRORS FOR THIS UNIT:
{errors}

Reply with ONLY a ```rust code block containing the corrected code for this
unit (the full unit, not a diff).
"""

REPAIR_SHARED_PROMPT = """\
The following Rust type definitions and dependency stubs fail to compile (or
are referenced incorrectly by code built against them). Fix the definitions.
Keep every existing type and function NAME stable — implementations elsewhere
call them; you may fix bodies, fields, derives, generics, and signatures'
types, but renaming or deleting items will break callers. Safe, idiomatic
Rust; no raw pointers (use opaque structs for foreign handles). No comments
about the errors or the fix — only `///` docs and `// invariant:` lines.

CURRENT DEFINITIONS:
```rust
{types_rs}
```

COMPILER ERRORS ATTRIBUTED TO THESE DEFINITIONS:
{errors}

Reply with ONLY a ```rust code block containing the corrected definitions.
"""

REPAIR_FFI_PROMPT = """\
The following Rust C-ABI export layer (extern "C" shims) fails to compile.
Fix it. Constraints: every `#[no_mangle]` symbol NAME, arity, and C-compatible
ABI must stay exactly as-is (equivalence testing binds to these symbols);
`unsafe` and raw pointers are allowed here (FFI boundary only); do not modify
or re-emit the shared types or idiomatic functions — adapt to them. No
comments about the errors or the fix — only `///`/`// SAFETY:` lines.

SHARED TYPES (context):
```rust
{types_rs}
```

IDIOMATIC SIGNATURES (context, call these):
{idiomatic_sigs}

CURRENT FFI MODULE:
```rust
{ffi_rs}
```

COMPILER ERRORS FOR THIS MODULE:
{errors}

Reply with ONLY a ```rust code block containing the corrected ffi module.
"""

REPAIR_CLUSTER_PROMPT = """\
Several sections of a generated Rust library fail to compile because of
DISAGREEMENTS BETWEEN THEM (mismatched types at call sites, missing or
misnamed items). You are given every involved section: its role, its design
spec (where it has one), and its current code — plus the errors.

THE AUTHORITY RULE: each section's design spec is its contract. Fix whichever
side DEVIATES from its own spec. If both sides conform to their specs, change
the CALLER to match the callee — callee signatures are load-bearing for other
code you cannot see. Never rename a `#[no_mangle]` symbol, a shared type, or
a deps stub. Keep each section's `// invariant:` comments and behavior.
Do not move code between sections. No comments about the disagreement or the
fix — inside code, only `///` docs and `// invariant:` citations.

SHARED TYPES (context; only re-emit if the section list includes __shared__):
```rust
{types_rs}
```

OTHER SECTIONS' SIGNATURES (context):
{sibling_sigs}

INVOLVED SECTIONS:
{sections}

COMPILER ERRORS:
{errors}

Reply with the corrected code for EVERY involved section, using EXACTLY this
layout (one block per section, same ids, full section code — not a diff):

=== SECTION <section id> ===
```rust
<corrected code>
```
"""


TODO_STUB_PROMPT = """\
The following Rust code implements one behavioral unit of a program, but it
still contains `todo!()` stubs. These compile, so no compiler error will ever
flag them — at runtime they PANIC. The stubs were left on the assumption that
some other unit provides the capability; the sibling signatures below are the
COMPLETE list of what other units actually provide. If a stub's capability is
listed there, call it. Otherwise NO other unit provides it — implement it
yourself now, using only the shared types, the sibling signatures, and Rust
std. Keep this unit's signatures exactly as they are. Only if the capability
is genuinely impossible with the given types may a `todo!("<why>")` remain.
Comment discipline: inside the code block, only `///` docs and
`// invariant:` citations — nothing about the stubs, the spec, or your
reasoning.

SHARED TYPES (context, do not re-emit):
```rust
{types_rs}
```

SIBLING SIGNATURES (the complete list — implemented elsewhere):
{sibling_sigs}

UNIT BEHAVIOR (must be preserved):
{unit}

ITS RUST DESIGN SPEC (still the contract):
{spec}

CURRENT CODE OF THIS UNIT (contains the todo!() stubs):
```rust
{code}
```

Reply with ONLY a ```rust code block containing the completed code for this
unit (the full unit, not a diff).
"""


SURGICAL_PROMPT = """\
The Rust code below has {n_errors} compiler error(s). Produce the MINIMAL
edits that fix them — do not reformat, rename, restructure, or touch any line
the errors don't require. Each edit is an exact-match replacement: "find" must
be copied VERBATIM from the code (including whitespace) and long enough to be
unique; "replace" is its corrected form.

CODE:
```rust
{code}
```

COMPILER ERRORS:
{errors}

Reply with ONLY a JSON object:
{{"edits": [{{"find": "<verbatim snippet>", "replace": "<corrected snippet>"}}, ...]}}
"""


# --------------------------------------------------------------------------- #
# tier 0 — rustc's own machine-applicable suggestions (deterministic, free)
# --------------------------------------------------------------------------- #

def _applicable_spans(errors: list[dict]) -> list[tuple[int, int, str]]:
    """(byte_start, byte_end, replacement) for machine-applicable suggestions
    in lib.rs, deduped and non-overlapping (kept in descending start order so
    they can be applied without offset bookkeeping)."""
    cands: set[tuple[int, int, str]] = set()
    for e in errors:
        for holder in [e] + e.get("children", []):
            for span in holder.get("spans", []):
                appl = str(span.get("suggestion_applicability") or "")
                appl = appl.lower().replace("-", "").replace("_", "")
                if (span.get("suggested_replacement") is not None
                        and appl == "machineapplicable"
                        and span.get("file_name", "").endswith("lib.rs")):
                    cands.add((span["byte_start"], span["byte_end"],
                               span["suggested_replacement"]))
    picked: list[tuple[int, int, str]] = []
    last_start = None
    for start, end, repl in sorted(cands, key=lambda t: (-t[0], t[1])):
        if last_start is not None and end > last_start:
            continue  # overlaps something already picked
        picked.append((start, end, repl))
        last_start = start
    return picked


def apply_rustc_suggestions(errors: list[dict], lib_rs: str) -> tuple[str, int]:
    edits = _applicable_spans(errors)
    if not edits:
        return lib_rs, 0
    data = lib_rs.encode()
    for start, end, repl in edits:  # already descending by start
        data = data[:start] + repl.encode() + data[end:]
    return data.decode(errors="replace"), len(edits)


# --------------------------------------------------------------------------- #
# tier 0.5 — duplicate-item deletion (deterministic, free)
# --------------------------------------------------------------------------- #
# Two units implementing the same item (E0428 duplicate fn, E0592 duplicate
# method, E0119 conflicting trait impls) has a deterministic fix no LLM
# needs: delete every copy but one. Keeper preference: the section whose
# spec OWNS the concern; otherwise the first in assembly order.

_DUP_NAME = re.compile(
    r"the name `(\w+)` is defined multiple times|duplicate definitions with name `(\w+)`")
_DUP_IMPL = re.compile(
    r"conflicting implementations of trait `([^`]+)` for type `([^`]+)`")


def _item_spans(code_str: str, header: re.Pattern) -> list[tuple[int, int]]:
    """Byte spans of full items (incl. body braces or trailing ';') whose
    header line matches `header`."""
    spans = []
    for m in header.finditer(code_str):
        i, depth, end = m.end(), 0, None
        while i < len(code_str):
            ch = code_str[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            elif ch == ";" and depth == 0:
                end = i + 1
                break
            i += 1
        if end is not None:
            while end < len(code_str) and code_str[end] == "\n":
                end += 1
            spans.append((m.start(), end))
    return spans


def _base_name(type_path: str) -> str:
    return re.sub(r"<.*", "", type_path).split("::")[-1].strip()


def apply_duplicate_deletions(errors: list[dict], sections_code: dict[str, str],
                              set_section, order: list[str],
                              specs: dict[str, dict]) -> int:
    """Delete redundant copies of items rustc reports as duplicated.
    Returns the number of items deleted."""
    deleted = 0
    seen_keys: set[str] = set()
    for e in errors:
        msg = e.get("message", "")
        m_name, m_impl = _DUP_NAME.search(msg), _DUP_IMPL.search(msg)
        if m_name:
            name = m_name.group(1) or m_name.group(2)
            key = f"name:{name}"
            hdr = re.compile(
                rf"(?m)^[ \t]*(?:#\[[^\n]*\]\s*)*(?:pub(?:\(crate\))?\s+)?"
                rf"(?:async\s+)?(?:fn|struct|enum|trait|const|static|type)\s+{re.escape(name)}\b")
            owns_test = lambda o, n=name: n in o
        elif m_impl:
            tr, ty = _base_name(m_impl.group(1)), _base_name(m_impl.group(2))
            key = f"impl:{tr}:{ty}"
            hdr = re.compile(
                rf"(?m)^[ \t]*impl(?:<[^>\n]*>)?\s+(?:[\w]+::)*{re.escape(tr)}"
                rf"(?:<[^>\n]*>)?\s+for\s+{re.escape(ty)}\b")
            owns_test = lambda o, a=tr, b=ty: a in o and b in o
        else:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)

        found: list[tuple[str, tuple[int, int]]] = []
        for sid in order:
            code_str = sections_code.get(sid, "")
            found.extend((sid, span) for span in _item_spans(code_str, hdr))
        if not found:
            continue
        if len(found) == 1:
            # E0119 with one explicit impl: the conflict is with a derive —
            # the derive (shared-model authority) wins, the manual impl goes
            if not m_impl:
                continue
            keep = None
        else:
            owners = [sid for sid, _ in found
                      if any(owns_test(o) for o in specs.get(sid, {}).get("owns", []))]
            keep_sid = owners[0] if len(set(owners)) == 1 and owners else found[0][0]
            keep = next(i for i, (sid, _) in enumerate(found) if sid == keep_sid)

        by_section: dict[str, list[tuple[int, int]]] = {}
        for i, (sid, span) in enumerate(found):
            if i != keep:
                by_section.setdefault(sid, []).append(span)
        for sid, spans in by_section.items():
            code_str = sections_code[sid]
            for start, end in sorted(spans, reverse=True):
                code_str = code_str[:start] + code_str[end:]
                deleted += 1
            set_section(sid, code_str)
            sections_code[sid] = code_str
    return deleted


_SPLIT_MARKER = re.compile(
    r"^// ===== (?:MTU ((?:\w+__)?exp_\d+)|(shared data model)|(FFI shims))[^\n]*$", re.MULTILINE)
_LOW_CONF = re.compile(r"^// LOW CONFIDENCE.*\n(?:// .*\n)?", re.MULTILINE)


def split_lib(lib_rs: str) -> dict[str, str]:
    """Inverse of assembly: lib.rs -> {section id: code}, stripping the marker
    and LOW CONFIDENCE comment lines assembly injects. Used after tier-0 edits
    so the fixed text flows back into the section state."""
    matches = list(_SPLIT_MARKER.finditer(lib_rs))
    out: dict[str, str] = {}
    for m, nxt in zip(matches, matches[1:] + [None]):
        sid = m.group(1) or (SHARED if m.group(2) else FFI)
        body = lib_rs[m.end():nxt.start() if nxt else len(lib_rs)]
        body = _LOW_CONF.sub("", body, count=1)
        out[sid] = body.strip("\n")
    return out


# --------------------------------------------------------------------------- #
# tier 1 — surgical exact-match edits (all-or-nothing application)
# --------------------------------------------------------------------------- #

def apply_surgical_edits(code_str: str, edits: list[dict]) -> str | None:
    """Apply find/replace edits; None if any find is absent (nothing applied —
    the caller falls back to a full rewrite)."""
    if not edits or len(edits) > 8:
        return None
    result = code_str
    for e in edits:
        find, replace = e.get("find"), e.get("replace")
        if not isinstance(find, str) or not isinstance(replace, str) or find not in result:
            return None
        result = result.replace(find, replace, 1)
    return result


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

@dataclass
class CompileReport:
    rounds: list[dict] = field(default_factory=list)
    final_errors: int = -1
    clean_units: int = 0
    total_units: int = 0

    def summary(self) -> str:
        parts = []
        tiers = ""
        for r in self.rounds:
            if "total_errors" in r:
                parts.append(str(r["total_errors"]))
            elif "reverted_to_errors" in r:
                parts.append(f"reverted({r['reverted_to_errors']})")
            elif "tier_stats" in r:
                t = r["tier_stats"]
                tiers = (f"; repairs: {t['suggestions']} rustc-auto, "
                         f"{t['surgical']} surgical, {t['full']} rewrites")
        return (f"errors per round: {' -> '.join(parts)}; final: {self.final_errors}; "
                f"MTU sections clean: {self.clean_units}/{self.total_units}{tiers}")


# --------------------------------------------------------------------------- #
# cargo + section machinery
# --------------------------------------------------------------------------- #

def cargo_check(crate: Path) -> list[dict]:
    proc = subprocess.run(
        ["cargo", "check", "--message-format=json", "--quiet"],
        cwd=crate, capture_output=True, text=True, timeout=300)
    errors = []
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        m = msg.get("message", {})
        if m.get("level") != "error":
            continue
        if m.get("code") is None and "aborting due to" in m.get("message", ""):
            continue
        errors.append(m)
    return errors


def section_index(lib_rs: str) -> list[tuple[int, str]]:
    idx = []
    for i, line in enumerate(lib_rs.split("\n"), start=1):
        m = re.match(r"// ===== MTU ((?:\w+__)?exp_\d+)", line)
        if m:
            idx.append((i, m.group(1)))
        elif line.startswith("// ===== shared data model"):
            idx.append((i, SHARED))
        elif line.startswith("// ===== FFI shims"):
            idx.append((i, FFI))
    return idx


def owner_of(line: int, index: list[tuple[int, str]]) -> str:
    owner = SHARED
    for start, sid in index:
        if start <= line:
            owner = sid
        else:
            break
    return owner


def _all_span_lines(err: dict) -> list[int]:
    """Every lib.rs line any part of the diagnostic points at — primary and
    secondary spans plus the spans of child notes ('function defined here')."""
    lines = []
    for holder in [err] + err.get("children", []):
        for span in holder.get("spans", []):
            if span.get("file_name", "").endswith("lib.rs") and span.get("line_start"):
                lines.append(span["line_start"])
    return lines


_NAME_ERR = re.compile(r"cannot find (?:function|value|type|method)|no method named|no function or associated item|no variant, associated function, or constant named")
_BACKTICKED = re.compile(r"`(\w+)`")

_MISSING_NAME = re.compile(
    r"cannot find (?:function|value|method) `(\w+)`"
    r"|no (?:method|function or associated item|variant, associated function, or constant) named `(\w+)`")


def missing_capability_notes(errs: list[dict], sections_code: dict[str, str]) -> str:
    """For name errors whose identifier is defined NOWHERE in the crate,
    a note telling the repair to implement the behavior instead of keeping
    the phantom call — the model otherwise stalls, re-calling a capability
    it keeps assuming some sibling provides."""
    missing: list[str] = []
    for e in errs:
        m = _MISSING_NAME.search(e.get("message", ""))
        if not m:
            continue
        name = m.group(1) or m.group(2)
        if name in missing:
            continue
        pat = re.compile(rf"\bfn\s+{re.escape(name)}\b")
        if not any(pat.search(c) for c in sections_code.values()):
            missing.append(name)
    if not missing:
        return ""
    return "\n\n" + "\n".join(
        f"NOTE: `{n}` does not exist ANYWHERE in this crate — no unit "
        f"implements it and none will. Do not keep calling it: implement the "
        f"behavior inline (or as a private helper) right here."
        for n in missing)


def _name_lookup(err: dict, sections_code: dict[str, str]) -> str | None:
    """For one-span name errors: which section defines (or should define) the
    missing identifier?"""
    if not _NAME_ERR.search(err.get("message", "")):
        return None
    for name in _BACKTICKED.findall(err.get("message", "")):
        pat = re.compile(rf"\bfn\s+{re.escape(name)}\b")
        for sid, code_str in sections_code.items():
            if pat.search(code_str):
                return sid
    return None


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def route_errors(errors: list[dict], lib_rs: str,
                 sections_code: dict[str, str]) -> list[tuple[frozenset, list[dict]]]:
    """Group errors into repair clusters. Returns [(sections, errors), ...]
    where singleton clusters take the single-repair tier and multi-section
    clusters take the joint tier."""
    index = section_index(lib_rs)
    uf = _UnionFind()
    err_sections: list[tuple[dict, set[str]]] = []

    for e in errors:
        touched = {owner_of(l, index) for l in _all_span_lines(e)} or {SHARED}
        if len(touched) == 1:
            other = _name_lookup(e, sections_code)
            only = next(iter(touched))
            if other and other != only:
                touched.add(other)
        err_sections.append((e, touched))
        sids = sorted(touched)
        for a, b in zip(sids, sids[1:]):
            uf.union(a, b)

    clusters: dict[str, tuple[set, list]] = {}
    for e, touched in err_sections:
        root = uf.find(sorted(touched)[0])
        secs, errs = clusters.setdefault(root, (set(), []))
        secs.update(touched)
        errs.append(e)
    return [(frozenset(secs), errs) for secs, errs in clusters.values()]


def render_errors(errs: list[dict], cap: int) -> str:
    out = []
    for e in errs[:cap]:
        rendered = e.get("rendered") or e.get("message", "")
        out.append(rendered.strip()[:1200])
    if len(errs) > cap:
        out.append(f"... and {len(errs) - cap} more errors")
    return "\n\n".join(out)


_FN_SIG = re.compile(r"^\s*(?:pub\s+)?(?:unsafe\s+)?(?:async\s+)?fn\s+\w+.*?(?=\s*\{|$)")
_IMPL_HDR = re.compile(r"^\s*impl\b[^{]*")


def extracted_signatures(code_map: dict[str, str], skip: set[str] | None = None) -> str:
    """Actual function signatures parsed from the CURRENT code (repairs drift
    signatures away from the stage-S specs — repair against reality)."""
    skip = skip or set()
    lines: list[str] = []
    for uid, code in code_map.items():
        if uid in skip:
            continue
        impl_hdr = ""
        for raw in code.split("\n"):
            m = _IMPL_HDR.match(raw)
            if m and "{" in raw:
                impl_hdr = m.group(0).strip()
                continue
            f = _FN_SIG.match(raw)
            if f:
                sig = f.group(0).strip()
                lines.append(f"- {impl_hdr} :: {sig}" if impl_hdr else f"- {sig}")
    return "\n".join(lines) or "(none)"


_SECTION_BLOCK = re.compile(r"=== SECTION (\S+) ===\s*```(?:rust)?\s*\n(.*?)```", re.DOTALL)


def parse_cluster_reply(reply: str) -> dict[str, str]:
    return {sid: body.strip() for sid, body in _SECTION_BLOCK.findall(reply)}


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

async def compile_loop(cfg: Config, llm: LLM, crate: Path,
                       units: list[Explanation], specs: dict[str, dict],
                       types_rs: str, code: dict[str, str],
                       reassemble, ffi_rs: str = ""
                       ) -> tuple[str, dict[str, str], str, CompileReport]:
    report = CompileReport(total_units=len(units))
    units_by_id = {u.id: u for u in units}
    best: tuple | None = None          # (n_errors, types, code, ffi, dirty_sections)
    prev_count: int | None = None
    stalled_once = False

    def sections_code() -> dict[str, str]:
        m = dict(code)
        m[SHARED] = types_rs
        if ffi_rs:
            m[FFI] = ffi_rs
        return m

    def set_section(sid: str, new_code: str) -> None:
        nonlocal types_rs, ffi_rs
        if not new_code:
            return
        if sid == SHARED:
            types_rs = new_code
        elif sid == FFI:
            ffi_rs = new_code
        else:
            code[sid] = new_code

    def section_label(sid: str) -> str:
        if sid == SHARED:
            return "shared data model + deps stubs"
        if sid == FFI:
            return "extern \"C\" shim layer (symbol names immutable)"
        return f"MTU {sid}"

    def render_sections(sids: list[str]) -> str:
        parts = []
        sc = sections_code()
        for sid in sids:
            parts.append(f"=== SECTION {sid} === ({section_label(sid)})")
            if sid in units_by_id:
                parts.append(unit_block(units_by_id[sid]))
                parts.append("Design spec:\n" + render_spec(specs.get(sid, {})))
            parts.append(f"```rust\n{sc.get(sid, '')}\n```")
        return "\n\n".join(parts)

    tier_stats = {"suggestions": 0, "surgical": 0, "full": 0}

    async def resolve_todo_stub(sid: str) -> None:
        reply = await llm.ask(TODO_STUB_PROMPT.format(
            types_rs=types_rs,
            sibling_sigs=extracted_signatures(code, skip={sid}),
            unit=unit_block(units_by_id[sid]),
            spec=render_spec(specs.get(sid, {})),
            code=code.get(sid, "")),
            max_tokens=cfg.rustgen_repair_max_tokens)
        new_code = extract_rust(reply)
        # only accept progress: fewer stubs, section not emptied
        if new_code and new_code.count("todo!") < code[sid].count("todo!"):
            code[sid] = new_code

    # todo!() gate — stubs compile clean, so the error-driven rounds below
    # never see them; they surface as runtime panics instead (a unit deferring
    # to a sibling that never implemented the capability). One targeted pass
    # before the loop, with the REAL sibling surface as context.
    stubbed = [sid for sid, c in code.items()
               if "todo!" in c and sid in units_by_id]
    if stubbed:
        await asyncio.gather(*(resolve_todo_stub(sid) for sid in stubbed))
        remaining = [sid for sid in stubbed if "todo!" in code.get(sid, "")]
        report.rounds.append({"todo_stubs": sorted(stubbed),
                              "todo_remaining": sorted(remaining)})

    async def try_surgical(sid: str, errs: list[dict]) -> bool:
        current = sections_code().get(sid, "")
        if not current:
            return False
        resp = await llm.ask_json(SURGICAL_PROMPT.format(
            n_errors=len(errs), code=current,
            errors=render_errors(errs, cfg.rustgen_max_errors_per_section)),
            max_tokens=cfg.rustgen_repair_max_tokens)
        edits = resp.get("edits", []) if isinstance(resp, dict) else []
        fixed = apply_surgical_edits(current, [e for e in edits if isinstance(e, dict)])
        if fixed is None or fixed == current:
            return False
        set_section(sid, fixed)
        tier_stats["surgical"] += 1
        return True

    async def repair_single(sid: str, errs: list[dict]) -> None:
        # tier 1: surgical exact-match edits for small error counts; falls
        # back to the full-section rewrite when edits don't apply
        if len(errs) <= cfg.rustgen_surgical_max_errors and await try_surgical(sid, errs):
            return
        tier_stats["full"] += 1
        rendered = (render_errors(errs, cfg.rustgen_max_errors_per_section)
                    + missing_capability_notes(errs, sections_code()))
        if sid == SHARED:
            reply = await llm.ask(REPAIR_SHARED_PROMPT.format(
                types_rs=types_rs, errors=rendered),
                max_tokens=cfg.rustgen_repair_max_tokens)
        elif sid == FFI:
            reply = await llm.ask(REPAIR_FFI_PROMPT.format(
                types_rs=types_rs, idiomatic_sigs=extracted_signatures(code),
                ffi_rs=ffi_rs, errors=rendered),
                max_tokens=cfg.rustgen_repair_max_tokens)
        elif sid in units_by_id:
            reply = await llm.ask(REPAIR_CODE_PROMPT.format(
                types_rs=types_rs,
                sibling_sigs=extracted_signatures(code, skip={sid}),
                unit=unit_block(units_by_id[sid]),
                spec=render_spec(specs.get(sid, {})),
                code=code.get(sid, ""), errors=rendered),
                max_tokens=cfg.rustgen_repair_max_tokens)
        else:
            return
        set_section(sid, extract_rust(reply))

    async def repair_cluster(sids: frozenset, errs: list[dict]) -> None:
        ranked = sorted(sids)[:MAX_CLUSTER]
        reply = await llm.ask(REPAIR_CLUSTER_PROMPT.format(
            types_rs=types_rs,
            sibling_sigs=extracted_signatures(sections_code(), skip=set(ranked)),
            sections=render_sections(ranked),
            errors=render_errors(errs, cfg.rustgen_max_errors_per_section * 2)
                   + missing_capability_notes(errs, sections_code())),
            max_tokens=cfg.rustgen_repair_max_tokens * 2)
        for sid, new_code in parse_cluster_reply(reply).items():
            if sid in ranked:  # never accept updates for uninvolved sections
                set_section(sid, new_code)

    section_order = [SHARED] + [u.id for u in sorted(units, key=Explanation.sort_key)] + [FFI]

    def fingerprint_set(errs: list[dict], lib_text: str) -> set:
        idx = section_index(lib_text)
        return {(str((e.get("code") or {}).get("code", "")),
                 owner_of(min(_all_span_lines(e) or [1]), idx),
                 e.get("message", "")[:60]) for e in errs}

    prev_fps: set | None = None
    prev_repaired: set[str] | None = None
    reverts = 0
    conservative = False   # after churn: per-section repairs only, no joints

    for round_num in range(cfg.rustgen_compile_rounds + 1):
        reassemble(types_rs, code, ffi_rs)
        errors = cargo_check(crate)
        lib_rs = (crate / "src" / "lib.rs").read_text()

        # tier 0: rustc's own machine-applicable fixes — free, no LLM. Applied
        # to lib.rs by byte offset, split back into section state, re-checked.
        fixed_lib, n_applied = apply_rustc_suggestions(errors, lib_rs)
        if n_applied:
            for sid, body in split_lib(fixed_lib).items():
                set_section(sid, body)
            reassemble(types_rs, code, ffi_rs)
            errors = cargo_check(crate)
            lib_rs = (crate / "src" / "lib.rs").read_text()
            tier_stats["suggestions"] += n_applied

        # tier 0.5: duplicate-item deletion — deterministic, free
        n_dupes = apply_duplicate_deletions(errors, sections_code(), set_section,
                                            section_order, specs)
        if n_dupes:
            reassemble(types_rs, code, ffi_rs)
            errors = cargo_check(crate)
            lib_rs = (crate / "src" / "lib.rs").read_text()
            tier_stats["dupes_deleted"] = tier_stats.get("dupes_deleted", 0) + n_dupes

        fps = fingerprint_set(errors, lib_rs)

        # SET-BASED churn guard. Error counts are not monotone progress:
        # fixing an error legitimately unmasks deeper ones IN the repaired
        # sections (rustc reports later-phase errors only once earlier
        # phases pass). Growth is only a regression when NEW error
        # fingerprints appear OUTSIDE the sections the last round repaired —
        # that means the repair changed a contract and broke bystanders.
        churn = False
        if (prev_fps is not None and prev_repaired and best is not None
                and len(errors) > best[0]):
            out_of_cluster = [fp for fp in fps - prev_fps
                              if fp[1] not in prev_repaired]
            churn = bool(out_of_cluster)
        if churn:
            _, types_rs, code, ffi_rs, _ = best
            code = dict(code)
            reverts += 1
            conservative = True
            reassemble(types_rs, code, ffi_rs)
            errors = cargo_check(crate)
            lib_rs = (crate / "src" / "lib.rs").read_text()
            fps = fingerprint_set(errors, lib_rs)
            report.rounds.append({"round": round_num,
                                  "reverted_to_errors": len(errors),
                                  "mode": "conservative"})

        clusters = route_errors(errors, lib_rs, sections_code())
        dirty = {sid for secs, _ in clusters for sid in secs} if errors else set()
        report.rounds.append({
            "round": round_num,
            "total_errors": len(errors),
            "rustc_suggestions_applied": n_applied,
            "duplicates_deleted": n_dupes,
            "clusters": [{"sections": sorted(s), "errors": len(e)} for s, e in clusters],
        })

        if best is None or len(errors) <= best[0]:
            best = (len(errors), types_rs, dict(code), ffi_rs, dirty)

        stalled = prev_count is not None and len(errors) == prev_count
        done = (not errors
                or round_num == cfg.rustgen_compile_rounds
                or reverts >= 2
                or (stalled and stalled_once))
        if done:
            # never return worse than the best state seen
            if best is not None and len(errors) > best[0]:
                _, types_rs, code, ffi_rs, dirty = best
                code = dict(code)
                reassemble(types_rs, code, ffi_rs)
                errors = [None] * best[0]
            report.final_errors = len(errors)
            report.clean_units = sum(1 for u in units if u.id not in dirty)
            report.rounds.append({"tier_stats": dict(tier_stats)})
            return types_rs, code, ffi_rs, report
        prev_count = len(errors)
        prev_fps = fps
        prev_repaired = set(dirty)

        if conservative:
            # post-churn fallback: repair every dirty section individually —
            # no joint rewrites, so no section can re-decide another's contract
            by_sec: dict[str, list[dict]] = {}
            for secs, errs in clusters:
                for sid in secs:
                    by_sec.setdefault(sid, []).extend(errs)
            await asyncio.gather(*(repair_single(sid, errs)
                                   for sid, errs in by_sec.items()))
            continue

        if stalled:
            # escalation tier: one bundled joint call over everything dirty
            stalled_once = True
            all_secs = frozenset(s for secs, _ in clusters for s in secs)
            all_errs = [e for _, errs in clusters for e in errs]
            report.rounds[-1]["escalated"] = sorted(all_secs)[:MAX_CLUSTER]
            await repair_cluster(all_secs, all_errs)
            continue

        singles = [(next(iter(s)), e) for s, e in clusters if len(s) == 1]
        joints = [(s, e) for s, e in clusters if len(s) > 1]
        await asyncio.gather(
            *(repair_single(sid, errs) for sid, errs in singles),
            *(repair_cluster(sids, errs) for sids, errs in joints))

    return types_rs, code, ffi_rs, report  # unreachable; loop returns inside
