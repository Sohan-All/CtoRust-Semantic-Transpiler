"""Oracle-free semantic checks over the assembled crate. Diagnostic only.

The compile loop proves the Rust TYPE-CHECKS. That is a weak gate: a
translation can restructure across an MTU boundary, drop a function, or
replace `printf("task[%d] ...")` with `{:?}` and still reach zero errors.
Measured on the binary_heap corpus, a crate with `final_errors: 0` diverged
from the C on 10/16 executable cases.

These checks read source only — no C is executed, nothing is compared against
a behavioural oracle. They detect OMISSION:

  literals  — the printable text a unit's C emits (printf-family format
              strings, split on their specifiers) must appear in the Rust that
              unit produced. Scoped per unit, so a finding names the section
              responsible. On binary_heap v3-full it reports 2 findings, both
              true (the crate prints `{:?}` where the C prints
              `task[%d] prio=%d ... cmd="%s"`). An earlier "6/6 precision"
              claim for this check was an artifact of comparing an unescaped C
              fragment against raw Rust source — see _unescape. It still has a
              known false-positive mode: output composed across several Rust
              format strings (C's `"task[%d] prio=%d"` built as
              `format!("task[{}]", i)` plus `"{} prio={}"`) reads as a loss
              when nothing was lost, so treat findings as leads, not proof.
  symbols   — every C function a unit defines must appear in that unit's
              `symbol_map` (stage S), and a mapped Rust path must actually
              exist in the crate. Catches silent disappearance.
  callshape — a C function still called from Rust, but from a different NUMBER
              of sites, indicates restructuring across a boundary. Noisy
              (measured 1/7), so it is reported as a HINT only, never counted
              as a finding.

Omission is not wrongness: a unit can keep every literal and still compute the
wrong answer. Nothing here gates or repairs — findings are recorded for a human
and for the behavioural oracle that follows; see semantic_report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Config
from state import Explanation
from rustgen.common import c_format_strings

_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_C_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_SPEC = re.compile(r"%[-+ #0]*[0-9*]*(?:\.[0-9*]+)?(?:hh|h|ll|l|L|z|j|t)?"
                   r"[diouxXeEfFgGaAcspn%]")
_CDEF = re.compile(r"^[A-Za-z_][\w \t\*]*?\b(\w+)\s*\([^;]*?\)\s*\{", re.MULTILINE)
_NOT_FUNCTIONS = {"if", "for", "while", "switch", "return", "sizeof", "do"}
_VALID_REASONS = ("drop_glue", "stdlib_equivalent", "inlined_into", "dead_code")

MAX_FINDINGS_PER_UNIT = 8


@dataclass
class SemanticReport:
    findings: dict[str, list[str]] = field(default_factory=dict)  # unit id -> findings
    hints: dict[str, list[str]] = field(default_factory=dict)     # unit id -> call-shape hints
    total: int = 0

    def summary(self) -> str:
        if not self.total:
            return "no omissions detected"
        return (f"{self.total} finding(s) in {len(self.findings)} section(s): "
                + ", ".join(sorted(self.findings)))


def unit_c_source(unit: Explanation, lines: list[str]) -> str:
    return "\n".join("\n".join(lines[s - 1:e]) for s, e in unit.ranges)


def output_fragments(c_snippet: str) -> list[str]:
    """Printable fragments the C emits, printf specifiers removed.

    Only fragments of >= 4 chars containing a letter: shorter pieces (", ",
    "]") match somewhere in any crate by accident and would be pure noise."""
    # #include lines carry "foo.h" literals that are build plumbing, never
    # output — a Rust crate correctly contains no trace of them.
    text = "\n".join(l for l in _C_COMMENT.sub("", c_snippet).splitlines()
                     if not l.lstrip().startswith("#include"))
    frags: list[str] = []
    for lit in _C_STRING.findall(text):
        body = lit[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        for piece in _SPEC.split(body):
            piece = piece.replace("\\n", "").replace("\\t", "").strip()
            if len(piece) >= 4 and re.search(r"[A-Za-z]", piece) \
                    and piece not in frags:
                frags.append(piece)
    return frags


def _unescape(text: str) -> str:
    r"""Normalise source-level string escaping before comparing.

    The C fragment is unescaped (`\"` -> `"`), so the Rust side must be too:
    C's `"report name=\"%s\""` yields the fragment `report name="`, while the
    equivalent Rust source reads `"report name=\"{}\""` and contains
    `report name=\"`. Comparing raw text reports a loss that is not there —
    re-measured on binary_heap v3-full, this accounted for 4 of the 6 findings
    the check produced before the fix."""
    return text.replace('\\"', '"').replace("\\'", "'")


def c_functions(c_snippet: str) -> list[str]:
    seen = _CDEF.findall(_C_COMMENT.sub("", c_snippet))
    return [f for f in dict.fromkeys(seen) if f not in _NOT_FUNCTIONS]


def _mapped_names(spec: dict) -> tuple[dict[str, dict], set[str]]:
    """symbol_map as {c_name: entry}, plus the set of Rust identifiers it
    claims exist (last path segment — `Scheduler::spawn` -> `spawn`)."""
    entries, rust_names = {}, set()
    for e in spec.get("symbol_map") or []:
        if not isinstance(e, dict) or not e.get("c"):
            continue
        entries[str(e["c"])] = e
        rust = e.get("rust")
        if rust:
            rust_names.add(str(rust).replace("::", ".").split(".")[-1].strip("() "))
    return entries, rust_names


def check_unit(unit: Explanation, c_lines: list[str], spec: dict,
               unit_code: str, crate_text: str,
               symbols: bool = True) -> list[str]:
    """Findings for one unit. Empty means nothing detected (NOT 'correct').

    `symbols=False` drops the symbol-coverage half: with cfg.rustgen_symbol_map
    off no spec has a symbol_map, so every C function would report UNMAPPED and
    the literal findings would be buried under the cap."""
    findings: list[str] = []
    c_src = unit_c_source(unit, c_lines)

    # -- literals -----------------------------------------------------------
    # Absent from the WHOLE crate is a hard loss. Present elsewhere is a move,
    # which is legitimate when the spec declared it (inlined_into) — so it is
    # reported only as a note, and never when a declaration explains it.
    entries, _ = _mapped_names(spec)
    declared_move = any(str(e.get("reason", "")).startswith("inlined_into")
                        for e in entries.values())
    unit_norm, crate_norm = _unescape(unit_code), _unescape(crate_text)
    for frag in output_fragments(c_src):
        if frag in unit_norm:
            continue
        if frag not in crate_norm:
            findings.append(
                f"OUTPUT TEXT LOST: this unit's C emits {frag!r}, which appears"
                " nowhere in the crate. C printf output is the program's"
                " observable behaviour — reproduce it byte-for-byte.")
        elif not declared_move:
            findings.append(
                f"OUTPUT TEXT MOVED: {frag!r} is emitted by this unit's C but"
                " appears in another section, and no symbol_map entry declares"
                " it was inlined elsewhere. Emit it here, or declare where it"
                " went.")

    # -- symbol coverage ----------------------------------------------------
    for fn in c_functions(c_src) if symbols else ():
        entry = entries.get(fn)
        if entry is None:
            findings.append(
                f"UNMAPPED SYMBOL: C function {fn!r} has no symbol_map entry."
                " Say where it went (a Rust path) or why it is gone.")
            continue
        rust, reason = entry.get("rust"), str(entry.get("reason", ""))
        if rust:
            leaf = str(rust).replace("::", ".").split(".")[-1].strip("() ")
            if leaf and not re.search(rf"\bfn\s+{re.escape(leaf)}\b", crate_text):
                findings.append(
                    f"MAPPED BUT MISSING: symbol_map sends {fn!r} to {rust!r},"
                    f" but no `fn {leaf}` exists in the crate.")
        elif not reason.startswith(_VALID_REASONS):
            findings.append(
                f"BAD MAPPING: {fn!r} is unmapped with reason {reason!r};"
                f" must be one of {', '.join(_VALID_REASONS)}.")
        elif reason.startswith("inlined_into"):
            target = reason.split(":", 1)[-1].replace("::", ".").split(".")[-1]
            target = target.strip("() ")
            if target and not re.search(rf"\bfn\s+{re.escape(target)}\b", crate_text):
                findings.append(
                    f"INLINE TARGET MISSING: {fn!r} was declared inlined into"
                    f" {reason.split(':', 1)[-1]!r}, which does not exist.")

    return findings[:MAX_FINDINGS_PER_UNIT]


def call_shape_hints(c_text: str, crate_text: str) -> dict[str, str]:
    """fn -> hint, for C functions still called in Rust but from a different
    number of sites. Ignores rust_count == 0: that is the signature of a C
    helper replaced by a std method (text_equals -> `==`), which is benign and
    was every false positive in the measured sample."""
    hints: dict[str, str] = {}
    cdefs = set(c_functions(c_text))
    for fn in sorted(cdefs):
        if not re.search(rf"\bfn\s+{re.escape(fn)}\b", crate_text):
            continue
        c_n = len(re.findall(rf"\b{re.escape(fn)}\s*\(", c_text)) - 1
        r_n = len(re.findall(rf"\b{re.escape(fn)}\s*\(", crate_text)) - 1
        if r_n > 0 and c_n > 0 and c_n != r_n:
            hints[fn] = (f"C calls {fn}() from {c_n} site(s); the Rust calls it"
                         f" {r_n}. If call sites were merged, check no"
                         f" per-site argument was lost.")
    return hints


def semantic_report(cfg: Config, units: list[Explanation],
                    specs: dict[str, dict], code: dict[str, str],
                    c_lines_by_unit: dict[str, list[str]],
                    types_rs: str) -> SemanticReport:
    """Check the assembled crate and REPORT. Nothing is regenerated.

    There used to be a repair loop here that fed findings back to the model.
    It was measured twice on binary_heap and did nothing: scoring a run's
    crate before and after gave byte-identical divergence on an identical set
    of cases, while costing ~40% of the run's wall clock (7.8min -> 5.5min with
    it off, same 42 LLM calls). The checks themselves are worth keeping — they
    cost ~20ms and no LLM calls — so they stay as a diagnostic.

    `call_shape_hints` rides along as `hints` for the same reason: too noisy to
    act on (measured 1/7), useful to a human reading the record.
    """
    report = SemanticReport()
    crate_text = "\n".join(list(code.values()) + [types_rs])
    symbols = cfg.rustgen_symbol_map
    for u in units:
        lines = c_lines_by_unit.get(u.id)
        if not lines:
            continue
        found = check_unit(u, lines, specs.get(u.id, {}),
                           code.get(u.id, ""), crate_text, symbols=symbols)
        if found:
            report.findings[u.id] = found
        hints = call_shape_hints(unit_c_source(u, lines), crate_text)
        if hints:
            report.hints[u.id] = sorted(hints.values())
    report.total = sum(len(v) for v in report.findings.values())
    return report
