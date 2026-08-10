"""Stub repair — resolve `todo!()` in a finished crate instead of scoring nothing.

WHY THIS EXISTS. `remaining_stubs` voids any crate about to be scored that still
carries a `todo!()`, because a stub is a runtime panic and a panic is not a
translation-quality signal. That gate is right and stays. But it has voided 21
runs, and **13 of those died on one or two stubs** — a whole ~15-minute run
thrown away for a single unresolved call. Resolving even a modest share of stubs
therefore rescues a much larger share of runs.

WHAT IT IS NOT. It is not a retry. The unit that wrote the stub already had its
MTU description and its sibling signatures, and re-asking the same model at
temperature 1.0 with the same context buys nothing. The loop earns its keep only
by supplying information the unit did not have — which is why the handler, not
the prompt, is the substance here. If the `noinfo` control arm scores the same,
this is an expensive retry and should be deleted.

KEYED ON LOCATION, NOT ON THE MESSAGE. Stub messages are free text and 5 of the
80 live stubs carry no message at all (a bare `todo!()` inside a `*_deps`
module, which `illegal_stubs` permits by design). An earlier design routed
stubs by pattern-matching the message; it was fitted after the fact and would
have been blind to the deps-module ones. Every stub is found by where it is —
which section, which function, which signature — and the message is one more
piece of evidence rather than the thing that decides.

THE RULE IS HOW A FIX CAN BE VERIFIED, NOT WHAT CLASS IT IS IN. Three tiers,
and the middle one is the dangerous one:

  1. the compiler can check it — a borrow or ownership restructure. Attempt
     freely; `cargo check` catches a bad one and it is reverted.
  2. evidence can check it — wiring a call to a function that already exists,
     or to the concrete callback the C call site actually passes. The compiler
     is NOT sufficient here: a callback with the right type and the wrong
     behaviour compiles clean and is silently wrong. The model must cite it.
  3. nothing can check it — inventing what a callback should do when nothing
     calls it. Leave the stub.

An earlier version of this design declined whole classes (borrow conflicts,
callbacks) up front. That was wrong in both cases: many borrow errors are fixable
inside one body by ordinary Rust, and a callback is often answerable from the
call site (`setlist.c:292` passes `reduce_seconds` to `cc_array_reduce`).

GIVING UP IS A NORMAL OUTCOME. A stub is an honest failure the gate catches; an
invented body is a wrong translation that scores as ordinary divergence. So the
loop is never measured on how many stubs it closed — that objective is
satisfiable by fabricating — and `give_up` costs nothing.

SHARED TYPES ARE NOT PATCHABLE. The compile loop refuses to rewrite the shared
types block by design, and this loop honours the same rule: a stub there means
stage T failed, and patching it downstream hides the cause (that is exactly how
`binary_heap srvB t2` was misdiagnosed). The ONE exception is a `*_deps` module,
which sits inside the types block textually but is not a type — it is a
placeholder for an external function that `sibling_deps` already rewrites on
every run, so this loop is not the first writer there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm import LLM
from state import Explanation
from rustgen.common import (_blank_literals, _DEPS_MOD, extract_rust,
                            unbalanced_delimiters)

# Must match compile_loop.SHARED — the same region under the same name, so a
# section map can be handed between the two without translation.
SHARED_ID = "__shared__"

REPAIR_ROUNDS = 4
MAX_QUESTIONS = 3

_STUB_ANY = re.compile(r"\b(?:todo|unimplemented)!\s*\(")

# TWO SCHEMAS, NOT ONE, and this is load-bearing rather than tidiness.
#
# The first live run rejected 9 patches in a row, every one "the patch is
# empty". The cause was a single flat schema whose only REQUIRED field was
# `action`: constrained decoding is then free to emit `{"action": "patch"}`
# with no code, and that is exactly what the model did — replying with
# `tier`, `evidence`, `why` AND `reason` (a give_up field) while omitting the
# one field that carries the work. Re-prompting did not help, because the
# grammar still permitted the omission.
#
# `oneOf` per action is the obvious fix and this vLLM's grammar backend cannot
# compile it — it emits an unterminated string and the reply never parses. So
# the decision is split from the payload. DECIDE has no `code` property at all,
# so a patch cannot be half-expressed; PATCH_SCHEMA *requires* `code` with a
# minimum length, so the model cannot claim a patch without writing one.
# Cost is one extra call, and only on rounds that actually patch.
DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ask", "patch", "give_up"]},
        "questions": {
            "type": "array",
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "q": {"type": "string",
                          "enum": ["defines", "signature_of", "callers_of",
                                   "c_source", "mtu", "siblings_of"]},
                    "arg": {"type": "string"},
                },
                "required": ["q", "arg"],
            },
        },
        "reason": {"type": "string"},
        # The justification is committed HERE, before any code is written, and
        # the code comes back as plain fenced Rust in a separate call. See
        # PATCH_NOTE for why source must never travel inside a JSON string.
        "tier": {"type": "integer", "enum": [1, 2, 3]},
        "evidence": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["action"],
}

# RUST NEVER TRAVELS INSIDE A JSON STRING, and this cost a full diagnosis to
# learn. The payload was briefly a JSON `code` field, to force it non-empty
# after the flat-schema bug. Putting source there means the model must escape
# every `"` as `\\"` and every newline as `\\n` under a constrained grammar, and
# it does not: a replayed `populate_projects` came back with every newline
# stripped and every string literal mangled — `"Atlas"` decoded as
# `temporada Atlas`, the quotes replaced by an unrelated token. Structurally
# plausible, completely corrupt.
#
# The rest of this pipeline never had the problem because it never does this:
# `code_stage`, `types_stage` and every repair prompt return a fenced ```rust
# block parsed by `extract_rust`. So the split stays — the DECIDE call commits
# to action and justification in JSON, where the fields are short and safe —
# but the code comes back as plain fenced Rust, with no escaping in the way.
PATCH_NOTE = """\
Now write the replacement.

Reply with ONE Rust function in a ```rust fence, and nothing else:

```rust
{sig} {{
    ... your implementation ...
}}
```

It must be `fn {fn}`, complete, from its signature through its closing brace,
with the `todo!()` replaced by a real implementation.

Do NOT repeat the rest of the section. It is kept exactly as it is, and your
function is spliced in over the old one. A reply carrying the whole section
duplicates every other item in it and will be rejected.

No prose outside the fence. No diff."""

PATCH_NOTE_SECTION = """\
Now write the replacement.

The `todo!()` is not inside a function, so reply with the WHOLE section in a
```rust fence, with the stub replaced. Every item present now must still be
present. Keep it as short as you can — a reply that runs out of room comes back
truncated and is rejected.

No prose outside the fence. No diff."""

STUB_PROMPT = """\
This Rust function is unfinished — its body reaches a `todo!()`, which panics at
run time. The crate compiles, so nothing else will catch this.

SECTION: {section}
FUNCTION: {signature}
{message}
The section's current code:

```rust
{code}
```

{unit_note}
YOUR JOB is to replace the `todo!()` with a real implementation, OR to say that
you cannot. Both are acceptable answers. An honest `give_up` is much better than
a plausible guess: the stub is caught by a gate, an invented body is not, and a
wrong implementation is scored as an ordinary translation bug.

DECIDE BY HOW YOUR FIX COULD BE CHECKED. There are three cases.

TIER 1 — the compiler can check it. Borrow and ownership problems live here:
"cannot borrow `x` as immutable while also borrowed as mutable" is usually
fixable inside this one body by taking an index instead of a reference, or by
copying what you need out before taking the mutable borrow, or by letting the
first borrow end before the second begins. Try these. If your version does not
compile it will be rejected and you can try again, so the risk is low.

TIER 2 — evidence can check it, and the compiler CANNOT. Wiring a call to a
function that already exists, or supplying the concrete callback the C passes at
the call site. The danger here is specific: a function with the right type and
the wrong behaviour compiles perfectly and is silently wrong. So you must cite
the evidence — ask `defines` or `callers_of` first and put what you found in the
`evidence` field. A tier-2 patch with no evidence will be rejected.

TIER 3 — nothing can check it. You are being asked what a caller-supplied
callback should do, nothing in this project calls it, and there is no C to point
at. Do not guess. Reply `give_up`.

ASK BEFORE YOU EDIT. You can request facts about the rest of the crate:

  {{"action": "ask", "questions": [{{"q": "defines", "arg": "scheduler_print_report"}}]}}

    defines       — is this function defined anywhere in the crate, and where
    signature_of  — its exact signature
    callers_of    — who calls it, and with what arguments (includes the C call
                    sites, which is where a concrete callback is named)
    c_source      — the original C for a named function
    mtu           — one unit's description and invariants (arg is its id)
    siblings_of   — the sibling signatures a unit was given (arg is its id)

TO SUBMIT, reply with the WHOLE section, not a diff, and not just the one
function. Every item that is in the section now must still be there:

  {{"action": "patch", "tier": 2,
    "evidence": "defines(scheduler_print_report) says it is defined at line 812",
    "code": "pub(crate) fn report(&self) -> String {{ scheduler_print_report(self) }}",
    "why": "the sibling function exists; the stub was deferring to it"}}

TO DECLINE:

  {{"action": "give_up",
    "reason": "the parameter is a caller-supplied comparator, nothing in this
               crate calls this function, and no C call site names a concrete
               one — any body I write here would be invented"}}
"""

ANSWER_NOTE = """\
Answers to your questions:

{answers}

Now reply again with `patch`, `ask` or `give_up`."""


_FN_START = re.compile(r"\bfn\s+(\w+)\s*(?:<[^>]*>)?\s*\(")
_FN_MODIFIERS = re.compile(
    r"[ \t]*(?:pub(?:\([^)]*\))?\s+|async\s+|unsafe\s+|const\s+|extern\s+\"[^\"]*\"\s+)*$")


def function_span(code: str, name: str) -> tuple[int, int] | None:
    """(start, end) of `fn name`'s complete definition, or None.

    `start` backs up over visibility and qualifier keywords on the same line so
    the span covers `pub(crate) async fn f(...) {...}`, not just the `fn`.
    """
    blanked = _blank_literals(code)
    for m in _FN_START.finditer(blanked):
        if m.group(1) != name:
            continue
        start = m.start()
        line_start = blanked.rfind("\n", 0, start) + 1
        if _FN_MODIFIERS.fullmatch(blanked[line_start:start]):
            start = line_start
        i = blanked.find("{", m.end() - 1)
        if i == -1:
            return None                      # a signature, not a definition
        depth, k = 0, i
        while k < len(blanked):
            if blanked[k] == "{":
                depth += 1
            elif blanked[k] == "}":
                depth -= 1
                if depth == 0:
                    return (start, k + 1)
            k += 1
        return None                          # unterminated
    return None


def splice_function(section: str, name: str, new_fn: str) -> tuple[str, str]:
    """Replace ONE function's definition in `section`. Returns (section, error).

    This is why the loop asks for a function rather than a whole section, and
    the reason is not brevity. Asking for the section back meant regenerating
    ~2k characters to change three lines, and the reply reliably ran past its
    token budget — at which point constrained decoding must still emit valid
    JSON, so it closes the string and hands back well-formed JSON containing
    TRUNCATED Rust. That is what "1 unclosed '{'" was, three attempts running.
    It also cost minutes per attempt and tripped llm.ask's doubling retry
    (4000 -> 8000 -> 16000), so a single stub could burn half an hour.

    Splicing fixes both, and buys a stronger property than the check it
    replaces: `validate_stub_patch` DETECTS a repair that drops sibling
    functions, whereas here everything outside the target span is preserved
    byte-for-byte, so the repair-deletes-methods failure cannot occur at all.
    Fix it at the writer, not at the gate.
    """
    if not new_fn.strip():
        return section, "the replacement is empty"
    span = function_span(section, name)
    if span is None:
        return section, f"could not locate `fn {name}` in the section"
    bad = unbalanced_delimiters(new_fn)
    if bad:
        return section, (f"the replacement does not parse ({bad}) — most likely "
                         f"it was cut off. Reply with ONLY `fn {name}`.")
    names = [mm.group(1) for mm in _FN_START.finditer(_blank_literals(new_fn))]
    if name not in names:
        return section, (f"the replacement does not define `fn {name}` "
                         f"(it defines: {', '.join(names) or 'nothing'})")
    s, e = span
    spliced = section[:s] + new_fn.strip() + section[e:]
    # A model told to return "the function" often returns the whole section
    # anyway. Splicing that over one function silently DUPLICATES every
    # sibling, which compiles as E0428 at best and is invisible in a diff at
    # worst — the resulting section still contains every name it should, so a
    # "nothing was lost" check passes it. Catch it on the way out: no function
    # may be defined twice in the spliced result.
    after = [mm.group(1) for mm in _FN_START.finditer(_blank_literals(spliced))]
    dupes = sorted({n for n in after if after.count(n) > 1})
    if dupes:
        return section, (f"the replacement duplicated {', '.join(dupes[:4])} — "
                         f"reply with ONLY `fn {name}`, not the whole section")
    return spliced, ""


@dataclass
class Stub:
    """One unresolved `todo!()`, identified by where it is."""
    section: str                 # section id (unit id, or SHARED)
    function: str = ""           # enclosing fn name, "" if not inside one
    signature: str = ""
    in_deps_module: bool = False
    in_shared: bool = False
    message: str | None = None   # the todo!("...") text, None if bare
    line: int = 0

    @property
    def patchable(self) -> bool:
        """Shared types are off limits — except a `*_deps` module, which lives
        inside that block textually but is a placeholder for an external
        function, not a type, and which `sibling_deps` already rewrites."""
        return (not self.in_shared) or self.in_deps_module


def _deps_spans(blanked: str) -> list[tuple[int, int]]:
    spans = []
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
    return spans


def find_open_stubs(sections: dict[str, str], shared_id: str = "") -> list[Stub]:
    """Every reachable `todo!()`/`unimplemented!()`, keyed on its location.

    Finds the bare ones too. `illegal_stubs` deliberately permits a bare stub
    inside a `*_deps` module and `remaining_stubs` deliberately does not — this
    agrees with `remaining_stubs`, because the question here is the scoring
    one: will this crate panic.
    """
    out: list[Stub] = []
    for sid, code in sections.items():
        if not code:
            continue
        blanked = _blank_literals(code)
        spans = _deps_spans(blanked)
        for m in _STUB_ANY.finditer(blanked):
            # the real (unblanked) text, to recover a documented message
            tail = code[m.end():m.end() + 400]
            msg = None
            qm = re.match(r'\s*"((?:[^"\\]|\\.)*)"', tail)
            if qm:
                msg = qm.group(1)
            fn, sig = "", ""
            # Nearest preceding fn signature. The generic parameter list is not
            # optional in this pattern: the first live run attributed a stub
            # inside `pub fn callback<T>(...)` to the `fmt` above it, because
            # the old regex required `(` immediately after the name and so
            # skipped every generic function. Misattribution is not cosmetic —
            # the name is what the prompt shows and what the "did the patch
            # keep the function" check tests.
            for fm in re.finditer(r"\bfn\s+(\w+)\s*(?:<[^>]*>)?\s*\(", blanked):
                if fm.start() < m.start():
                    fn = fm.group(1)
                    sig = code[fm.start():fm.start() + 200].split("{")[0].strip()
                else:
                    break
            out.append(Stub(
                section=sid, function=fn, signature=sig,
                in_deps_module=any(a <= m.start() <= b for a, b in spans),
                in_shared=(sid == shared_id),
                message=msg,
                line=blanked.count("\n", 0, m.start()) + 1))
    return out


class StubContext:
    """Answers the loop's questions from records that already exist.

    Nothing new is collected: the crate text, the chunker's call sites, the C
    source, the MTU descriptions and the sibling registry are all already in the
    run. It reads those and nothing else — it has no filesystem access of its
    own, which is what keeps the DARPA test vectors structurally out of reach.

    `blind=True` answers every question with "no information available". That is
    the control arm: if the loop does as well blind, the retrieval is not doing
    anything and this module is an expensive retry.
    """

    def __init__(self, crate_text: str, units: list[Explanation],
                 c_sources: dict[str, str] | None = None,
                 call_texts: dict[str, list[str]] | None = None,
                 specs: dict[str, dict] | None = None,
                 blind: bool = False):
        self.crate = crate_text
        self.blanked = _blank_literals(crate_text)
        self.units = {u.id: u for u in units}
        self.c_sources = c_sources or {}
        self.call_texts = call_texts or {}
        self.specs = specs or {}
        self.blind = blind

    # ---- individual questions

    def defines(self, name: str) -> str:
        name = name.strip().split("::")[-1]
        hits = [self.blanked.count("\n", 0, m.start()) + 1
                for m in re.finditer(rf"\bfn\s+{re.escape(name)}\b", self.blanked)]
        if not hits:
            return (f"Nothing in this crate defines `{name}`. If the stub is "
                    f"waiting for it, nobody is going to provide it.")
        return (f"`{name}` IS defined in this crate, at line(s) "
                f"{', '.join(str(h) for h in hits[:6])}.")

    def signature_of(self, name: str) -> str:
        name = name.strip().split("::")[-1]
        m = re.search(rf"\bfn\s+{re.escape(name)}\s*(\([^{{;]*)", self.blanked)
        if not m:
            return f"(no function named `{name}` in this crate)"
        return self.crate[m.start():m.start() + 250].split("{")[0].strip()

    def callers_of(self, name: str) -> str:
        name = name.strip().split("::")[-1]
        out = []
        for m in re.finditer(rf"\b{re.escape(name)}\s*\(", self.blanked):
            # skip the definition itself
            pre = self.blanked[max(0, m.start() - 8):m.start()]
            if pre.rstrip().endswith("fn"):
                continue
            line = self.blanked.count("\n", 0, m.start()) + 1
            text = self.crate[m.start():m.start() + 120].split("\n")[0]
            out.append(f"  crate line {line}: {text}")
        for site in self.call_texts.get(name, [])[:6]:
            out.append(f"  C call site: {site}")
        if not out:
            return (f"Nothing calls `{name}` — not in this crate and not in the "
                    f"C. There is no evidence of what it should do.")
        return f"Calls to `{name}`:\n" + "\n".join(out[:10])

    def c_source_of(self, name: str) -> str:
        name = name.strip()
        for src in self.c_sources.values():
            m = re.search(rf"^[^\n]*\b{re.escape(name)}\s*\([^;]*?\)\s*\{{",
                          src, re.M)
            if not m:
                continue
            depth, k = 0, src.index("{", m.start())
            while k < len(src):
                if src[k] == "{":
                    depth += 1
                elif src[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            return src[m.start():k + 1][:2500]
        return f"(no definition of `{name}` found in the C source)"

    def mtu(self, unit_id: str) -> str:
        from rustgen.common import unit_block
        want = unit_id.strip()
        u = self.units.get(want) or self.units.get(want.split("__")[-1])
        return unit_block(u) if u else f"(no unit with id `{unit_id}`)"

    def siblings_of(self, unit_id: str) -> str:
        spec = self.specs.get(unit_id.strip()) or {}
        sigs = spec.get("signatures") or []
        if not sigs:
            return f"(no recorded sibling signatures for `{unit_id}`)"
        return "\n".join(f"  {s}" for s in
                         (sigs if isinstance(sigs, list) else [str(sigs)])[:20])

    def answer(self, q: dict) -> str:
        kind, arg = str(q.get("q", "")), str(q.get("arg", ""))
        if self.blind:
            return f"{kind}({arg}): no information available"
        fn = {"defines": self.defines, "signature_of": self.signature_of,
              "callers_of": self.callers_of, "c_source": self.c_source_of,
              "mtu": self.mtu, "siblings_of": self.siblings_of}.get(kind)
        if fn is None:
            return f"({kind}: unknown question)"
        return f"{kind}({arg}):\n{fn(arg)}"


@dataclass
class StubReport:
    considered: int = 0
    skipped_shared: int = 0
    resolved: int = 0
    gave_up: int = 0
    rejected: int = 0
    errored: int = 0
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (f"[stubs] {self.considered} considered, {self.resolved} resolved, "
                f"{self.gave_up} declined, {self.rejected} patch(es) rejected, "
                f"{self.skipped_shared} in shared types (untouchable)")


def validate_stub_patch(old: str, new: str, stub: Stub) -> str:
    """"" if the patched section is acceptable, else why not.

    Structural only — the compile check is the caller's job, because it needs
    the whole crate reassembled and cargo run over it. Kept separate so the
    cheap checks can reject a hopeless patch before paying for cargo.
    """
    if not new.strip():
        return "the patch is empty"
    bad = unbalanced_delimiters(new)
    if bad:
        return f"patch does not parse: {bad}"
    # the point of the exercise
    if stub.function and stub.function not in new:
        return f"the patch dropped `{stub.function}`, the function being fixed"
    old_fns = set(re.findall(r"\bfn\s+(\w+)", _blank_literals(old)))
    new_fns = set(re.findall(r"\bfn\s+(\w+)", _blank_literals(new)))
    gone = sorted(old_fns - new_fns)
    if gone:
        return (f"the patch dropped {len(gone)} function(s) the section had: "
                f"{', '.join(gone[:6])}")
    return ""


async def repair_stubs(llm: LLM, sections: dict[str, str],
                       units: list[Explanation], ctx: StubContext, *,
                       shared_id: str = "",
                       compile_check=None,
                       max_tokens: int = 4000,
                       rounds: int = REPAIR_ROUNDS,
                       ) -> tuple[dict[str, str], StubReport]:
    """Resolve what can be resolved. Returns (sections, report).

    `compile_check(sections) -> str` reassembles and runs cargo, returning ""
    when clean. A patch that fails it is reverted — a wiring fix with the wrong
    arity would otherwise turn a STUB_CRATE into a BUILD_FAILED, which is not an
    improvement. If it is None the compile gate is skipped and only the
    structural checks apply.

    Sections are only ever replaced by a patch that passed every check, so on
    any failure the caller is left exactly where it started.
    """
    out = dict(sections)
    report = StubReport()
    # One pass per FUNCTION, not per `todo!()`. The first live run found three
    # stubs inside a single `populate_projects` body and repaired it three
    # times over, tripling the calls and the rejections for one defect — the
    # patch replaces the whole section, so the first attempt already had to
    # deal with all three. Dedupe on (section, function); a stub not inside any
    # function keeps its line number as the key so it is not folded in with
    # unrelated ones.
    stubs, seen = [], set()
    for s in find_open_stubs(out, shared_id):
        key = (s.section, s.function or f"@{s.line}")
        if key in seen:
            continue
        seen.add(key)
        stubs.append(s)

    for stub in stubs:
        if not stub.patchable:
            report.skipped_shared += 1
            # same keys as every other detail: these records are swept later,
            # and a row missing a column is a KeyError in whatever reads it
            report.details.append({"section": stub.section,
                                   "function": stub.function,
                                   "line": stub.line,
                                   "bare": stub.message is None,
                                   "questions": [], "tier": None,
                                   "outcome": "skipped: shared types"})
            continue
        # the section may have been rewritten by an earlier stub's patch
        current = out.get(stub.section, "")
        if not _STUB_ANY.search(_blank_literals(current)):
            continue
        report.considered += 1
        detail = {"section": stub.section, "function": stub.function,
                  "line": stub.line, "bare": stub.message is None,
                  "questions": [], "outcome": "exhausted", "tier": None}

        msg = (f"The stub's own note: {stub.message!r}\n" if stub.message
               else "The stub carries no message.\n")
        unit = ctx.units.get(stub.section) or \
            ctx.units.get(stub.section.split("__")[-1])
        unit_note = ""
        if unit:
            from rustgen.common import unit_block
            unit_note = ("What this unit is supposed to do:\n\n"
                         + unit_block(unit) + "\n\n")
        prompt = STUB_PROMPT.format(
            section=stub.section, signature=stub.signature or stub.function,
            message=msg, code=current, unit_note=unit_note)

        for _ in range(rounds):
            try:
                reply = await llm.ask_json(prompt, max_tokens=max_tokens,
                                           schema=DECIDE_SCHEMA)
            except Exception as e:
                detail["outcome"] = f"error: {type(e).__name__}: {e}"
                report.errored += 1
                break
            if not isinstance(reply, dict):
                prompt += "\n\nReply with a JSON object."
                continue
            action = str(reply.get("action", "")).strip()

            if action == "give_up":
                detail["outcome"] = "gave_up"
                detail["why"] = str(reply.get("reason", ""))[:300]
                report.gave_up += 1
                break

            if action == "ask":
                qs = reply.get("questions") or []
                if not isinstance(qs, list) or not qs:
                    prompt += "\n\nYou asked nothing. Ask, patch or give_up."
                    continue
                answers = []
                for q in qs[:MAX_QUESTIONS]:
                    if isinstance(q, dict):
                        detail["questions"].append(f"{q.get('q')}({q.get('arg')})")
                        answers.append(ctx.answer(q))
                prompt += "\n\n" + ANSWER_NOTE.format(answers="\n\n".join(answers))
                continue

            if action == "patch":
                tier = reply.get("tier")
                evidence = str(reply.get("evidence") or "").strip()
                # tier 2 is the case the compiler cannot adjudicate: a
                # type-correct wrong callback compiles clean. Evidence is the
                # only check there is, so refuse an uncited claim BEFORE paying
                # for the generation.
                if tier == 2 and not evidence:
                    report.rejected += 1
                    prompt += ("\n\nREJECTED: you claimed tier 2 but cited no "
                               "evidence. Ask `defines` or `callers_of` and put "
                               "what you found in `evidence`, or use another tier.")
                    continue
                # second call: plain text, fenced Rust, no JSON in the way
                note = (PATCH_NOTE.format(fn=stub.function,
                                          sig=stub.signature or f"fn {stub.function}")
                        if stub.function else PATCH_NOTE_SECTION)
                try:
                    raw = await llm.ask(prompt + "\n\n" + note,
                                        max_tokens=max_tokens)
                except Exception as e:
                    detail["outcome"] = f"error: {type(e).__name__}: {e}"
                    report.errored += 1
                    break
                got = extract_rust(raw or "")

                if stub.function:
                    # splice: everything outside this one function is preserved
                    # byte-for-byte, so a repair CANNOT drop a sibling item
                    cand, serr = splice_function(current, stub.function, got)
                    if serr:
                        report.rejected += 1
                        detail.setdefault("rejections", []).append(serr)
                        prompt += (f"\n\nYour patch was REJECTED: {serr}\n"
                                   f"Reply again with only `fn {stub.function}`.")
                        continue
                else:
                    cand = got
                bad = validate_stub_patch(current, cand, stub)
                if not bad and _STUB_ANY.search(_blank_literals(cand)):
                    # a patch may legitimately leave OTHER stubs in the section,
                    # but not the one it was asked to fix
                    if stub.function:
                        m = re.search(rf"\bfn\s+{re.escape(stub.function)}\b",
                                      _blank_literals(cand))
                        if m:
                            nxt = re.search(r"\bfn\s+\w+", _blank_literals(cand)[m.end():])
                            end = m.end() + (nxt.start() if nxt else 10 ** 9)
                            if _STUB_ANY.search(_blank_literals(cand)[m.start():end]):
                                bad = "the patch leaves the same function stubbed"
                if bad:
                    report.rejected += 1
                    detail.setdefault("rejections", []).append(bad)
                    prompt += (f"\n\nYour patch was REJECTED: {bad}\n"
                               f"Fix that and reply with the whole section again.")
                    continue

                trial = dict(out)
                trial[stub.section] = cand
                cerr = compile_check(trial) if compile_check else ""
                if cerr:
                    report.rejected += 1
                    detail.setdefault("rejections", []).append(f"compile: {cerr[:200]}")
                    prompt += (f"\n\nYour patch did not compile:\n{cerr[:1500]}\n"
                               f"Fix that and reply with the whole section again.")
                    continue

                out[stub.section] = cand
                detail["outcome"] = "resolved"
                detail["tier"] = tier
                detail["evidence"] = evidence[:300]
                detail["why"] = str(reply.get("why", ""))[:300]
                report.resolved += 1
                break

            prompt += "\n\nUnknown action. Use ask, patch or give_up."

        report.details.append(detail)

    return out, report
