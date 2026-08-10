"""Stage T repair — fix a bad shared-types block instead of dying on it.

WHY THIS EXISTS. `synthesize_types` already detects a bad block (`illegal_stubs`
for a phantom API, `illegal_type_bodies` for a method a unit should own) and
already retries. What it does when the retries run out is regenerate the whole
block from the same prompt, twice, and then give up — which at temperature 1.0
is re-rolling the dice, not repairing anything.

That outcome is not survivable. Measured over 255 recorded run logs, EVERY run
where `[types] ... still present after 2 retries` appears ended dead: 4
BUILD_FAILED, 3 STUB_CRATE, none scored. 6 of the 7 are `binary_heap`, which is
also the project with the worst stub rate. The damage is out of proportion to
the defect because stage T is upstream of everything: one stubbed `impl` block
hands all ~40 units a phantom API to defer to, each assumes a sibling owns it,
and the ones that implement it anyway collide as E0592. The compile loop cannot
undo that — it will not rewrite the shared block by design — so the only move
left is deleting real code, which `emptied_blocks` correctly refuses.
`binary_heap base_srvB_t2` deadlocked exactly there: nine repairs refused, zero
accepted.

THE FIX IS USUALLY DELETION, NOT COMPLETION. Stage T is told "types and stubs
only". A stubbed method in the shared block is therefore wrong in itself, not
merely incomplete: either a unit owns that behaviour and will emit its own
copy — in which case the stub is a phantom that misleads every other unit — or
nobody owns it and it should never have been declared. Filling it in is usually
the wrong repair, and the prompt says so.

VALIDATION IS A REGRESSION TEST, NOT AN ABSOLUTE ONE. A per-file types block is
generated with the project's shared types as context, so compiling it alone can
fail on cross-file references that were never this block's fault. Every check
here compares the repair against the block it replaced and rejects only what the
repair made WORSE. Same asymmetry as `parse_regression`: the thing being
repaired is broken by definition, and a check on the end state alone would
refuse to let it be touched at all.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from llm import LLM
from state import Explanation
from rustgen.common import (extract_rust, illegal_stubs, illegal_type_bodies,
                            lost_type_definitions, unbalanced_delimiters,
                            unit_block)

# Question/answer exchanges before the loop gives up. Each round is one LLM
# call; a stuck block should cost a handful of calls, not a stage.
REPAIR_ROUNDS = 4
MAX_QUESTIONS = 3

CARGO_TIMEOUT = 180

# Constrained decoding: the reply is one of three shapes. `remove` is not an
# action — the model returns the corrected block as `patch` and deletion is
# just what that block does not contain. A structured item-remover would need a
# Rust surgeon; `lost_type_definitions` gets the same protection for free by
# checking what survived rather than by controlling what was cut.
# Split for the same reason as stub_repair's — see the long note there. A flat
# schema requiring only `action` lets constrained decoding emit
# `{"action": "patch"}` with no code, which is what sank the stub loop's first
# live run (9 rejections, all "the patch is empty"). `oneOf` per action would
# be the clean fix and this vLLM's grammar backend cannot compile it. So DECIDE
# carries no `code` property at all and PATCH_SCHEMA requires one.
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
                          "enum": ["mentions", "c_source", "mtu"]},
                    "arg": {"type": "string"},
                },
                "required": ["q", "arg"],
            },
        },
        "reason": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["action"],
}

# Rust never travels inside a JSON string — see the long note in
# stub_repair.py. Under a constrained grammar the model fails to escape `"` and
# newlines, and a replayed section came back with every newline stripped and
# every string literal mangled. The DECIDE call stays JSON (short, safe
# fields); the block comes back as plain fenced Rust, like every other prompt
# in this pipeline.
PATCH_NOTE = """\
Now write the corrected block.

Reply with the whole types block in a ```rust fence, and nothing else — the
rejected items removed, no explanation outside the fence, no diff.

Every `struct`, `enum`, `trait` and `type` in the block above must still be
there. Cut methods, never types."""

REPAIR_PROMPT = """\
The shared Rust types block below was REJECTED by an automatic check:

    {problem}

Stage T defines the vocabulary every behavioural unit is then generated
against. It must contain type definitions, the error enum with its Display and
Error impls, and `pub mod <stem>_deps` stubs for external domain functions.
Nothing else.

WHAT IS ALMOST ALWAYS THE RIGHT FIX: DELETE, DO NOT COMPLETE.

A method stubbed with `todo!()` in this block is not an unfinished good idea,
it is a defect in itself. Every unit reads this block, sees the method already
declared, concludes a sibling implements it, and writes nothing. The crate then
builds clean and panics at run time. If a unit genuinely owns that behaviour it
will emit its own copy, and yours collides with it (Rust rejects two inherent
methods of one name whatever their signatures). So remove the stubbed methods,
and remove the `impl` block itself if nothing legitimate is left in it.

The same goes for a method here with a REAL body: writing a unit's behaviour
here does not help that unit, it duplicates and collides with it.

WHAT YOU MUST NOT DO:

- Do not drop a type definition. Every `struct`, `enum`, `trait` and `type`
  present below must still be present in your reply. Units are already being
  written against these names; removing one breaks every reference to it. Cut
  methods, never types.
- Do not delete the error enum's `impl std::fmt::Display` or
  `impl std::error::Error`. Those are required and are not unit behaviour.
- Do not touch a `pub mod <stem>_deps` module. A bare `todo!()` inside one is
  correct and is resolved later.

YOU MAY ASK FIRST. If you cannot tell whether some behaviour belongs to a unit,
ask before editing. `mentions` is usually the question that settles it: if a
unit's description names the type or method, that unit will implement it and
your stub must go.

  {{"action": "ask", "questions": [{{"q": "mentions", "arg": "Scheduler"}}]}}

    mentions  — which units' descriptions name this type or method
    c_source  — the original C for a named function
    mtu       — one unit's full description and invariants (arg is its id)

To submit the corrected block, reply with the WHOLE block, not a diff:

  {{"action": "patch", "code": "pub(crate) struct Scheduler {{ ... }}",
    "why": "removed the four stubbed methods on Scheduler; the scheduler unit
            names all four in its description and will implement them"}}

If the block cannot be repaired without guessing at behaviour, say so and it
will be left alone. That is a legitimate answer and is preferred to inventing
something:

  {{"action": "give_up",
    "reason": "the block declares no phantom API; the rejected item is the
               error enum's Display impl, which is required"}}

THE BLOCK:

```rust
{block}
```
"""

ANSWER_NOTE = """\
Answers to your questions:

{answers}

Now reply again with `patch`, `ask` or `give_up`."""


class TypesContext:
    """Answers the repair loop's questions from records that already exist.

    Deliberately narrow. At stage T there is no crate, no unit code and no
    responsibility assignment yet — only the MTU descriptions, the C, and the
    block itself. `mentions` is the load-bearing one: it is what distinguishes
    "a unit owns this, delete your stub" from "nobody owns this, it should not
    have been declared", which is the whole decision the repair has to make.

    It reads pipeline records and the C source, and nothing else. It has no
    filesystem access of its own, which is also what keeps the DARPA test
    vectors structurally out of reach of anything built on top of it.
    """

    def __init__(self, units: list[Explanation], c_source: str = ""):
        self.units = units
        self.c_source = c_source

    @staticmethod
    def _mentions_in(name: str, hay: str) -> bool:
        """Does this prose plausibly describe `name`?

        An exact word match is not enough and the gap is not academic: unit
        descriptions are English, identifiers are not. A method called
        `dispatch` is described as "dispatches it", `scheduler_print_report`
        as "prints a report of the scheduler". A literal `\\bdispatch\\b` misses
        both, and a miss here tells the model nothing owns the behaviour.

        Two chances, deliberately generous. Errors in EITHER direction lead to
        the same action — a false positive says "a unit owns this, delete your
        stub", a false negative says "nobody owns this, do not declare it" —
        and deleting the stub is the correct repair either way. So this is
        tuned to be useful rather than precise.
        """
        if re.search(rf"\b{re.escape(name)}(?:s|es|ed|ing|d)?\b", hay, re.I):
            return True
        # snake_case and CamelCase both split into the words prose would use
        words = [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z0-9]+",
                                               name) if len(w) > 2]
        return bool(words) and all(
            re.search(rf"\b{re.escape(w)}\w*\b", hay, re.I) for w in words)

    def mentions(self, name: str) -> str:
        name = name.strip().split("::")[-1]
        if not name:
            return "(empty name)"
        hits = []
        for u in self.units:
            hay = (u.text or "") + " " + " ".join(u.invariants or [])
            if self._mentions_in(name, hay):
                hits.append(u.id)
        if not hits:
            return (f"No unit description mentions `{name}`. Nothing downstream "
                    f"will implement it — it should not be declared here.")
        return (f"{len(hits)} unit(s) mention `{name}`: {', '.join(hits[:12])}. "
                f"Those units will implement it; a stub here collides with them.")

    def c_source_of(self, name: str) -> str:
        name = name.strip()
        if not self.c_source or not name:
            return "(no C source available)"
        # the function's definition plus a little context either side
        m = re.search(rf"^[^\n]*\b{re.escape(name)}\s*\([^;]*?\)\s*\{{",
                      self.c_source, re.M)
        if not m:
            return f"(no definition of `{name}` found in the C source)"
        start = m.start()
        depth, k = 0, self.c_source.index("{", m.start())
        while k < len(self.c_source):
            if self.c_source[k] == "{":
                depth += 1
            elif self.c_source[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        return self.c_source[start:k + 1][:2000]

    def mtu(self, unit_id: str) -> str:
        want = unit_id.strip().split("__")[-1]
        for u in self.units:
            if u.id == unit_id.strip() or u.id == want:
                return unit_block(u)
        return f"(no unit with id `{unit_id}`)"

    def answer(self, q: dict) -> str:
        kind, arg = str(q.get("q", "")), str(q.get("arg", ""))
        if kind == "mentions":
            return f"mentions({arg}): {self.mentions(arg)}"
        if kind == "c_source":
            return f"c_source({arg}):\n{self.c_source_of(arg)}"
        if kind == "mtu":
            return f"mtu({arg}):\n{self.mtu(arg)}"
        return f"({kind}: unknown question)"


def cargo_error_signatures(code: str, context_rs: str = "",
                           timeout: int = CARGO_TIMEOUT) -> set[str] | None:
    """Normalised rustc error signatures for `context_rs + code`, or None if
    cargo could not be run at all.

    None is NOT "no errors" — it means the check did not happen, and the caller
    must not read it as a pass. A gate that silently degrades to permissive is
    how a stubbed crate gets scored; this one degrades to "I don't know" and
    the caller falls back to the cheap structural checks.

    Line numbers are stripped so the same error before and after a repair
    compares equal even when the edit moved it.
    """
    if not shutil.which("cargo"):
        return None
    tmp = Path(tempfile.mkdtemp(prefix="typescheck_"))
    try:
        (tmp / "src").mkdir()
        (tmp / "Cargo.toml").write_text(
            '[package]\nname="typescheck"\nversion="0.1.0"\nedition="2021"\n')
        (tmp / "src" / "lib.rs").write_text(
            (context_rs + "\n\n" if context_rs else "") + code)
        try:
            proc = subprocess.run(
                ["cargo", "check", "--offline", "--message-format=short"],
                cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None
        sigs = set()
        for line in (proc.stderr or "").splitlines():
            if ": error" not in line:
                continue
            # "src/lib.rs:12:5: error[E0412]: cannot find type `Foo` ..."
            msg = line.split(": error", 1)[1]
            msg = re.sub(r"\b\d+\b", "N", msg)
            sigs.add(msg.strip())
        return sigs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compile_regression(old: str, new: str, context_rs: str = "",
                       checker=cargo_error_signatures) -> str:
    """"" if `new` introduces no rustc error `old` did not already have.

    Asymmetric on purpose. A per-file types block is written against the
    project's shared types, so compiling it in isolation can report errors that
    were never this block's fault — and the block reaching a repair is broken
    anyway. Blaming the repair for pre-existing errors would reject every
    correct fix. Only NEW signatures count.
    """
    new_sigs = checker(new, context_rs)
    if new_sigs is None:
        return ""                     # cargo unavailable: not a verdict
    if not new_sigs:
        return ""
    old_sigs = checker(old, context_rs)
    if old_sigs is None:
        return ""
    introduced = sorted(new_sigs - old_sigs)
    if not introduced:
        return ""
    shown = "; ".join(introduced[:4]) + (" ..." if len(introduced) > 4 else "")
    return f"{len(introduced)} new compile error(s): {shown}"


def validate_repair(old: str, new: str, context_rs: str = "",
                    checker=cargo_error_signatures) -> str:
    """"" if the repaired block is acceptable, else why it is not.

    Four checks, cheapest first, and the order matters: an unbalanced block
    makes every later check meaningless, and there is no point paying for cargo
    on a block that still carries the defect it was sent to fix.
    """
    if not new.strip():
        return "the repair is empty"
    problem = unbalanced_delimiters(new)
    if problem:
        return f"repair does not parse: {problem}"
    problem = lost_type_definitions(old, new)
    if problem:
        return problem
    problem = illegal_stubs(new) or illegal_type_bodies(new)
    if problem:
        return f"repair still carries the original defect: {problem}"
    return compile_regression(old, new, context_rs, checker)


async def repair_types_block(llm: LLM, types_rs: str, problem: str,
                             units: list[Explanation], *,
                             context_rs: str = "", c_source: str = "",
                             max_tokens: int = 3000,
                             rounds: int = REPAIR_ROUNDS,
                             checker=cargo_error_signatures,
                             ) -> tuple[str, dict]:
    """Try to repair a rejected types block. Returns (block, report).

    On failure the ORIGINAL block comes back unchanged, so the caller's
    behaviour is exactly what it was before this module existed: record the
    failure, let the run die loudly. The loop can only ever turn a dead run
    into a live one; it has no path to making a good run worse.
    """
    ctx = TypesContext(units, c_source)
    report = {"rounds": 0, "questions": [], "action": None,
              "rejected": [], "repaired": False, "why": ""}
    prompt = REPAIR_PROMPT.format(problem=problem, block=types_rs)

    for _ in range(rounds):
        report["rounds"] += 1
        try:
            reply = await llm.ask_json(prompt, max_tokens=max_tokens,
                                       schema=DECIDE_SCHEMA)
        except Exception as e:                      # transport, JSON, anything
            report["action"] = "error"
            report["why"] = f"{type(e).__name__}: {e}"
            return types_rs, report
        if not isinstance(reply, dict):
            report["rejected"].append("reply was not an object")
            continue
        action = str(reply.get("action", "")).strip()

        if action == "give_up":
            report["action"] = "give_up"
            report["why"] = str(reply.get("reason", ""))[:400]
            return types_rs, report

        if action == "ask":
            qs = reply.get("questions") or []
            if not isinstance(qs, list) or not qs:
                report["rejected"].append("ask with no questions")
                prompt += "\n\nYou asked nothing. Ask a question or reply with a patch."
                continue
            answers = []
            for q in qs[:MAX_QUESTIONS]:
                if isinstance(q, dict):
                    report["questions"].append(f"{q.get('q')}({q.get('arg')})")
                    answers.append(ctx.answer(q))
            prompt += "\n\n" + ANSWER_NOTE.format(answers="\n\n".join(answers))
            continue

        if action == "patch":
            why = str(reply.get("why", ""))[:400]
            try:
                raw = await llm.ask(prompt + "\n\n" + PATCH_NOTE,
                                    max_tokens=max_tokens)
            except Exception as e:
                report["action"] = "error"
                report["why"] = f"{type(e).__name__}: {e}"
                return types_rs, report
            candidate = extract_rust(raw or "")
            bad = validate_repair(types_rs, candidate, context_rs, checker)
            if bad:
                report["rejected"].append(bad)
                prompt += (f"\n\nYour previous patch was REJECTED: {bad}\n"
                           f"Fix that and reply again with the whole block.")
                continue
            report["action"] = "patch"
            report["repaired"] = True
            report["why"] = why      # committed in the DECIDE call, before the code
            return candidate, report

        report["rejected"].append(f"unknown action {action!r}")
        prompt += "\n\nUnknown action. Use ask, patch or give_up."

    report["action"] = report["action"] or "exhausted"
    return types_rs, report
