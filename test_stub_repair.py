"""Stub repair: resolve `todo!()` in a finished crate, or leave it honestly.

    PYTHONPATH=. venv/bin/python test_stub_repair.py

Why this exists. `remaining_stubs` has voided 21 runs and 13 of them died on
one or two stubs. The loop is worth building because of that shape — but only
if it cannot buy its resolutions with fabrication, which is the one way it
could make the experiment record WORSE rather than better. A stub is a loud
failure the gate catches; an invented body is a wrong translation scored as
ordinary divergence.

So the tests that matter most are the refusals:

  - a patch that drops other functions is rejected (the repair-deletes-methods
    failure, which has cost this project more arms than any other single bug);
  - a patch that leaves the SAME function stubbed is rejected, so "resolved"
    cannot be claimed for moving a stub around;
  - a tier-2 claim with no evidence is rejected, because that is precisely the
    case `cargo check` cannot adjudicate — a callback with the right type and
    the wrong behaviour compiles clean;
  - a patch that does not compile is reverted;
  - shared types are never touched, except a `*_deps` module.

Following test_prompts.py and test_degradation.py: the paths are EXECUTED with
a fake LLM, including one that really raises.
"""

from __future__ import annotations

import asyncio
import glob
import sys

from state import Explanation
from rustgen.stub_repair import (SHARED_ID, Stub, StubContext, find_open_stubs,
                                 repair_stubs, validate_stub_patch)

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


# ---------------------------------------------------------------- fixtures

SHARED = """\
pub(crate) struct Task { pub id: i32 }
pub mod cli_deps {
    pub fn external_thing() -> i32 { todo!() }
}
impl Task {
    pub(crate) fn broken(&self) -> i32 { todo!("shared, must not be touched") }
}
"""

UNIT = """\
pub(crate) fn report(t: &Task) -> String {
    todo!("scheduler_print_report is not provided as a sibling signature")
}

pub(crate) fn keep_me(t: &Task) -> i32 { t.id }
"""

FIXED_UNIT = """\
pub(crate) fn report(t: &Task) -> String {
    scheduler_print_report(t)
}

pub(crate) fn keep_me(t: &Task) -> i32 { t.id }
"""

# the repair-deletes-methods failure, in this loop's clothing
LOSSY_UNIT = """\
pub(crate) fn report(t: &Task) -> String { scheduler_print_report(t) }
"""

# What the model is now ASKED for: the one function, not the section. Splicing
# preserves everything else byte-for-byte.
STILL_STUBBED_FN = """\
pub(crate) fn report(t: &Task) -> String {
    todo!("still stubbed")
}
"""

FIXED_FN = """\
pub(crate) fn report(t: &Task) -> String {
    scheduler_print_report(t)
}
"""

UNITS = [
    Explanation(id="u1", ranges=[[1, 20]],
                text="Prints a report of the scheduler's pending work.",
                invariants=[]),
]

CRATE = SHARED + "\n" + UNIT + "\npub fn scheduler_print_report(t: &Task) -> String { String::new() }\n"


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ask_json(self, prompt, max_tokens=None, schema=None, **kw):
        """The DECIDE call: JSON, short safe fields."""
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def ask(self, prompt, max_tokens=None, schema=None, **kw):
        """The PATCH call: plain text, fenced Rust. Source never travels in a
        JSON string — under a constrained grammar the model fails to escape
        quotes and newlines and the code comes back corrupted."""
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else "```rust\n" + str(r.get("code", "")) + "\n```"


# The full fixture has TWO patchable stubs — the deps-module one and the unit
# one — which is what discovery should see. The loop tests script a single
# exchange, so they use a shared block whose only stub is the untouchable one;
# otherwise the deps stub silently consumes the replies meant for the unit and
# every assertion fails for a reason that has nothing to do with what it tests.
SHARED_MIN = """\
pub(crate) struct Task { pub id: i32 }
impl Task {
    pub(crate) fn broken(&self) -> i32 { todo!("shared, must not be touched") }
}
"""


def patch(code, tier=1, evidence="", why="fix"):
    """A patch costs TWO calls now: decide, then payload.

    The split exists because a single flat schema let the model reply
    `{"action": "patch"}` with no code at all — 9 rejections in the first live
    run, every one "the patch is empty". DECIDE has no `code` property;
    The justification rides on the DECIDE reply; the code comes back as plain
    fenced Rust, because source inside a JSON string gets its quotes and
    newlines mangled under constrained decoding. Tests script both halves.
    """
    return [{"action": "patch", "tier": tier, "evidence": evidence, "why": why},
            "```rust\n" + code + "\n```"]


def sections():
    return {SHARED_ID: SHARED, "u1": UNIT}


def one_stub_sections():
    return {SHARED_ID: SHARED_MIN, "u1": UNIT}


def ctx(blind=False):
    return StubContext(CRATE, UNITS,
                       c_sources={"src/cli.c": "int scheduler_print_report(void){return 0;}"},
                       call_texts={"scheduler_print_report": ["scheduler_print_report(&s)"]},
                       specs={"u1": {"signatures": ["fn report(t:&Task)->String"]}},
                       blind=blind)


# ------------------------------------------------------------------- tests

def test_find_open_stubs_is_keyed_on_location() -> None:
    found = find_open_stubs(sections(), shared_id=SHARED_ID)
    by = {(s.section, s.function): s for s in found}
    check("finds the documented stub in a unit", ("u1", "report") in by)
    check("finds the BARE stub in a deps module",
          ("__shared__", "external_thing") in by)
    deps = by[("__shared__", "external_thing")]
    check("  it is marked as a deps module", deps.in_deps_module)
    check("  bare stubs carry no message", deps.message is None)
    check("  and a deps stub IS patchable despite living in shared types",
          deps.patchable)
    shared = by[("__shared__", "broken")]
    check("a shared-types stub is found", shared is not None)
    check("  but is NOT patchable", not shared.patchable)
    check("the message is recovered when present",
          by[("u1", "report")].message.startswith("scheduler_print_report"))
    check("a function with no stub is not reported",
          not any(s.function == "keep_me" for s in found))


def test_validate_rejects_the_deletion_failures() -> None:
    stub = Stub(section="u1", function="report")
    check("accepts a real fix", validate_stub_patch(UNIT, FIXED_UNIT, stub) == "")
    bad = validate_stub_patch(UNIT, LOSSY_UNIT, stub)
    check("rejects a patch that drops a sibling function", bad != "")
    check("  and names what went", "keep_me" in bad, bad)
    gone = validate_stub_patch(UNIT, "pub(crate) fn keep_me(t:&Task)->i32{t.id}", stub)
    check("rejects a patch that drops the function being fixed",
          "report" in gone, gone)
    check("rejects an unbalanced patch",
          "does not parse" in validate_stub_patch(UNIT, "fn report(){", stub))
    check("rejects an empty patch", validate_stub_patch(UNIT, "  ", stub) != "")


def test_splice_replaces_one_function_only() -> None:
    """The property that makes splicing better than section replacement: a
    repair CANNOT drop a sibling, because everything outside the target span is
    preserved byte-for-byte rather than regenerated and checked afterwards."""
    from rustgen.stub_repair import function_span, splice_function
    check("finds the function's span", function_span(UNIT, "report") is not None)
    check("returns None for a function that is not there",
          function_span(UNIT, "nope") is None)
    out, err = splice_function(UNIT, "report", FIXED_FN)
    check("a one-function replacement splices cleanly", err == "", err)
    check("  the stub is gone", "todo!" not in out)
    check("  the sibling survives untouched", out.count("fn keep_me") == 1, out)

    # the hole this closes: a model told to send one function often sends the
    # whole section, and splicing that duplicates every sibling — which still
    # contains every name it should, so a "nothing was lost" check passes it
    _, dup = splice_function(UNIT, "report", FIXED_UNIT)
    check("a whole-section reply is rejected, not silently duplicated",
          "duplicated" in dup, dup)

    _, trunc = splice_function(UNIT, "report",
                               "pub(crate) fn report(t:&Task)->String { let x=1;")
    check("a truncated replacement is rejected", "does not parse" in trunc, trunc)
    check("  and says it was probably cut off", "cut off" in trunc, trunc)
    _, wrong = splice_function(UNIT, "report", "fn other() -> i32 { 1 }")
    check("a replacement for the wrong function is rejected",
          "does not define" in wrong, wrong)


def test_context_answers() -> None:
    c = ctx()
    d = c.defines("scheduler_print_report")
    check("defines finds a function that exists", "IS defined" in d, d)
    check("defines is explicit when nothing provides it",
          "Nothing in this crate defines" in c.defines("nope_at_all"))
    call = c.callers_of("scheduler_print_report")
    check("callers_of surfaces the C call site", "C call site" in call, call)
    check("callers_of is explicit when nothing calls it",
          "Nothing calls" in c.callers_of("nope_at_all"))
    check("c_source returns the C body",
          "return 0" in c.c_source_of("scheduler_print_report"))
    check("mtu returns the unit block", "Prints a report" in c.mtu("u1"))
    check("siblings_of returns recorded signatures",
          "fn report" in c.siblings_of("u1"))
    check("unknown question is answered, not raised",
          "unknown question" in c.answer({"q": "bogus", "arg": "x"}))


def test_blind_context_is_the_control_arm() -> None:
    c = ctx(blind=True)
    for q in ("defines", "callers_of", "c_source", "mtu"):
        a = c.answer({"q": q, "arg": "scheduler_print_report"})
        check(f"blind handler withholds {q}", "no information available" in a, a)


def test_resolves_a_stub() -> None:
    llm = FakeLLM(patch(FIXED_FN, tier=2,
                        evidence="defines() says it is in the crate",
                        why="wired to the sibling"))
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID))
    check("the stub is resolved", rep.resolved == 1, rep.summary())
    check("  the section is replaced", "scheduler_print_report(t)" in out["u1"])
    check("  the sibling function survives", "keep_me" in out["u1"])
    check("  the tier is recorded",
          any(d.get("tier") == 2 for d in rep.details), str(rep.details))


def test_tier2_without_evidence_is_refused() -> None:
    """The check the compiler cannot make. A type-correct wrong callback
    compiles clean, so evidence is the only thing standing between a tier-2
    claim and a silent behaviour change."""
    llm = FakeLLM(patch(FIXED_FN, tier=2, evidence="", why="looks right")
                  + patch(FIXED_FN, tier=2,
                          evidence="callers_of shows the C site",
                          why="wired to the sibling"))
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID))
    check("an uncited tier-2 patch is rejected", rep.rejected == 1, rep.summary())
    check("  the reason is fed back",
          any("cited no evidence" in x for x in llm.prompts))
    check("  and the cited retry is accepted", rep.resolved == 1)


def test_patch_that_leaves_the_same_stub_is_refused() -> None:
    llm = FakeLLM(patch(STILL_STUBBED_FN, why="no change") * 6)
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID, rounds=2))
    check("a patch leaving the same function stubbed is rejected",
          rep.resolved == 0 and rep.rejected >= 1, rep.summary())
    check("  and the section is unchanged", out["u1"] == UNIT)


def test_failed_compile_reverts() -> None:
    llm = FakeLLM(patch(FIXED_FN) * 6)
    out, rep = asyncio.run(repair_stubs(
        llm, one_stub_sections(), UNITS, ctx(), shared_id=SHARED_ID, rounds=2,
        compile_check=lambda trial: "error[E0061]: wrong number of arguments"))
    check("a patch that does not compile is not kept", rep.resolved == 0,
          rep.summary())
    check("  the section is reverted", out["u1"] == UNIT)
    check("  and the compiler output is fed back",
          any("E0061" in x for x in llm.prompts))


def test_give_up_is_a_normal_outcome() -> None:
    llm = FakeLLM([{"action": "give_up",
                    "reason": "caller-supplied comparator, nothing calls it"}])
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID))
    check("give_up is recorded", rep.gave_up == 1, rep.summary())
    check("  nothing is changed", out["u1"] == UNIT)
    check("  and it is not counted as a rejection", rep.rejected == 0)


def test_ask_then_patch() -> None:
    llm = FakeLLM([{"action": "ask", "questions": [
                        {"q": "defines", "arg": "scheduler_print_report"}]}]
                  + patch(FIXED_FN, tier=2, evidence="defines said it exists"))
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID))
    check("ask then patch resolves", rep.resolved == 1, rep.summary())
    check("  the answer reached the next prompt",
          any("IS defined" in x for x in llm.prompts))
    check("  the question is recorded",
          any("defines" in q for d in rep.details for q in d["questions"]))


def test_shared_types_are_never_patched() -> None:
    llm = FakeLLM(patch("pub struct X{} // twenty chars plus") * 6)
    out, rep = asyncio.run(repair_stubs(llm, sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID, rounds=1))
    check("the shared-types stub is skipped", rep.skipped_shared == 1,
          rep.summary())
    check("  and the shared block is byte-identical", out[SHARED_ID] == SHARED)


def test_llm_that_raises_does_not_kill_the_run() -> None:
    llm = FakeLLM([RuntimeError("vLLM went away")])
    out, rep = asyncio.run(repair_stubs(llm, one_stub_sections(), UNITS, ctx(),
                                        shared_id=SHARED_ID))
    check("a raising LLM is caught", rep.errored == 1, rep.summary())
    check("  the section is untouched", out["u1"] == UNIT)
    check("  and the failure is recorded",
          any("vLLM went away" in str(d.get("outcome")) for d in rep.details))


def test_finds_the_stubs_in_the_recorded_crates() -> None:
    """Known-good/known-bad against real data: the loop must find stubs in the
    crates the gate voided, and find none in the crates that scored."""
    pat = ("/nobackup2/alleshwaram/mtu_runs/abl2/runs/*/_project_*/"
           "rust_crate/src/lib.rs")
    voided = clean = 0
    for f in glob.glob(pat):
        tag = f.split("/")[6]
        vf = f"/nobackup2/alleshwaram/mtu_runs/abl2/score/{tag}.verdict"
        try:
            verdict = open(vf).read()
        except OSError:
            continue
        src = open(f, errors="ignore").read()
        n = len(find_open_stubs({"crate": src}))
        if verdict.startswith("STUB_CRATE"):
            voided += n > 0
        elif verdict.startswith("bineq") and " N/A " not in verdict:
            clean += n == 0
            if n:
                check(f"scored crate {tag} unexpectedly has {n} stub(s)", False)
    check(f"stubs found in all {voided} crates the gate voided", voided > 0,
          f"voided={voided}")
    check(f"no stubs in the {clean} crates that scored", clean > 0,
          f"clean={clean}")


def main() -> int:
    test_find_open_stubs_is_keyed_on_location()
    test_validate_rejects_the_deletion_failures()
    test_splice_replaces_one_function_only()
    test_context_answers()
    test_blind_context_is_the_control_arm()
    test_resolves_a_stub()
    test_tier2_without_evidence_is_refused()
    test_patch_that_leaves_the_same_stub_is_refused()
    test_failed_compile_reverts()
    test_give_up_is_a_normal_outcome()
    test_ask_then_patch()
    test_shared_types_are_never_patched()
    test_llm_that_raises_does_not_kill_the_run()
    test_finds_the_stubs_in_the_recorded_crates()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
