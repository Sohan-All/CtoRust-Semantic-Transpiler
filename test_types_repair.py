"""Stage T repair: fix a rejected types block instead of dying on it.

    PYTHONPATH=. venv/bin/python test_types_repair.py

Why this exists. `synthesize_types` detects a bad shared-types block and
retries twice, then gives up. Measured over 255 recorded run logs, every run
reaching that branch died — 4 BUILD_FAILED, 3 STUB_CRATE, none scored. The
repair loop is the response, and it is only safe if two things hold:

  1. it cannot make a good block worse — on ANY failure the original block is
     returned unchanged, so the caller behaves exactly as it did before;
  2. it cannot smuggle a broken block past the gates that rejected it.

Half 2 is the one worth testing hardest. A repair that satisfies the model but
drops a struct every unit is written against is not a repair, it is a different
failure — and it would be a quiet one, because the block would still compile.
`lost_type_definitions` exists for that and is asserted against BOTH directions
here: dropping a method is the correct fix and must be allowed, dropping a type
must not be.

Following test_prompts.py and test_degradation.py: the paths are EXECUTED with
a fake LLM, including one that really raises. Testing the checker is not
testing the response to the checker.
"""

from __future__ import annotations

import asyncio
import glob
import re
import sys

from state import Explanation
from testpaths import RUNS, skip_unless
from rustgen.common import (illegal_stubs, illegal_type_bodies,
                            lost_type_definitions)
from rustgen.types_repair import (TypesContext, compile_regression,
                                  repair_types_block, validate_repair)

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


# ---------------------------------------------------------------- fixtures

# The real shape that deadlocked `binary_heap base_srvB_t2`: an impl block of
# stubbed methods, which every unit then defers to.
BAD_BLOCK = """\
pub(crate) struct SchedTask { pub priority: i32 }
pub(crate) enum SchedulerError { Full }
pub(crate) struct Scheduler { pub tasks: Vec<SchedTask> }

impl Scheduler {
    pub(crate) fn spawn(&mut self, t: SchedTask) -> Result<(), SchedulerError> {
        todo!()
    }
    pub(crate) fn dispatch(&mut self) -> Result<SchedTask, SchedulerError> {
        todo!()
    }
}
"""

# The correct repair: the stubbed impl block is gone, every TYPE survives.
GOOD_REPAIR = """\
pub(crate) struct SchedTask { pub priority: i32 }
pub(crate) enum SchedulerError { Full }
pub(crate) struct Scheduler { pub tasks: Vec<SchedTask> }
"""

# The dangerous repair: stubs gone, but a type went with them.
LOSSY_REPAIR = """\
pub(crate) struct SchedTask { pub priority: i32 }
pub(crate) struct Scheduler { pub tasks: Vec<SchedTask> }
"""

UNITS = [
    Explanation(id="exp_0001", ranges=[[1, 40]],
                text="Spawns a task into the scheduler and dispatches it.",
                invariants=["A spawned task is present in the queue"]),
    Explanation(id="exp_0002", ranges=[[41, 60]],
                text="Formats a report of pending work.", invariants=[]),
]

C_SRC = """\
int scheduler_spawn(Scheduler *s, SchedTask t) {
    s->tasks[s->len++] = t;
    return 0;
}
"""


class FakeLLM:
    """Returns a scripted sequence of replies. `ask_json` is what the loop
    uses; anything left over is an error the test should notice."""

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


def tpatch(code, why="fix"):
    """A patch costs TWO calls: decide, then payload. See stub_repair's schema
    note — a flat schema requiring only `action` let the model claim a patch
    with no code at all."""
    return [{"action": "patch", "why": why}, "```rust\n" + code + "\n```"]


def no_cargo(code, context_rs=""):
    """Stand-in for cargo_error_signatures: never reports an error. Keeps the
    unit tests off the real toolchain; the cargo path has its own test."""
    return set()


# ------------------------------------------------------------------- tests

def test_lost_type_definitions_both_directions() -> None:
    """The check that makes a repair safe. Both directions, per the convention
    that a relaxation is tested against every fixture at once."""
    check("dropping a stubbed METHOD is allowed (that is the fix)",
          lost_type_definitions(BAD_BLOCK, GOOD_REPAIR) == "")
    lossy = lost_type_definitions(BAD_BLOCK, LOSSY_REPAIR)
    check("dropping a TYPE is rejected", lossy != "")
    check("  and it names the type that went", "SchedulerError" in lossy, lossy)
    check("a type named only in a comment does not count as defined",
          lost_type_definitions("enum E { A }", "// enum E\n") != "")
    check("unchanged block is clean",
          lost_type_definitions(BAD_BLOCK, BAD_BLOCK) == "")


def test_validate_repair_rejects_each_way() -> None:
    check("accepts the correct repair",
          validate_repair(BAD_BLOCK, GOOD_REPAIR, checker=no_cargo) == "")
    check("rejects a repair that drops a type",
          "type definition" in validate_repair(BAD_BLOCK, LOSSY_REPAIR,
                                               checker=no_cargo))
    check("rejects a repair that still carries the stubs",
          "original defect" in validate_repair(BAD_BLOCK, BAD_BLOCK,
                                               checker=no_cargo))
    check("rejects an unbalanced repair",
          "does not parse" in validate_repair(BAD_BLOCK, "struct A { ",
                                              checker=no_cargo))
    check("rejects an empty repair",
          validate_repair(BAD_BLOCK, "   ", checker=no_cargo) != "")


def test_compile_regression_is_asymmetric() -> None:
    """A per-file block is written against project-level types, so compiling it
    alone reports errors that were never its fault. Only NEW ones count."""
    pre = {"[E0412]: cannot find type `Shared` in this scope"}
    check("a pre-existing error is not charged to the repair",
          compile_regression("old", "new", checker=lambda c, x="": pre) == "")
    seq = iter([pre | {"[E0308]: mismatched types"}, pre])
    check("a NEW error is charged to the repair",
          "new compile error" in
          compile_regression("old", "new", checker=lambda c, x="": next(seq)))
    check("cargo unavailable is not read as a pass",
          compile_regression("old", "new", checker=lambda c, x="": None) == "")


def test_context_answers_questions() -> None:
    ctx = TypesContext(UNITS, C_SRC)
    m = ctx.mentions("dispatch")
    check("mentions finds the unit that owns the behaviour", "exp_0001" in m, m)
    check("  and says the stub collides with it", "collide" in m, m)
    none = ctx.mentions("Nonexistent")
    check("mentions reports when NOTHING owns it", "No unit" in none, none)
    check("  and says it should not be declared",
          "should not be declared" in none, none)
    src = ctx.c_source_of("scheduler_spawn")
    check("c_source returns the function body", "s->tasks" in src, src[:60])
    check("c_source on an unknown name says so",
          "no definition" in ctx.c_source_of("nope"))
    check("mtu returns the unit block", "Spawns a task" in ctx.mtu("exp_0001"))
    check("unknown question kind is answered, not raised",
          "unknown question" in ctx.answer({"q": "bogus", "arg": "x"}))


def test_loop_patches() -> None:
    llm = FakeLLM(tpatch(GOOD_REPAIR, "removed stubs"))
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS, checker=no_cargo))
    check("a valid patch is accepted", rep["repaired"] is True, str(rep))
    check("  and the block is replaced", out.strip() == GOOD_REPAIR.strip())
    check("  and the result passes the gates that rejected the original",
          illegal_stubs(out) == "" and illegal_type_bodies(out) == "")


def test_loop_asks_then_patches() -> None:
    llm = FakeLLM([{"action": "ask",
                    "questions": [{"q": "mentions", "arg": "dispatch"}]}]
                  + tpatch(GOOD_REPAIR, "unit exp_0001 owns it"))
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS, checker=no_cargo))
    check("ask then patch succeeds", rep["repaired"] is True, str(rep))
    check("  the question is recorded", rep["questions"] == ["mentions(dispatch)"],
          str(rep["questions"]))
    check("  the answer reached the second prompt",
          any("exp_0001" in x for x in llm.prompts))


def test_loop_rejects_lossy_patch_and_retries() -> None:
    llm = FakeLLM(tpatch(LOSSY_REPAIR, "tidied")
                  + tpatch(GOOD_REPAIR, "kept the error enum"))
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS, checker=no_cargo))
    check("a lossy patch is rejected", len(rep["rejected"]) == 1, str(rep))
    check("  the rejection is fed back",
          any("REJECTED" in x for x in llm.prompts))
    check("  and the next good patch is accepted", rep["repaired"] is True)
    check("  the block that survives is the good one",
          "SchedulerError" in out)


def test_give_up_returns_original() -> None:
    llm = FakeLLM([{"action": "give_up", "reason": "cannot tell who owns it"}])
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS, checker=no_cargo))
    check("give_up returns the ORIGINAL block", out == BAD_BLOCK)
    check("  and is recorded as give_up", rep["action"] == "give_up")
    check("  and is not marked repaired", rep["repaired"] is False)


def test_llm_that_raises_does_not_kill_the_run() -> None:
    """The half that matters: a repair that explodes must degrade to the old
    behaviour, not take the stage with it."""
    llm = FakeLLM([RuntimeError("vLLM went away")])
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS, checker=no_cargo))
    check("a raising LLM is caught", rep["action"] == "error", str(rep))
    check("  the original block comes back", out == BAD_BLOCK)
    check("  and the error is recorded", "vLLM went away" in rep["why"])


def test_rounds_cap_binds() -> None:
    llm = FakeLLM(tpatch(LOSSY_REPAIR) * 9)
    out, rep = asyncio.run(repair_types_block(
        llm, BAD_BLOCK, illegal_stubs(BAD_BLOCK), UNITS,
        rounds=3, checker=no_cargo))
    check("the round cap binds", rep["rounds"] == 3, str(rep["rounds"]))
    check("  exhaustion returns the original", out == BAD_BLOCK)
    check("  and is recorded as exhausted", rep["action"] == "exhausted")


def test_entry_rate_on_recorded_stage_t_output() -> None:
    """Known-good half, against what stage T actually PRODUCED.

    The fixture has to be the `rust_types` records, not the assembled crate.
    A crate's leading region is `types_all` AFTER the compile loop has had a
    go at it, and a shared-block repair writing a real method body is exactly
    why `illegal_type_bodies` also gates `set_section` — so slicing the crate
    measures the compile loop, not stage T, and reports firings on runs that
    scored perfectly well.

    The loop is entered only on the gates' verdict, so what matters is that
    the gates are quiet on the overwhelming majority: a repair that ran on
    every block would be a new failure mode, not a fix.
    """
    import json
    # Tag is derived RELATIVE to the runs root — see the note in
    # test_stub_repair.py. A hardcoded index breaks silently when this tree
    # moves, and reads as a corpus problem rather than a path problem.
    if skip_unless(RUNS, "recorded runs"):
        return
    runs = str(RUNS)
    pat = f"{runs}/*/_project_*/files/*/state.jsonl"
    clean = fires = 0
    fired_runs = set()
    for sf in glob.glob(pat):
        tag = sf[len(runs) + 1:].split("/")[0]
        for line in open(sf, errors="ignore"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") != "rust_types":
                continue
            t = r.get("types_rs", "")
            if illegal_stubs(t) or illegal_type_bodies(t):
                fires += 1
                fired_runs.add(tag)
            else:
                clean += 1
    total = clean + fires
    check("recorded stage-T outputs were found", total > 100, f"total={total}")
    check(f"the loop stays out of the way on {clean}/{total} blocks",
          total and fires / total < 0.10, f"{fires}/{total} would enter")
    check("it does fire on the runs known to have died at stage T",
          {"binary_heap_base_srvB_t2", "binary_heap_base_srvB_t1"}
          <= fired_runs, str(sorted(fired_runs))[:200])


def test_synthesize_types_still_records_exhaustion() -> None:
    """The integration point, and the safety property that matters most.

    Testing the checker is not testing the response to the checker. With the
    repair wired in, a run whose repair fails for ANY reason must behave
    exactly as it did before this module existed: record the exhaustion so the
    harness refuses to score, and hand back the last draw. Here the repair
    fails in the crudest possible way — the LLM has no `ask_json` at all — and
    the old contract still has to hold.
    """
    from rustgen.types_stage import synthesize_types

    stubbed = ("```rust\npub struct S { pub a: i32 }\n"
               "impl S { pub fn f(&self) -> i32 { todo!() } }\n```\n"
               'GLOSSARY:\n```json\n{"thing": "S"}\n```')

    class OnlyAsk:
        async def ask(self, prompt, **kw):
            return stubbed

    unit = Explanation(id="exp_0001", ranges=[[1, 5]], text="does a thing",
                       invariants=[], status="locked")
    failures: list[dict] = []
    rs, _ = asyncio.run(synthesize_types(OnlyAsk(), [unit], 100,
                                         failures=failures, repair=True))
    check("a broken repair still records exhaustion",
          len(failures) == 1 and "stub gate exhausted" in failures[0]["error"],
          str(failures))
    check("  and the last draw is still returned", "struct S" in rs)

    failures2: list[dict] = []
    rs2, _ = asyncio.run(synthesize_types(OnlyAsk(), [unit], 100,
                                          failures=failures2, repair=False))
    check("repair=False is unchanged from the old behaviour",
          len(failures2) == 1 and rs2 == rs, str(failures2))


def test_successful_repair_does_not_degrade_the_run() -> None:
    """The mirror of the test above, and the one that was missing.

    Exhaustion was tested; SUCCESS was not, so the success record went into
    `failures` — the caller's degraded list — for months. A rescued shared
    block then marked its own run INCOMPLETE and unscoreable, and because the
    record carried no `error` key the degraded printer died on KeyError first.
    Both live firings (binary_heap srvB t12, cc_array srvA t3, 2026-08-03) were
    repairs that worked on crates that compiled clean, and both scored nothing.

    A guard on the failure path says nothing about the success path.
    """
    from rustgen.types_stage import synthesize_types

    stubbed = ("```rust\npub struct S { pub a: i32 }\n"
               "impl S { pub fn f(&self) -> i32 { todo!() } }\n```\n"
               'GLOSSARY:\n```json\n{"thing": "S"}\n```')
    good = "pub struct S { pub a: i32 }\n"

    class RepairsCleanly:
        """Keyed on call ORDER, not prompt text. Stage T's retry prompt quotes
        the stub problem back, so matching on "todo!()"/"stub" made stage T fix
        itself on retry and the repair never ran — the test passed its first
        assertion while measuring nothing."""

        def __init__(self) -> None:
            self.asks = 0

        async def ask(self, prompt, **kw):
            self.asks += 1
            # stage T gets TYPES_RETRIES+1 draws, all stubbed, so it exhausts;
            # anything after that is the repair's PATCH call
            from rustgen.types_stage import TYPES_RETRIES
            if self.asks <= TYPES_RETRIES + 1:
                return stubbed
            return f"```rust\n{good}```"

        async def ask_json(self, prompt, **kw):
            return {"action": "patch", "why": "dropped the phantom method"}

    unit = Explanation(id="exp_0001", ranges=[[1, 5]], text="does a thing",
                       invariants=[], status="locked")
    failures: list[dict] = []
    notes: list[dict] = []
    rs, _ = asyncio.run(synthesize_types(RepairsCleanly(), [unit], 100,
                                         failures=failures, notes=notes,
                                         repair=True))
    repaired = "todo!()" not in rs
    check("the repair resolved the block", repaired, rs)
    if repaired:
        check("  a SUCCESS records NOTHING in failures (would void the run)",
              failures == [], str(failures))
        check("  and IS recorded in notes", len(notes) == 1
              and notes[0].get("repaired") is True, str(notes))

    # the degraded printer must survive a malformed record either way
    bad = {"stage": "types", "unit": "(shared types)", "repaired": True}
    rendered = (f"{bad['stage']} {bad.get('unit') or bad.get('file')}: "
                f"{bad.get('error') or f'(no error recorded: {bad})'}")
    check("  the degraded printer does not KeyError on a record without "
          "'error'", "no error recorded" in rendered, rendered)


def test_cargo_path_actually_runs() -> None:
    """The one test that touches the real toolchain. `todo!()` type-checks, so
    cargo is NOT the stub gate — this asserts it catches what it is for
    (structural errors) and stays quiet on what it is not."""
    from rustgen.types_repair import cargo_error_signatures
    sigs = cargo_error_signatures("pub struct A { pub x: i32 }")
    if sigs is None:
        check("cargo unavailable — path skipped", True)
        return
    check("a valid block reports no errors", sigs == set(), str(sigs))
    bad = cargo_error_signatures("pub struct A { pub x: NoSuchType }")
    check("an undefined type IS reported", bool(bad), str(bad))
    check("  line numbers are normalised out",
          all(not re.search(r":\d+:", s) for s in bad), str(bad))


def main() -> int:
    test_lost_type_definitions_both_directions()
    test_validate_repair_rejects_each_way()
    test_compile_regression_is_asymmetric()
    test_context_answers_questions()
    test_loop_patches()
    test_loop_asks_then_patches()
    test_loop_rejects_lossy_patch_and_retries()
    test_give_up_returns_original()
    test_llm_that_raises_does_not_kill_the_run()
    test_rounds_cap_binds()
    test_entry_rate_on_recorded_stage_t_output()
    test_synthesize_types_still_records_exhaustion()
    test_successful_repair_does_not_degrade_the_run()
    test_cargo_path_actually_runs()
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
