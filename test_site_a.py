"""plan.md site A: escalation of stage T, and the stage-T draft persistence.

Run: cd diffusionMTUs && PYTHONPATH=. venv/bin/python test_site_a.py

The properties that matter, in order of what would hurt most if wrong:

  1. escalation is charged ONLY after the local repair fails (the cost premise)
  2. a failed escalation leaves the run exactly where it was before — recorded
     exhaustion, last draw returned (the safety property the local repair has)
  3. every firing is recorded, accepted or not, including one that RAISES
  4. rejected drafts are persisted, so a gate-voided run can be audited

Fakes are keyed on call ORDER, not prompt text: stage T's retry prompt quotes
the problem back, so a text-keyed fake lets stage T fix itself and the test
measures nothing.
"""
import asyncio
import dataclasses
import os

os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"
os.environ["VLLM_API_KEY"] = "test"

from config import Config
from rustgen.types_stage import synthesize_types, TYPES_RETRIES
from state import Explanation

PASS = FAIL = 0
STUBBED = ("```rust\npub struct S { pub a: i32 }\n"
           "impl S { pub fn f(&self) -> i32 { todo!() } }\n```\n"
           'GLOSSARY:\n```json\n{"thing": "S"}\n```')
GOOD = "pub struct S { pub a: i32 }\n"


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


def unit():
    return Explanation(id="exp_0001", ranges=[[1, 5]], text="does a thing",
                       invariants=[], status="locked")


class Model:
    """Stage T draws STUBBED until exhausted, then behaves per `repairs`.

    `repairs`: "clean" patches successfully, "giveup" declines, "boom" raises.
    `cfg` exists because Escalation reads `llm.cfg.worker_model` for the record.
    """

    def __init__(self, repairs="clean", name="fake-model"):
        self.repairs = repairs
        self.asks = 0
        self.json_calls = 0
        self.cfg = dataclasses.replace(Config(), worker_model=name)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def ask(self, prompt, **kw):
        self.asks += 1
        self.calls += 1
        self.output_tokens += 100
        if self.asks <= TYPES_RETRIES + 1:
            return STUBBED                     # stage T's own draws
        return f"```rust\n{GOOD}```"           # the repair's PATCH call

    async def ask_json(self, prompt, **kw):
        self.json_calls += 1
        self.calls += 1
        self.output_tokens += 50
        if self.repairs == "boom":
            raise RuntimeError("vertex went away")
        if self.repairs == "giveup":
            return {"action": "give_up", "reason": "cannot tell"}
        return {"action": "patch", "why": "dropped the phantom method"}


class Escalator(Model):
    """Escalation model. Stage T never draws from this one, so its first `ask`
    is already the repair's PATCH call."""

    async def ask(self, prompt, **kw):
        self.asks += 1
        self.calls += 1
        self.output_tokens += 400
        if self.repairs == "giveup":
            return "```rust\n" + STUBBED.split("```rust\n")[1]
        return f"```rust\n{GOOD}```"


def run(local, esc=None, **kw):
    failures, notes, escs, drafts = [], [], [], []
    rs, _ = asyncio.run(synthesize_types(
        local, [unit()], 100, failures=failures, notes=notes,
        escalations=escs, drafts=drafts, repair=True,
        escalation_llm=esc, **kw))
    return rs, failures, notes, escs, drafts


def main():
    # ---- 1. local repair succeeds -> escalation is NEVER touched
    local, esc = Model("clean"), Escalator("clean")
    rs, failures, notes, escs, drafts = run(local, esc)
    check("struct S" in rs and "todo!()" not in rs, "local repair applied")
    check(esc.calls == 0, "escalation NOT called when local repair succeeds")
    check(escs == [], "no escalation record when it never fired")
    check(len(notes) == 1 and notes[0]["by"] == "local",
          "the note says which model repaired it")
    check(failures == [], "a successful repair does not degrade the run")

    # ---- 2. local fails, escalation succeeds
    local, esc = Model("giveup"), Escalator("clean")
    rs, failures, notes, escs, drafts = run(local, esc)
    check("todo!()" not in rs, "escalated repair applied")
    check(esc.calls > 0, "escalation was called after the local repair failed")
    check(len(escs) == 1, "exactly one escalation record")
    check(escs[0]["accepted"] is True, "the firing is recorded as accepted")
    check(escs[0]["site"] == "types" and escs[0]["trigger"] == "gate_exhausted",
          "site and trigger recorded")
    check(escs[0]["output_tokens"] > 0, "the firing's cost is recorded")
    check(escs[0]["model"] == "fake-model", "the escalation model is recorded")
    check(len(notes) == 1 and notes[0]["by"] == "escalated",
          "the note attributes the rescue to escalation")
    check(failures == [], "an escalated rescue does not degrade the run")

    # ---- 3. both fail -> exactly the pre-escalation behaviour, plus a record
    local, esc = Model("giveup"), Escalator("giveup")
    rs, failures, notes, escs, drafts = run(local, esc)
    check(len(failures) == 1 and "stub gate exhausted" in failures[0]["error"],
          "a failed escalation still records exhaustion")
    check("struct S" in rs, "  and the last draw is still returned")
    check(len(escs) == 1 and escs[0]["accepted"] is False,
          "the failed firing is recorded as not accepted")
    check(notes == [], "no rescue note when nothing was rescued")

    # ---- 4. escalation RAISES: recorded, survivable, not a crashed run
    local, esc = Model("giveup"), Escalator("boom")
    rs, failures, notes, escs, drafts = run(local, esc)
    check(len(escs) == 1, "a raising firing is still recorded")
    check(escs[0]["accepted"] is False, "  and is not accepted")
    check(len(failures) == 1, "  and the run degrades as it would have anyway")
    check("struct S" in rs, "  and the last draw is still returned")

    # ---- 4b. a failure that ESCAPES repair_types_block.
    # `repair_types_block` catches its own transport errors, so case 4 above
    # never reaches types_stage's `except`. This one does: the escalation
    # object's counters raise, which blows up inside the `with` before any
    # repair happens. That branch is the only thing standing between a broken
    # escalation and a crashed run, and nothing else exercises it.
    class Exploding(Escalator):
        @property
        def calls(self):
            raise RuntimeError("counter is broken")

        @calls.setter
        def calls(self, v):
            pass

    local = Model("giveup")
    rs, failures, notes, escs, drafts = run(local, Exploding("clean"))
    check(len(escs) == 1, "an escaping failure is still recorded")
    check(escs[0]["accepted"] is False, "  and is not accepted")
    check(escs[0]["calls"] == 0 and escs[0]["output_tokens"] == 0,
          "  with zero cost rather than a traceback out of the metering")
    check(0 <= escs[0]["seconds"] < 60,
          "  and a sane wall time, not the machine's uptime")
    check(len(failures) == 1 and "struct S" in rs,
          "  and the run degrades exactly as it would have anyway")

    # ---- 5. drafts are persisted for auditing
    local = Model("giveup")
    rs, failures, notes, escs, drafts = run(local)
    check(len(drafts) == TYPES_RETRIES + 1,
          f"every rejected draft persisted (got {len(drafts)})")
    check(all("block" in d and "problem" in d and "attempt" in d
              for d in drafts), "each draft carries block, problem, attempt")
    check("todo!()" in drafts[0]["block"],
          "the draft holds the ACTUAL rejected text, not a summary")
    check([d["attempt"] for d in drafts] == list(range(TYPES_RETRIES + 1)),
          "drafts are numbered in order")

    # ---- 6. escalation off (the default) is byte-identical to before
    local_a, local_b = Model("giveup"), Model("giveup")
    rs_a, fa, na, ea, da = run(local_a, None)
    failures_b: list[dict] = []
    rs_b, _ = asyncio.run(synthesize_types(local_b, [unit()], 100,
                                           failures=failures_b, repair=True))
    check(rs_a == rs_b, "escalation_llm=None matches the old code path")
    check(ea == [], "no escalation records when disabled")
    check(len(fa) == len(failures_b) == 1, "both still record exhaustion")

    # ---- 7. a clean block never reaches any of this
    class Clean(Model):
        async def ask(self, prompt, **kw):
            self.asks += 1
            return f"```rust\n{GOOD}```\nGLOSSARY:\n```json\n{{}}\n```"

    local, esc = Clean(), Escalator("clean")
    rs, failures, notes, escs, drafts = run(local, esc)
    check(escs == [] and drafts == [] and failures == [] and notes == [],
          "a clean stage T produces no drafts, records or firings")
    check(esc.calls == 0, "  and never touches the escalation model")

    print(f"\n{PASS} passed, {FAIL} failed")
    print("ALL PASS" if FAIL == 0 else "FAILURES")
    return 1 if FAIL else 0


raise SystemExit(main())
