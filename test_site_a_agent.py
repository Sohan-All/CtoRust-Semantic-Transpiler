"""Site A's agent path, replayed against the REAL voided draft.

Run: PYTHONPATH=. venv/bin/python test_site_a_agent.py

`double_linked_list_escon_srvA_t1` died with `1 unexplained stub body/bodies:
todo!() at line 35` on a block whose only stub was inside `pub mod deps` — legal
in substance, refused because `_DEPS_MOD` wants a literal `*_deps`. The escalated
model was handed a correct block and asked to fix a defect that was not there,
and gave up, which was the honest answer available to it.

These assertions use that exact recorded draft. The point is not that a fake LLM
can rename a module — of course it can — but that the PATH works: the agent's
proposal reaches `validate_repair` and the same gates the local repair answers
to, a success routes to `notes` and never to `failures`, and a failure leaves the
stage exactly where it was.
"""
import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from rustgen.agent import AgentBudget, StagedRoot
from rustgen.agent_sites import agent_repair_types, types_task
from rustgen.common import illegal_stubs
from testpaths import RUNS

PASS = 0
RUN = RUNS / ("double_linked_list_escon_srvA_t1/_project_B03_organic/"
              "project.jsonl")


def ok(cond, label):
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1


def recorded_draft():
    for line in RUN.open(errors="ignore"):
        rec = json.loads(line)
        if rec.get("type") == "types_drafts":
            return rec["drafts"]
    raise SystemExit("no types_drafts record — has the corpus moved?")


# Every assertion in this file replays ONE recorded void, so without the
# recording there is nothing here to test. Skip the module rather than
# raising at import, which otherwise reads as a code defect.
if not RUN.exists():
    print(f"  SKIP  the recorded void is not on disk ({RUN})")
    print("ALL PASS (skipped: no recorded run)")
    raise SystemExit(0)

DRAFTS = recorded_draft()
BLOCK = DRAFTS[-1]["block"]
PROBLEM = DRAFTS[-1]["problem"]


class ScriptedAgent:
    """Keyed on call order, per the convention. Turn 1 reads the checker,
    turn 2 proposes the renamed module."""

    def __init__(self, fixed_block):
        self.fixed = fixed_block
        self.n = 0
        self.prompts = []

    async def ask(self, prompt, max_tokens=None, **kw):
        self.n += 1
        self.prompts.append(prompt)
        if self.n == 1:
            return ('{"action": "read", "calls": [{"tool": "grep", '
                    '"pattern": "_DEPS_MOD", "path": "pipeline"}]}')
        return ('{"action": "propose", "why": "the gate wants a *_deps suffix"}\n'
                "```rust section=__shared__\n" + self.fixed + "\n```")


class GivingUpAgent:
    async def ask(self, prompt, max_tokens=None, **kw):
        return '{"action": "give_up", "why": "cannot tell what is wrong"}'


def _staged():
    """Real staged root: the pipeline source is what makes site A reachable."""
    tmp = Path(tempfile.mkdtemp())
    from rustgen.agent import stage
    return tmp, stage(tmp / "root",
                      pipeline_src=Path(__file__).resolve().parent)


# ------------------------------------------------------------------ the case

def test_the_recorded_draft_is_a_gate_false_positive():
    """Known-bad and known-good on the DEFECT itself, before testing any fix."""
    ok(illegal_stubs(BLOCK) != "", "the recorded draft is refused as-is")
    renamed = BLOCK.replace("pub mod deps", "pub mod editor_deps")
    ok(illegal_stubs(renamed) == "",
       "the byte-identical block is accepted once the module is renamed")
    ok("todo!()" in renamed,
       "and it still contains the stub — nothing was removed to pass")


def test_agent_proposal_reaches_validate_repair_and_is_accepted():
    tmp, sr = _staged()
    try:
        fixed = BLOCK.replace("pub mod deps", "pub mod editor_deps")
        llm = ScriptedAgent(fixed)
        block, rep = asyncio.run(agent_repair_types(
            llm, BLOCK, PROBLEM, staged=sr, drafts=DRAFTS,
            budget=AgentBudget(max_turns=4),
            checker=lambda *a, **k: []))      # cargo stubbed: not under test here
        ok(block is not None, "a valid proposal is returned")
        ok(rep["repaired"] is True, "and reported as a repair")
        ok(rep["section_id"] == "__shared__", "the section id is recorded")
        ok(illegal_stubs(block) == "", "the returned block passes the gate")
        ok(rep["rounds"] == 2 and rep["agent"]["tool_calls"] == 1,
           "turns and tool calls are carried into the rep")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_agent_can_actually_find_the_defect_in_the_staged_pipeline():
    """The load-bearing containment claim for site A: the checker's source is
    readable, so the complaint is diagnosable. Asserted against the REAL
    pipeline tree, not a fixture."""
    tmp, sr = _staged()
    try:
        from rustgen.agent import tool_grep
        hit = tool_grep(sr, r"_DEPS_MOD\s*=", "")
        ok("_deps" in hit and "common.py" in hit,
           "grep over the staged pipeline returns the regex that refused the block")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_rejected_proposal_leaves_the_stage_where_it_was():
    tmp, sr = _staged()
    try:
        # Proposes a block that still fails validate_repair — a type deleted.
        gutted = "pub mod editor_deps {\n    use super::*;\n}\n"
        block, rep = asyncio.run(agent_repair_types(
            ScriptedAgent(gutted), BLOCK, PROBLEM, staged=sr, drafts=DRAFTS,
            budget=AgentBudget(max_turns=4), checker=lambda *a, **k: []))
        ok(block is None, "a proposal that loses type definitions is refused")
        ok(rep["repaired"] is False, "and not reported as a repair")
        ok("validate_repair" in rep["why"], "the refusing check is named in the rep")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multiple_sections_are_refused():
    tmp, sr = _staged()
    try:
        class Multi:
            async def ask(self, prompt, max_tokens=None, **kw):
                return ('{"action": "propose", "why": "x"}\n'
                        "```rust section=__shared__\npub struct A;\n```\n"
                        "```rust section=other\npub struct B;\n```")
        block, rep = asyncio.run(agent_repair_types(
            Multi(), BLOCK, PROBLEM, staged=sr, budget=AgentBudget(max_turns=2),
            checker=lambda *a, **k: []))
        ok(block is None, "the shared types are ONE block; several fences is refused")
        ok("single block" in rep["why"], "and the reason says so")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_give_up_is_survived():
    tmp, sr = _staged()
    try:
        block, rep = asyncio.run(agent_repair_types(
            GivingUpAgent(), BLOCK, PROBLEM, staged=sr,
            budget=AgentBudget(max_turns=2), checker=lambda *a, **k: []))
        ok(block is None and rep["repaired"] is False, "give_up returns no block")
        ok(rep["action"] == "give_up", "and is recorded as a give_up, not an error")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_task_text_does_not_name_the_defect():
    """If the PROMPT says 'check your deps module name', the prompt solved it and
    the measurement is of nothing."""
    text = types_task(PROBLEM, DRAFTS)
    low = text.lower()
    ok("_deps" not in low and "suffix" not in low,
       "the task text does not name the defect or its fix")
    ok("pipeline/" in text, "but it does say where the checkers can be read")
    ok(PROBLEM in text, "and it carries the gate's actual complaint")
    ok(BLOCK[:60] in text, "and the rejected draft itself")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"ALL PASS ({PASS} assertions)")
