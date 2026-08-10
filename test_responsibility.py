"""The responsibility pass must be scoped to the PROJECT, not to one file.

    PYTHONPATH=. venv/bin/python test_responsibility.py

Why this exists. The pass assigns each single-owner concern to exactly one
unit, so no two units emit the same item. It ran inside run_project.py's
per-file loop, which meant it only ever saw one file's units — so a concern two
FILES both implement was invisible to it. Each file's pass was individually
correct and the crate still ended up with two definitions (E0592/E0428).

All four duplicates that BUILD_FAILED both `array_list` t3 arms crossed a file
boundary this way:

    find_task     project__exp_0001   vs  demo__exp_0007
    find_project  workspace__exp_0003 vs  demo__exp_0007
    trim_notes    task__exp_0001      vs  demo__exp_0007
    is_complete   (shared types)      vs  task__exp_0002

Following test_prompts.py and test_degradation.py, the paths are EXECUTED with
a fake LLM rather than inspected. Testing that the pass exists is not testing
that its result reaches the unit that has to obey it, which is the half that
was broken: owners are named `<stem>__<unit>` while the units inside a file
still carry `exp_0001`, so a comparison that skipped the qualification would
silently assign every concern to nobody.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys

from config import Config
from state import Explanation

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


def units(n: int, prefix: str = "") -> list[Explanation]:
    return [Explanation(id=f"{prefix}exp_{i:04d}", text=f"unit {i}",
                        invariants=[], ranges=[(i * 10 + 1, i * 10 + 5)],
                        status="locked")
            for i in range(1, n + 1)]


class _CountingLLM:
    """Counts responsibility calls; returns a fixed assignment for them."""

    def __init__(self, concerns: list[dict] | None = None):
        self.responsibility_calls = 0
        self.seen_unit_ids: list[str] = []
        self.concerns = concerns or []

    @staticmethod
    def _is_responsibility(prompt: str) -> bool:
        return "single-owner concern" in prompt or "EXACTLY ONE unit" in prompt

    async def ask_json(self, prompt: str, **kw):
        if self._is_responsibility(prompt):
            self.responsibility_calls += 1
            for line in prompt.splitlines():
                if line.startswith("[") and "] (C lines" in line:
                    self.seen_unit_ids.append(line[1:line.index("]")])
            return {"concerns": self.concerns}
        return {"signatures": ["pub(crate) fn f() -> i32"],
                "behavior_note": "n"}

    async def ask(self, prompt: str, **kw):
        return "```rust\npub(crate) fn f() -> i32 { 1 }\n```"


RICH = dataclasses.replace(Config(), rustgen_spec_mode="rich")


def test_supplied_concerns_suppress_the_per_file_call() -> None:
    print("\n=== a project-scoped assignment is not re-run per file ===")
    from rustgen.spec_stage import generate_specs

    llm = _CountingLLM()
    out = asyncio.run(generate_specs(llm, units(3), "", {}, RICH,
                                     concerns=[], id_prefix="demo__"))
    check("every unit still gets a spec",
          set(out) == {"exp_0001", "exp_0002", "exp_0003"}, str(set(out)))
    check("no responsibility call is made when concerns are supplied",
          llm.responsibility_calls == 0, f"{llm.responsibility_calls} calls")


def test_single_file_path_still_self_assigns() -> None:
    print("\n=== single-file translation still assigns for itself ===")
    from rustgen.spec_stage import generate_specs

    llm = _CountingLLM()
    asyncio.run(generate_specs(llm, units(3), "", {}, RICH))
    check("omitting concerns runs the pass exactly once",
          llm.responsibility_calls == 1, f"{llm.responsibility_calls} calls")


def test_qualified_owner_reaches_the_right_unit() -> None:
    print("\n=== a `<stem>__<unit>` owner lands on that file's unit ===")
    from rustgen.spec_stage import generate_specs

    # the real array_list collision: demo.c owns nothing, project.c owns the
    # lookup both files were writing.
    concerns = [{"concern": "Workspace::find_task", "owner": "project__exp_0001"},
                {"concern": "impl Display for Task", "owner": "demo__exp_0002"}]

    demo = asyncio.run(generate_specs(_CountingLLM(), units(3), "", {}, RICH,
                                      concerns=concerns, id_prefix="demo__"))
    check("the owning unit owns its concern",
          demo["exp_0002"]["owns"] == ["impl Display for Task"],
          str(demo["exp_0002"]["owns"]))
    check("a cross-file concern is forbidden to the non-owner",
          "Workspace::find_task" in demo["exp_0002"]["must_not_implement"],
          str(demo["exp_0002"]["must_not_implement"]))
    check("a same-file non-owner is also forbidden it",
          demo["exp_0001"]["owns"] == []
          and "impl Display for Task" in demo["exp_0001"]["must_not_implement"],
          str(demo["exp_0001"]))

    proj = asyncio.run(generate_specs(_CountingLLM(), units(3), "", {}, RICH,
                                      concerns=concerns, id_prefix="project__"))
    check("the other file's owner owns the lookup",
          proj["exp_0001"]["owns"] == ["Workspace::find_task"],
          str(proj["exp_0001"]["owns"]))
    check("and is forbidden the other file's concern",
          proj["exp_0001"]["must_not_implement"] == ["impl Display for Task"],
          str(proj["exp_0001"]["must_not_implement"]))


def test_unqualified_prefix_would_have_assigned_nobody() -> None:
    print("\n=== known-bad: qualified owners with no prefix match nothing ===")
    from rustgen.spec_stage import generate_specs

    concerns = [{"concern": "Workspace::find_task", "owner": "project__exp_0001"}]
    out = asyncio.run(generate_specs(_CountingLLM(), units(3), "", {}, RICH,
                                     concerns=concerns, id_prefix=""))
    check("without the prefix the concern reaches no owner",
          all(s["owns"] == [] for s in out.values()),
          str({k: v["owns"] for k, v in out.items()}))
    check("...and every unit is merely forbidden it",
          all(s["must_not_implement"] == ["Workspace::find_task"]
              for s in out.values()), "")
    print("        (this is the shape the id_prefix argument exists to avoid:")
    print("         it fails silently — every spec still generates)")


def test_pass_sees_every_file_and_qualifies_ids() -> None:
    print("\n=== the pass is given all files' units, with qualified ids ===")
    from rustgen.spec_stage import assign_responsibilities

    llm = _CountingLLM()
    qualified = (units(2, "demo__") + units(2, "project__")
                 + units(1, "workspace__"))
    asyncio.run(assign_responsibilities(llm, qualified, "", failures=[]))
    check("one call covers the whole project",
          llm.responsibility_calls == 1, f"{llm.responsibility_calls} calls")
    check("every file's units are in that one prompt",
          len(llm.seen_unit_ids) == 5, str(llm.seen_unit_ids))
    check("ids are file-qualified, so owners are unambiguous",
          {i.split("__")[0] for i in llm.seen_unit_ids}
          == {"demo", "project", "workspace"}, str(llm.seen_unit_ids))


def test_failure_degrades_and_is_recorded() -> None:
    print("\n=== a failed pass degrades rather than aborting ===")
    from rustgen.spec_stage import assign_responsibilities

    class _Failing(_CountingLLM):
        async def ask_json(self, prompt: str, **kw):
            if self._is_responsibility(prompt):
                raise ValueError("synthetic responsibility failure")
            return {"signatures": [], "behavior_note": ""}

    failures: list[dict] = []
    try:
        got = asyncio.run(assign_responsibilities(_Failing(), units(3), "",
                                                  failures=failures))
        check("failure returns an empty assignment, does not raise",
              got == [], str(got))
        check("the loss is recorded so the run cannot be scored",
              len(failures) == 1
              and "responsibility" in failures[0].get("unit", ""),
              str(failures))
    except Exception as e:
        check("failure does not raise", False, f"{type(e).__name__}: {e}")


def test_malformed_concerns_are_dropped() -> None:
    print("\n=== malformed entries never reach a unit ===")
    from rustgen.spec_stage import assign_responsibilities

    llm = _CountingLLM(concerns=[
        {"concern": "impl Drop for Cache", "owner": "a__exp_0001"},
        {"concern": "no owner"},
        {"owner": "a__exp_0002"},
        "not a dict",
        {"concern": "", "owner": "a__exp_0003"},
    ])
    got = asyncio.run(assign_responsibilities(llm, units(1, "a__"), "",
                                              failures=[]))
    check("only the well-formed concern survives",
          got == [{"concern": "impl Drop for Cache", "owner": "a__exp_0001"}],
          str(got))


def main() -> int:
    test_supplied_concerns_suppress_the_per_file_call()
    test_single_file_path_still_self_assigns()
    test_qualified_owner_reaches_the_right_unit()
    test_unqualified_prefix_would_have_assigned_nobody()
    test_pass_sees_every_file_and_qualifies_ids()
    test_failure_degrades_and_is_recorded()
    test_malformed_concerns_are_dropped()
    test_prohibitions_are_scoped()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S): " + ", ".join(_failures))
        return 1
    print("ALL PASS")
    return 0


def test_prohibitions_are_scoped() -> None:
    """`must_not_implement` must not be the global complement.

    Scoping the pass to the project made `owns` project-wide, which was the
    point — and made the PROHIBITION list project-wide too, which was not.
    Per-file it averaged 1.9 entries per unit; project-wide it averaged 17-29,
    so a unit's own assigned concern sat inside ~25 near-identical "never
    implement these" lines naming items in files it never touches.
    `array_list base_srvB_t6`: project__exp_0001 owned `impl Project
    constructor`, carried 29 prohibitions beside it, and its repair deleted the
    constructor it owned.
    """
    print("\n=== prohibitions are scoped to types the unit mentions ===")
    from rustgen.spec_stage import generate_specs

    concerns = [
        {"concern": "impl Task constructor", "owner": "task__exp_0001"},
        {"concern": "impl fmt::Display for Task", "owner": "task__exp_0003"},
        {"concern": "impl Drop for Workspace", "owner": "workspace__exp_0001"},
        {"concern": "impl Activity constructor", "owner": "activity__exp_0001"},
    ]
    us = [Explanation(id="exp_0001", text="Builds a Task and prints it",
                      invariants=["a Task must have a title"],
                      ranges=[(1, 5)], status="locked")]
    out = asyncio.run(generate_specs(_CountingLLM(), us, "", {}, RICH,
                                     concerns=concerns, id_prefix="demo__"))
    # The RECORD stays complete — it is the audit trail of who owns what, and
    # filtering it would lose information. Only the PROMPT is scoped.
    mni = out["exp_0001"]["must_not_implement"]
    check("the full owner map still reaches the spec record",
          len(mni) == 4, str(mni))

    # what the PROMPT carries is the filtered set — that is the flood fix
    import rustgen.spec_stage as ss
    seen = {}

    class _Capture(_CountingLLM):
        async def ask_json(self, prompt, **kw):
            if not self._is_responsibility(prompt):
                seen["prompt"] = prompt
            return await super().ask_json(prompt, **kw)

    asyncio.run(generate_specs(_Capture(), us, "", {}, RICH,
                               concerns=concerns, id_prefix="demo__"))
    p = seen.get("prompt", "")
    check("a concern on a type the unit mentions is kept",
          "impl Task constructor" in p)
    check("a concern on a type it never mentions is dropped",
          "impl Drop for Workspace" not in p and "impl Activity constructor" not in p,
          "unrelated prohibitions leaked into the prompt")
    check("owning something is stated as mandatory",
          "mandatory" in p.lower() or "OWNS" in p)


if __name__ == "__main__":
    sys.exit(main())
