"""Transaction-scoped gates for multi-section edits. See ../../plan.md item 6.

WHY THIS EXISTS. Every write path in this pipeline until now has been
single-section: `compile_loop.set_section` takes one section id,
`stub_repair.splice_function` takes one function. That is the right shape for a
repair and the wrong shape for the residue the first escalation batch left
behind. `array_list`'s three surviving `populate_*` stubs are E0502 borrow
conflicts — `populate_atlas(&mut Project, &Workspace)` cannot be called while
`project` comes from `workspace.projects.iter_mut()` — and the fix changes a
signature in ONE section and its call sites in ANOTHER. There is no sequence of
single-section edits that passes through a valid intermediate state.

WHAT A TRANSACTION IS. A set of section edits accepted or rejected as a unit.
The all-or-nothing property is not a convenience: a signature change whose call
sites do not land is strictly worse than no change at all, because the crate now
fails to compile for a reason nothing in the loop can attribute. Same step
function that made site C's stub gate all-or-nothing, arriving from the other
direction.

THE CHECKS ARE THE OLD CHECKS, RESCOPED — NOT RELAXED. This is the part that
matters. `lost_impl_methods` asks "did this edit drop a method other sections
still call", and answers it by looking at `elsewhere`. Run per-section against
the state BEFORE the transaction, it reports every legitimate cross-section move
as a deletion. Run against the state AFTER, it reports exactly what it was
written to report. The rescoping is the whole fix; the rule is untouched.

WHAT IS DELIBERATELY NOT RESCOPED. `emptied_blocks` stays per-section and stays
strict. `common.py:220-240` records a deduplication exemption for exactly this
check that was tried, looked sound, and was WRONG TWICE — once because `*_deps`
stubs count as definitions so the exemption permitted a genuine gutting, and once
because restricting it to non-stub definitions flipped the other fixture. That
comment ends "do not reintroduce this without doing the same", and a transaction
is a bigger hole than the one that was refused, not a smaller one. A legitimate
refactor that must empty an entire impl block can be rejected; the cost is a
refused edit, and the cost of the other error is a silently gutted crate.

Convention this module is built on: a gate must cover every writer. The
transaction path is a NEW writer, so the checks live here at its write point
rather than at each caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rustgen.common import (demote_dangling_docs, emptied_blocks,
                            illegal_type_bodies, lost_impl_methods,
                            parse_regression)


@dataclass
class TxResult:
    """The outcome of one proposed transaction.

    A REJECTED transaction is recorded as fully as an accepted one. The first
    escalation batch computed a `StubReport` for the escalated pass and threw it
    away (`run_project.py:690`), which is why "which two of the seven stubs did
    it clear" had to be re-derived with cargo instead of read out of the record.
    """
    accepted: bool = False
    problems: list[dict] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)   # state to keep
    touched: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    compile_problem: str = ""

    @property
    def record(self) -> dict:
        return {
            "type": "transaction",
            "accepted": self.accepted,
            "touched": sorted(self.touched),
            "checks_run": sorted(set(self.checks_run)),
            "problems": self.problems,
            "compile_problem": self.compile_problem[:2000],
        }

    def summary(self) -> str:
        if self.accepted:
            return (f"[tx] ACCEPTED {len(self.touched)} section(s): "
                    f"{', '.join(sorted(self.touched))}")
        first = (self.problems[0]["problem"] if self.problems
                 else self.compile_problem.split("\n")[0] or "no reason recorded")
        return (f"[tx] REJECTED {len(self.touched)} section(s) "
                f"({len(self.problems)} problem(s)): {first}")


def _elsewhere(sections: dict[str, str], sid: str) -> str:
    """Every section but `sid`, concatenated. The `elsewhere` argument the
    checkers already take — the only question is which STATE it is drawn from,
    and for a transaction the answer is the post-edit one."""
    return "\n".join(c for s, c in sections.items() if s != sid and c)


def validate_edits(before: dict[str, str],
                   edits: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    """Structural admissibility of the edit set itself, before any checker runs.

    Refuses an edit naming a section that does not exist. A transaction may
    rewrite the crate; it may not GROW it. Creating sections is assembly's job
    and an agent inventing one produces a section no MTU banner covers, which
    every downstream sweep keys on. This is also the cheap half of "may not
    write outside the crate" (plan.md D12).
    """
    clean: dict[str, str] = {}
    problems: list[dict] = []
    for sid, text in edits.items():
        if sid not in before:
            problems.append({"section": sid, "check": "unknown_section",
                             "problem": f"no section `{sid}` in this crate; a "
                                        f"transaction may rewrite sections, "
                                        f"not create them"})
            continue
        if not text or not text.strip():
            problems.append({"section": sid, "check": "empty_edit",
                             "problem": "the edit is empty; to delete a "
                                        "section's contents say so explicitly "
                                        "rather than sending nothing"})
            continue
        # Same deterministic cleanup `set_section` applies before its guards:
        # an orphaned `///` is a hard rustc error the loop cannot repair, and
        # edits produce them by deleting the item a doc comment belonged to.
        clean[sid] = demote_dangling_docs(text)
    return clean, problems


def check_transaction(before: dict[str, str], edits: dict[str, str], *,
                      shared_id: str = "",
                      compile_check=None) -> TxResult:
    """Accept or refuse a multi-section edit as a unit.

    `compile_check(sections) -> str` reassembles and runs cargo, returning ""
    when clean — the same contract `stub_repair.repair_stubs` already uses, so
    the existing baseline-subtracting closure in `run_project` is reusable
    unchanged. It runs ONCE per transaction rather than once per section:
    intermediate states of a coherent multi-section edit do not compile by
    construction, so checking them would reject every transaction worth making.

    On refusal the caller is left exactly where it started — `result.sections`
    is `before`, not a partially applied state.
    """
    result = TxResult(sections=dict(before), touched=sorted(edits))
    clean, problems = validate_edits(before, edits)
    result.problems.extend(problems)
    result.checks_run.append("validate_edits")
    if not clean:
        if not result.problems:
            result.problems.append({"section": "", "check": "empty_transaction",
                                    "problem": "the transaction proposes no edits"})
        return result

    after = dict(before)
    after.update(clean)

    for sid, new in clean.items():
        prev = before.get(sid, "")

        # Per-section and correctly so: an unbalanced section is a PARSE error,
        # and rustc reports exactly one of those per crate however much else is
        # wrong. Asymmetric, like the original — a section that was ALREADY
        # unbalanced stays writable, or a broken section freezes out of reach of
        # the edit that would fix it.
        problem = parse_regression(prev, new)
        result.checks_run.append("parse_regression")
        if problem:
            result.problems.append({"section": sid, "check": "parse_regression",
                                    "problem": problem})
            continue

        # NOT rescoped. See the module docstring: the exemption this would need
        # was tried on `emptied_blocks` and was wrong in both directions.
        problem = emptied_blocks(prev, new)
        result.checks_run.append("emptied_blocks")
        if problem:
            result.problems.append({"section": sid, "check": "emptied_blocks",
                                    "problem": problem})
            continue

        # THE RESCOPED ONE. `elsewhere` is drawn from `after`, so a method that
        # moved to another section in this same transaction is a move and not a
        # loss — which is precisely what a cross-section signature change looks
        # like, and precisely what the per-section form would refuse.
        problem = lost_impl_methods(prev, new, _elsewhere(after, sid))
        result.checks_run.append("lost_impl_methods")
        if problem:
            result.problems.append({"section": sid, "check": "lost_impl_methods",
                                    "problem": problem})
            continue

        if shared_id and sid == shared_id:
            problem = _shared_types_problem(prev, new)
            result.checks_run.append("illegal_type_bodies")
            if problem:
                result.problems.append({"section": sid,
                                        "check": "illegal_type_bodies",
                                        "problem": problem})
                continue

    if result.problems:
        return result

    # ONE cargo check, on the whole after-state, last — it is the expensive
    # gate and the structural ones above are free.
    if compile_check is not None:
        result.checks_run.append("compile_check")
        try:
            problem = compile_check(after)
        except Exception as e:
            # cargo unusable is not a verdict on the edit. Same choice as
            # `_compile_check`'s bare `except`: refusing here would make a
            # missing toolchain look like a bad transaction.
            problem = ""
            result.compile_problem = f"(compile check unavailable: {type(e).__name__})"
        if problem:
            result.compile_problem = problem
            result.problems.append({"section": "", "check": "compile_check",
                                    "problem": "the transaction introduces "
                                               "compile errors"})
            return result

    result.accepted = True
    result.sections = after
    return result


def _shared_types_problem(prev: str, new: str) -> str:
    """The shared block holds TYPES. Asymmetric for the same reason as
    `set_section`'s copy: a block that already had bodies stays writable, or the
    edit that would remove them is refused."""
    added = illegal_type_bodies(new)
    if added and not illegal_type_bodies(prev):
        return f"transaction implements behaviour in the types block: {added}"
    return ""


def behaviour_lost(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Function names defined anywhere in `before` and nowhere in `after`.

    The whole-crate view none of the per-section checkers has. `lost_impl_methods`
    asks its question about impl methods of an EDITED section; this asks it about
    the crate, which is the level a transaction operates at.

    Diagnostic rather than a gate: reported into the record so a transaction that
    reduced the crate is visible, but not refused here, because
    `lost_impl_methods` already refuses the case with live callers and refusing
    dead-code removal on top of that would block legitimate cleanup. If the agent
    arm turns out to shrink crates, this is the number that will show it.
    """
    import re
    from rustgen.common import _blank_literals
    def names(secs):
        out = set()
        for c in secs.values():
            out |= set(re.findall(r"\bfn\s+(\w+)", _blank_literals(c or "")))
        return out
    return sorted(names(before) - names(after))
