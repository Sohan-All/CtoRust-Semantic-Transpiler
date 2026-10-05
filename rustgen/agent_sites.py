"""Site-specific glue between the read agent and the transaction gate.

`agent.py` knows how to read and propose; `transaction.py` knows how to accept
or refuse. Neither knows what a stub is. This module is where a SITE says what
the job is, and it is kept separate so the generic halves stay testable without
a crate on disk.

The acceptance rule is the site's, not the agent's, and it does not soften
because the proposer got stronger (plan.md D5). Site C still means `after == 0`:
`remaining_stubs` voids a crate for one stub, so clearing four of five buys
exactly nothing — `cc_array base` cleared 5 and stayed STUB_CRATE in all 6 runs,
and the first escalation batch cleared 2 of 7 on `array_list` and was correctly
reverted.
"""
from __future__ import annotations

from pathlib import Path

from rustgen.agent import AgentBudget, StagedRoot, run_agent
from rustgen.transaction import behaviour_lost, check_transaction


def stub_task(stubs, sections: dict[str, str]) -> str:
    """The job description for site C.

    States the all-or-nothing rule explicitly, because it changes what a
    rational agent should do: with a step function at zero, clearing the easy
    four and declining the fifth is worth the same as declining all five, and an
    agent that does not know that will spend its budget on the easy ones. The
    first batch's `array_list` firing spent 16,187 output tokens doing exactly
    that.
    """
    rows = []
    for s in stubs:
        where = f"{s.section}:{s.line}"
        fn = f" in `{s.function}`" if s.function else ""
        msg = f" — its note: {s.message!r}" if s.message else " — no message"
        rows.append(f"  {where}{fn}{msg}")
    listing = "\n".join(rows) if rows else "  (none)"
    return f"""\
This crate still contains {len(stubs)} `todo!()` stub(s). A stub is a runtime
panic, so the crate cannot be scored at all while any of them remain:

{listing}

Clear ALL of them or none. This is a step function, not a gradient — a crate
with one stub left is worth exactly as much as a crate with {len(stubs)}, so
clearing the easy ones and declining the rest is worth nothing and costs real
money. If even one of them cannot be resolved without inventing behaviour the
project does not contain, say `give_up` and explain which one and why.

The stub messages are the previous model's guesses and several are known to be
WRONG. One said "no sibling provides populate_atlas" about a function that was
defined 600 lines above it. Another waited for `scheduler_print_report`, whose
behaviour was present as `impl Display for Scheduler`. Check the message against
the crate before you believe it; the translator renames things, so search for
what a function DOES as well as what it is called.

Sections you may edit: {', '.join(sorted(sections))}
"""


async def agent_clear_stubs(llm, sections: dict[str, str], stubs, *,
                            staged: StagedRoot,
                            shared_id: str = "",
                            compile_check=None,
                            budget: AgentBudget | None = None,
                            max_tokens: int = 8000
                            ) -> tuple[dict[str, str], dict]:
    """One agent firing against the stub residue.

    Returns `(sections, record)`. On any refusal `sections` is the input
    unchanged — the transaction gate is all-or-nothing and this returns its
    verdict rather than a partially applied state.

    The record carries the agent's side AND the transaction's side, because a
    firing can fail at either and they call for opposite responses: an agent
    that gave up needs better context, a transaction that was refused needs a
    different edit. The first batch could not tell those apart, because the
    escalated pass's report was computed and discarded.
    """
    res = await run_agent(llm, staged, stub_task(stubs, sections),
                          budget=budget, max_tokens=max_tokens)
    record = {"type": "agent_site", "site": "stubs",
              "stubs_before": len(stubs), "agent": res.record}

    if res.action != "propose":
        record["outcome"] = f"agent_{res.action}"
        record["accepted"] = False
        return sections, record

    tx = check_transaction(sections, res.edits, shared_id=shared_id,
                           compile_check=compile_check)
    record["transaction"] = tx.record
    record["accepted"] = tx.accepted
    if not tx.accepted:
        record["outcome"] = "transaction_refused"
        return sections, record

    # Diagnostic, not a gate — see `behaviour_lost`. If the agent arm turns out
    # to buy build rate by deleting behaviour, this is the number that shows it,
    # and it has to be recorded from the first firing or it is not there to look
    # at afterwards.
    record["behaviour_lost"] = behaviour_lost(sections, tx.sections)
    record["outcome"] = "applied"
    return tx.sections, record


TYPES_TASK = """\
You are stage T of a C-to-Rust translator. You produce the project's SHARED
TYPES block: the data definitions every translated unit builds against.

Your block did not pass this project's lint {n} time(s), and the automatic repair
could not resolve it either. The lint message is:

    {problem}

That message comes from this project's own lint rules, which you can read under
`pipeline/`. Worth reading before you rewrite anything: the rules are simple
regex and name checks, so the message tells you which rule matched, which is not
always the same as what is wrong with your design. Sometimes the block is right
in substance and simply does not follow a naming or placement convention the
project enforces — in that case say so in your `why`, because a block redesigned
for the wrong reason is worse than one left alone.

What this block is allowed to contain: data definitions, the error enum with its
Display/Error impls, and stubs for external DOMAIN functions. It must NOT contain
method bodies a translated unit ought to be writing — a stubbed method here is a
phantom API that every unit defers to, and an observed run shipped seven of them
and failed 20 of 26 differential tests.

{drafts}

EVERY TYPE YOUR CURRENT BLOCK DEFINES MUST STILL BE DEFINED IN YOUR REPLACEMENT.
Units in other files are generated in parallel against these definitions, so
removing one breaks every unit that names it — and you cannot see those units
from here. Count the `struct`/`enum`/`type` items in the block above and check
every one reappears before you reply. Methods may go; types may not.

Propose the complete replacement block in ONE fenced section:

```rust section=__shared__
...the complete block...
```
"""


def types_task(problem: str, drafts: list[dict]) -> str:
    """Site A's job description.

    Deliberately does NOT name the defect. Both classifiable site-A voids on
    record were `_DEPS_MOD` refusing a module called `deps` because it wants
    `*_deps` — and writing "check your deps module's name" into this prompt
    would mean the PROMPT solved it, not the agent, and the measurement would
    be of nothing. It says where the checkers live and lets the agent look.
    """
    if drafts:
        shown = "\n\n".join(
            f"Your draft on attempt {d.get('attempt', i)}, rejected with "
            f"{d.get('problem', '?')!r}:\n\n```rust\n{d.get('block', '')}\n```"
            for i, d in enumerate(drafts[-3:]))
        shown = "Your rejected drafts:\n\n" + shown
    else:
        shown = "(no drafts were recorded)"
    return TYPES_TASK.format(n=len(drafts) or 1, problem=problem, drafts=shown)


async def agent_repair_types(llm, types_rs: str, problem: str, *,
                             staged: StagedRoot,
                             drafts: list[dict] | None = None,
                             context_rs: str = "",
                             budget: AgentBudget | None = None,
                             max_tokens: int = 8000,
                             checker=None) -> tuple[str | None, dict]:
    """One agent firing at site A. Returns `(block_or_None, rep)`.

    `rep` is shaped like `repair_types_block`'s so the caller's acceptance path
    — re-running the gates and routing a success to `notes` rather than
    `failures` — is reached unchanged. That path is the load-bearing part and
    `types_stage` already warns that a second hand-written copy of it is exactly
    how the two drift.

    The candidate is validated by `validate_repair`, the SAME function the local
    repair's output goes through: brace balance, `lost_type_definitions`, and a
    real standalone `cargo check` of the block. No gate is relaxed because the
    proposer got stronger (plan.md D5).
    """
    from rustgen.types_repair import cargo_error_signatures, validate_repair

    res = await run_agent(llm, staged, types_task(problem, drafts or []),
                          budget=budget, max_tokens=max_tokens)
    rep = {"action": res.action, "why": res.why, "repaired": False,
           "rounds": res.turns, "questions": [f"{res.tool_calls} tool call(s)"],
           "agent": res.record}
    if res.action != "propose" or not res.edits:
        return None, rep
    if len(res.edits) > 1:
        # The shared types are ONE section. Several fences means the agent
        # thought it was editing units, and picking one silently would ship
        # whichever happened to sort first.
        rep["why"] = (f"proposed {len(res.edits)} sections; the shared types "
                      f"are a single block")
        return None, rep

    sid, candidate = next(iter(res.edits.items()))
    rep["section_id"] = sid          # recorded: a wrong id is worth seeing
    bad = validate_repair(types_rs, candidate, context_rs,
                          checker or cargo_error_signatures)
    if bad:
        rep["why"] = f"rejected by validate_repair: {bad}"[:500]
        return None, rep
    rep["repaired"] = True
    return candidate, rep


CLUSTER_TASK = """\
These sections of a machine-translated Rust crate do not compile, and the errors
tie them together — a change to one has to be matched in the others.

The compiler says:

{errors}

The sections, in full:

{sections}

You may edit ONLY these sections: {ranked}. Anything you send for another
section is discarded. Your edits land as ONE transaction — all of them or none —
so a signature change and its call sites go in the same reply or neither lands.

Two things worth knowing about this crate. It was translated from C, and the C
is under `c_source/` — no compile repair in this pipeline has ever been able to
read it, so when an error looks like a behavioural misunderstanding rather than a
type slip, the original is there. And the translator deliberately re-derives
control flow rather than transliterating, so a function's Rust name and shape may
differ from the C it came from; search for what something DOES, not only for what
it is called.

Do not make it compile by deleting what does not compile. Removing the call that
fails to type-check makes the build green and the translation wrong, and a wrong
translation scores worse than an honest failure.
"""


def cluster_task(ranked: list[str], sections: str, errors: str) -> str:
    """Site B's job description for one error cluster."""
    return CLUSTER_TASK.format(ranked=", ".join(ranked), sections=sections,
                               errors=errors)


def stage_for_types(tmp: Path, c_root: Path, *,
                    include_pipeline: bool = True,
                    label: str = "types") -> StagedRoot:
    """Site A's staged root: C source and the pipeline, and no crate.

    Stage T runs BEFORE any unit code exists, so there is nothing to stage as
    `crate/`. The pipeline half is the part that matters here — the complaint an
    agent has to interpret is produced by a regex it can go and read.
    """
    from rustgen.agent import stage
    pipeline = Path(__file__).resolve().parent.parent if include_pipeline else None
    return stage(tmp, c_src=Path(c_root) / "src", pipeline_src=pipeline,
                 label=label)


def stage_for_run(tmp: Path, crate: Path, c_root: Path, *,
                  include_pipeline: bool = True,
                  label: str = "") -> StagedRoot:
    """Build the staged root for one firing.

    `c_root` is the directory CONTAINING `src/`, and in the corpus layout
    `test_vectors/` is its sibling — so only `src/` is staged, and
    `assert_contained` re-checks the result by walking it rather than trusting
    that this function did the right thing.
    """
    from rustgen.agent import stage
    pipeline = Path(__file__).resolve().parent.parent if include_pipeline else None
    return stage(tmp,
                 crate_src=crate,
                 c_src=Path(c_root) / "src",
                 pipeline_src=pipeline,
                 label=label)
