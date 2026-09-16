"""Stage T — type synthesis: one call per file.

Reads every final unit's description + invariants and produces the shared Rust
data model (types, error enum) plus stubs for non-trivial external functions,
and a glossary mapping description concepts to type names. Every later stage
builds against this vocabulary.
"""

from __future__ import annotations

import json

from llm import LLM, extract_json
from rustgen.escalation import Escalation
from state import Explanation
from rustgen.common import (extract_rust, illegal_stubs, illegal_type_bodies,
                            stubbed_callbacks, unit_block)

# Regeneration attempts for a types block that stubs a unit's behaviour (see
# synthesize_types). Same budget as stage C's pre-flight, for the same reason:
# a fresh draw usually fixes it and the call is not cheap.
TYPES_RETRIES = 2
# Only a fallback for callers that do not pass one; run_project always
# passes cfg.escalation_max_turns. Matches types_repair.REPAIR_ROUNDS.
REPAIR_TURNS_DEFAULT = 4

STUB_RETRY_NOTE = """\
Your previous reply is rejected: {problem}.

Stubs are allowed in EXACTLY ONE place — inside a `pub mod <stem>_deps`
module, for external DOMAIN functions implemented outside this project. A
`todo!()` anywhere else is a defect, and a silent one: it type-checks, so the
build stays green and the panic only appears when a test runs it.

The specific mistake to avoid: do NOT emit an `impl SomeType {{ ... }}` block
whose method bodies are `todo!()`. That invents an API which the unit stage
then TRUSTS — each unit sees the method already declared, assumes a sibling
implements it, and writes nothing. The result compiles and every call panics.
An observed run shipped seven stubbed methods this way and failed 20 of 26
differential tests.

You are designing types, not behaviour. Emit the data definitions, the error
enum with its Display/Error impls, and the `*_deps` stubs — and nothing that
has a body a unit ought to be writing."""

BODY_RETRY_NOTE = """\
Your previous reply is rejected: {problem}.

You are designing TYPES. A method with a working body is a unit's job, not
yours, and writing one here does not help that unit — it collides with it.
Rust refuses two inherent methods of the same name on one type whatever their
signatures, so `impl Scheduler {{ fn spawn(&mut self, task: SchedTask) }}`
here and the unit's own four-argument `spawn` are a hard error, and by then
neither side can be removed without deleting real code.

Emit the data definitions, the error enum with its Display/Error impls, and
the `*_deps` stubs. Nothing else. If a behavior seems to need a method, that
is a signal the unit stage will write it — leave the type bare."""

CALLBACK_RETRY_NOTE = """\
Your previous reply is rejected: {problem}.

Those names are PARAMETERS of the C functions, not functions implemented
elsewhere. The C declares them as function pointers, e.g.
`bool cc_array_filter(CC_Array *ar, bool (*pred)(const void *))` — `pred` is
supplied fresh by each caller, so there is no single implementation and a
`deps` stub for it can never be filled. It compiles, then panics.

Remove those stubs entirely. The unit that receives the callback will take a
generic parameter bounded by `Fn`/`FnMut` and call it directly; nothing needs
to exist in the types block for that to work."""

TYPES_PROMPT = """\
A C source file has been decomposed into behavioral units, each described in
language-agnostic terms below. You are designing the SHARED Rust data model
that a later step will implement each unit against.

Produce:
1. Idiomatic, safe Rust type definitions covering every data concept the units
   mention: structs, enums, and ONE error enum for all failure modes described.
   Use owned types (String, Vec<u8>, Vec<T>), derive Debug/Clone/PartialEq
   where sensible, and never use raw pointers or unsafe. Recursive structures
   use Vec<Child> or Box. Add a one-line doc comment per type.
   Fallible operations return `Result<T, E>`. Never define a success/failure
   status enum (`Ok`/`Failed`/`Success` variants) — that is a C idiom; the
   error enum plus `Result` covers it. The error enum MUST also get a
   hand-written `impl std::fmt::Display` and `impl std::error::Error`.
   BUT: if the program's C exits with more than one distinct non-zero status
   (e.g. `return 1` from one path and `return 2` from another), those numbers
   are observable output, and `Result` must not flatten them. Give the error
   enum enough variants to tell them apart, so the entry point can map each
   back to its own exit code. An `Err(_) => 1` catch-all that turns exit code 2
   into 1 is a behavior change no compiler will flag.
   Visibility: `pub(crate)` on types and functions (struct fields may be
   `pub`) — the crate's only `pub` items are the program entry point and FFI
   exports.
2. A `pub mod deps` containing stub functions for external functions the units
   call that are DOMAIN functions specified elsewhere (e.g. git_*). Every stub
   MUST be a complete function with a `{{ todo!() }}` body, never a bare
   signature ending in `;`:
   ```rust
   pub fn example_fn(x: i32) -> Result<i32, ProjectError> {{ todo!() }}
   ```
   Do NOT stub C standard library operations (memory/string/IO primitives) —
   implementations will use Rust std instead. Do NOT stub any function named
   as a SIBLING FILE function above, if that section is present — those are
   called directly, never stubbed.
   Do NOT stub a CALLBACK the caller supplies. A C parameter like
   `bool (*pred)(const void *)` or `ArrayListCompareFunc compare_func` is a
   PARAMETER, not an external function: the unit that receives it takes a
   generic bounded by `Fn`/`FnMut` and calls it. There is no single
   implementation to stub, because every caller passes a different one, so a
   `deps` stub for it can never be filled and will panic at run time. If a unit
   description says a function "applies the caller's comparison function" or
   similar, that is this case.
3. A GLOSSARY mapping each recurring concept phrase from the descriptions to
   its Rust type name.

Do not implement any unit's behavior — types and stubs only.

UNITS:
{units}

Reply in exactly this layout:
```rust
<type definitions and pub mod deps>
```
GLOSSARY:
```json
{{"<concept phrase>": "<RustTypeName>", ...}}
```
"""


async def synthesize_types(llm: LLM, units: list[Explanation],
                           max_tokens: int,
                           project_block: str = "",
                           failures: list[dict] | None = None,
                           notes: list[dict] | None = None,
                           repair: bool = True,
                           context_rs: str = "",
                           c_source: str = "",
                           callback_names: frozenset[str] = frozenset(),
                           escalation_llm=None,
                           escalations: list[dict] | None = None,
                           escalation_rounds: int = REPAIR_TURNS_DEFAULT,
                           drafts: list[dict] | None = None,
                           agent: bool = False,
                           agent_c_root=None,
                           agent_reads_pipeline: bool = True,
                           agent_max_tool_calls: int = 40,
                           agent_wall_seconds: float = 600.0,
                           ) -> tuple[str, dict]:
    """`project_block` (multi-file translation): shared project types +
    sibling-function notice, prepended as fixed context — this file's Stage T
    then defines ONLY file-local types and must not re-stub sibling fns.

    `repair` (cfg.rustgen_types_repair, ablation only) runs a targeted repair
    when the gates exhaust their retries instead of giving up there. `context_rs`
    is the project's already-generated shared Rust — the repair compiles the
    block against it, so a cross-file reference is not mistaken for a defect the
    repair introduced. `c_source` is this file's C, for the repair's questions.

    `notes` receives a record when the repair SUCCEEDS — a separate list from
    `failures` on purpose, because `failures` is the caller's degraded list and
    anything in it voids the run. A rescued block is the opposite of a loss.

    `escalation_llm` (plan.md site A) is a STRONGER model, tried only after the
    local repair has failed. Cheapest possible placement for the most leverage
    in the pipeline: one block per project rather than one per unit, upstream of
    every unit so a fix propagates to all of them, and validated the same way —
    the candidate has to survive `validate_repair` and the gates below. On any
    failure the original block comes back and this branch proceeds exactly as it
    did before, so escalation cannot make a good run worse either. Records land
    in `escalations`, accepted or not: a firing that vanishes looks identical to
    one that never happened, except that it was paid for.

    `drafts` receives every REJECTED draft (attempt, the gate's complaint, the
    block). Previously only the final `project_types_all` was stored, so a
    gate-voided run could not be audited at all — and the two recorded repair
    notes both claim the gate rejected a `deps`-module stub it should permit,
    which is exactly the claim these drafts make checkable.

    `failures` receives a record when the stub gate exhausts its retries. That
    outcome is not survivable in practice and used to be a printed warning the
    run then ignored: a stubbed shared block gives every unit a phantom API to
    defer to, and when a unit implements the method for real anyway the two
    collide as E0592. The compile loop cannot resolve that — it deliberately
    will not rewrite the shared block — so the only move left is deleting the
    real implementation, which `emptied_blocks` refuses. `binary_heap
    base_srvB_t2` stalled exactly there: nine refused repairs, zero accepted,
    `12 -> 7 -> 7 -> 7`, and the two lines below were the only warning."""
    prompt = TYPES_PROMPT.format(units="\n\n".join(unit_block(u) for u in units))
    if project_block:
        prompt = project_block + "\n" + prompt

    # Regenerate a types block that stubs unit behaviour. Stage T is told to
    # confine `todo!()` to `pub mod *_deps`, but when it instead emits an
    # `impl` block of stubbed methods it manufactures a phantom API: the unit
    # stage sees the methods declared, assumes a sibling owns them, and emits
    # nothing. Nothing downstream catches that — the stubs type-check, the
    # crate builds, and the compile loop's own stub gate only looks at MTU
    # sections, not at the shared types block this lands in. Catching it here
    # is the only cheap place.
    base_prompt = prompt
    types_rs, glossary, problem = "", {}, ""
    for attempt in range(TYPES_RETRIES + 1):
        reply = await llm.ask(prompt, max_tokens=max_tokens)
        # first fence is the Rust; glossary is the json fence after "GLOSSARY:"
        glossary = {}
        if "GLOSSARY:" in reply:
            rust_part, gloss_part = reply.split("GLOSSARY:", 1)
            try:
                glossary = extract_json(gloss_part)
            except ValueError:
                glossary = {}
        else:
            rust_part = reply
        types_rs = extract_rust(rust_part)
        # Two distinct defects, deliberately reported separately: a stubbed
        # body invents a phantom API units defer to, an IMPLEMENTED body
        # duplicates one a unit will write. Both end as an impl block the
        # compile loop cannot repair, from opposite directions, and the retry
        # note that helps one would confuse the other.
        # Checked FIRST and reported separately. A callback stub is also a
        # `todo!()` inside a `*_deps` module, which `illegal_stubs` permits by
        # design — so without its own check this shape is invisible, and it is
        # the one `sibling_deps` has been observed FABRICATING a body for
        # (`cp` -> `Ok(item.clone())`) rather than leaving as an honest stub.
        problem = (stubbed_callbacks(types_rs, callback_names)
                   or illegal_stubs(types_rs) or illegal_type_bodies(types_rs))
        if not problem:
            break
        # Persist the rejected draft BEFORE re-prompting. Without this a
        # gate-voided run keeps only the final draw, so the one question worth
        # asking afterwards — what exactly did the gate object to, and was it
        # right — cannot be asked at all.
        if drafts is not None:
            drafts.append({"attempt": attempt, "problem": problem,
                           "block": types_rs})
        note = (CALLBACK_RETRY_NOTE if "CALLBACK" in problem else
                STUB_RETRY_NOTE if "stub" in problem else BODY_RETRY_NOTE)
        if attempt < TYPES_RETRIES:
            prompt = base_prompt + "\n\n" + note.format(problem=problem)
    else:
        # Exhausted. Regenerating from the same prompt is re-rolling the dice at
        # temperature 1.0, and the outcome is not survivable: over 255 recorded
        # run logs, EVERY run reaching this branch died (4 BUILD_FAILED, 3
        # STUB_CRATE, none scored). So try a targeted repair before giving up —
        # it is told what is wrong, it can ask which units own the behaviour,
        # and its output has to survive four checks including a real cargo
        # check. It cannot make a good block worse: on any failure the original
        # comes back and this branch proceeds exactly as it did before.
        # ONE attempt, driven by either model. Factored rather than copied
        # because the acceptance path is the load-bearing part — re-running the
        # gates on the candidate, and routing a success to `notes` and never to
        # `failures` — and a second hand-written copy of it for the escalated
        # attempt is precisely how the two drift.
        def _accept(repaired, rep, label: str):
            """The acceptance half of an attempt, shared by every proposer.

            Factored out when the agent path landed at site A, for the reason
            the comment above already gives: this is the load-bearing part, and
            a second hand-written copy of it is how the two drift. The agent
            path differs ONLY in how the candidate was produced — the gates it
            must survive and the notes-not-failures routing are identical, and
            no gate relaxes because the proposer got stronger.
            """
            nonlocal problem
            if rep.get("repaired"):
                fixed = illegal_stubs(repaired) or illegal_type_bodies(repaired)
                if not fixed:
                    print(f"[types] {label} repair succeeded after "
                          f"{rep['rounds']} round(s) "
                          f"(asked: {', '.join(rep['questions']) or 'nothing'})")
                    # A SUCCESS goes to `notes`, never to `failures`. This
                    # record used to land in `failures`, which is the degraded
                    # list — so the one outcome this module exists to produce
                    # marked its own run INCOMPLETE and unscoreable, and the
                    # record carried no `error` key so the degraded printer
                    # died on KeyError before the run could even be voided
                    # quietly. Both firings in the 08-03 batch were repairs
                    # that WORKED on crates that then compiled clean
                    # (binary_heap srvB t12 12/12, cc_array srvA t3 21/21).
                    # The crash was the only reason it was visible.
                    if notes is not None:
                        notes.append({"stage": "types",
                                      "unit": "(shared types)",
                                      "repaired": True,
                                      "by": label,
                                      "rounds": rep["rounds"],
                                      "questions": rep.get("questions", []),
                                      "note": f"types repair: {rep['why']}"})
                    return repaired, rep
                # a repair that passed validate_repair but not the gates should
                # be impossible; if it happens, keep the original and say so
                problem = fixed
            print(f"[types] {label} repair did not resolve it "
                  f"({rep.get('action')}: {rep.get('why') or '-'})")
            return None, rep

        async def _attempt(rllm, label: str, rounds: int | None = None):
            from rustgen.types_repair import repair_types_block, REPAIR_ROUNDS
            repaired, rep = await repair_types_block(
                rllm, types_rs, problem, units,
                context_rs=context_rs, c_source=c_source,
                max_tokens=max_tokens,
                rounds=REPAIR_ROUNDS if rounds is None else rounds)
            return _accept(repaired, rep, label)

        async def _agent_attempt(rllm, label: str, rounds: int | None = None):
            """SITE A's agent path (plan.md item 6). Same acceptance, same
            gates; only the proposer's read surface changes.

            Owns its staged root's whole lifetime in a `finally` — the tree is a
            copy, and one left behind per firing fills a scratch disk over a
            batch. A staging failure is a firing that did not happen, and is
            recorded as such rather than killing an already-failing stage.
            """
            import shutil
            import tempfile
            from pathlib import Path
            from rustgen.agent import AgentBudget, StagedRootError
            from rustgen.agent_sites import agent_repair_types, stage_for_types

            if agent_c_root is None:
                return None, {"action": "give_up", "repaired": False,
                              "why": "no c_root available to stage"}
            tmp = Path(tempfile.mkdtemp(prefix="mtu_agent_types_"))
            try:
                try:
                    staged = stage_for_types(
                        tmp / "root", agent_c_root,
                        include_pipeline=agent_reads_pipeline)
                except StagedRootError as e:
                    print(f"[types] agent NOT run — {e}")
                    return None, {"action": "give_up", "repaired": False,
                                  "why": f"staging refused: {e}"}
                repaired, rep = await agent_repair_types(
                    rllm, types_rs, problem, staged=staged, drafts=drafts,
                    context_rs=context_rs,
                    budget=AgentBudget(
                        max_turns=rounds or REPAIR_TURNS_DEFAULT,
                        max_tool_calls=agent_max_tool_calls,
                        wall_seconds=agent_wall_seconds),
                    max_tokens=max_tokens * 2)
                return _accept(repaired, rep, label)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        if repair:
            block, _ = await _attempt(llm, "local")
            if block is not None:
                return block, glossary if isinstance(glossary, dict) else {}

            # SITE A. Only after the local repair has failed, so the cheap
            # model is always tried first and escalation is charged only for
            # what it actually rescues.
            if escalation_llm is not None:
                esc = Escalation("types", escalation_llm,
                                 trigger="gate_exhausted", value=problem[:120])
                block = None
                # The agent path swaps the PROPOSER, never the acceptance. Both
                # branches end in `_accept`, so a candidate from either has to
                # survive the same gates and lands in `notes` the same way.
                proposer = _agent_attempt if agent else _attempt
                try:
                    with esc:
                        block, rep = await proposer(escalation_llm, "escalated",
                                                    escalation_rounds)
                        esc.done(accepted=block is not None,
                                 rejected_by=("" if block is not None
                                              else str(rep.get("action") or "")))
                except Exception as ex:
                    # This branch is already dying; an escalation that raises
                    # must not convert a loud, recorded stage-T failure into a
                    # crashed run. `esc.record` carries the exception and the
                    # cost, so it is survivable without being invisible.
                    block = None
                    print(f"[types] escalation raised: "
                          f"{type(ex).__name__}: {ex}")
                finally:
                    if escalations is not None:
                        escalations.append(esc.record)
                if block is not None:
                    return block, glossary if isinstance(glossary, dict) else {}

        # keep the last draw (the compile loop and the assembled crate's stub
        # gate still get a say) but make it loud in the log AND in the record —
        # a printed warning alone let a doomed run proceed to scoring as if it
        # were ordinary
        print(f"[types] {problem} — still present after {TYPES_RETRIES} "
              f"retries; units may defer to these stubs")
        if failures is not None:
            failures.append({"stage": "types", "unit": "(shared types)",
                             "error": f"stub gate exhausted after "
                                      f"{TYPES_RETRIES} retries: {problem}"})
    return types_rs, glossary if isinstance(glossary, dict) else {}
