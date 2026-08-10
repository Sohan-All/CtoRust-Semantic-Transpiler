"""Stage T — type synthesis: one call per file.

Reads every final unit's description + invariants and produces the shared Rust
data model (types, error enum) plus stubs for non-trivial external functions,
and a glossary mapping description concepts to type names. Every later stage
builds against this vocabulary.
"""

from __future__ import annotations

import json

from llm import LLM, extract_json
from state import Explanation
from rustgen.common import (extract_rust, illegal_stubs, illegal_type_bodies,
                            stubbed_callbacks, unit_block)

# Regeneration attempts for a types block that stubs a unit's behaviour (see
# synthesize_types). Same budget as stage C's pre-flight, for the same reason:
# a fresh draw usually fixes it and the call is not cheap.
TYPES_RETRIES = 2

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
                           callback_names: frozenset[str] = frozenset()
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
        if repair:
            from rustgen.types_repair import repair_types_block
            repaired, rep = await repair_types_block(
                llm, types_rs, problem, units,
                context_rs=context_rs, c_source=c_source,
                max_tokens=max_tokens)
            if rep.get("repaired"):
                fixed = illegal_stubs(repaired) or illegal_type_bodies(repaired)
                if not fixed:
                    print(f"[types] repaired after {rep['rounds']} round(s) "
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
                                      "rounds": rep["rounds"],
                                      "questions": rep.get("questions", []),
                                      "note": f"types repair: {rep['why']}"})
                    return repaired, glossary if isinstance(glossary, dict) else {}
                # a repair that passed validate_repair but not the gates should
                # be impossible; if it happens, keep the original and say so
                problem = fixed
            print(f"[types] repair did not resolve it "
                  f"({rep.get('action')}: {rep.get('why') or '-'})")

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
