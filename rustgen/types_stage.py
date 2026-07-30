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
from rustgen.common import extract_rust, illegal_stubs, unit_block

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
                           project_block: str = "") -> tuple[str, dict]:
    """`project_block` (multi-file translation): shared project types +
    sibling-function notice, prepended as fixed context — this file's Stage T
    then defines ONLY file-local types and must not re-stub sibling fns."""
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
        problem = illegal_stubs(types_rs)
        if not problem:
            break
        if attempt < TYPES_RETRIES:
            prompt = base_prompt + "\n\n" + STUB_RETRY_NOTE.format(problem=problem)
    else:
        # exhausted: keep the last draw (the compile loop and the assembled
        # crate's stub gate still get a say) but make it loud in the log
        print(f"[types] {problem} — still present after {TYPES_RETRIES} "
              f"retries; units may defer to these stubs")
    return types_rs, glossary if isinstance(glossary, dict) else {}
