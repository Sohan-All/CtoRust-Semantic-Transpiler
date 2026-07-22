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
from rustgen.common import extract_rust, unit_block

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
    reply = await llm.ask(prompt, max_tokens=max_tokens)
    # first fence is the Rust; glossary is the json fence after "GLOSSARY:"
    glossary: dict = {}
    if "GLOSSARY:" in reply:
        rust_part, gloss_part = reply.split("GLOSSARY:", 1)
        try:
            glossary = extract_json(gloss_part)
        except ValueError:
            glossary = {}
    else:
        rust_part = reply
    types_rs = extract_rust(rust_part)
    return types_rs, glossary if isinstance(glossary, dict) else {}
