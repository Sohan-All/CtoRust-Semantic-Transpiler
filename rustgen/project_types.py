"""Project Stage T — shared type synthesis for multi-file translation.

The per-file Stage T gives each file its own data model; on a multi-file
project that fractures shared concepts (every file invents its own "build
job"). This stage runs ONCE per project, before any per-file rustgen:

  input:  the SHARED types (declared in headers, used by >1 file — from
          project_index) with their header declaration text, plus each
          using file's MTU descriptions that mention them
  output: shared_types_rs — one canonical Rust definition per shared
          concept + one project error enum — and a project glossary

Per-file Stage T then runs with this as FIXED context: file-local types are
still the file's own business, but shared concepts must use these
definitions verbatim (the same do-not-redefine discipline as sibling
signatures).

The header text is ground truth for FIELDS (the C declaration is exact); the
MTU descriptions are ground truth for MEANING (what the fields are for, the
invariants). Both go in the prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm import LLM
from rustgen.common import extract_rust

PROJECT_TYPES_PROMPT = """\
A multi-file C project is being redesigned in idiomatic Rust, one module per
C file. The types below are SHARED — declared in headers and used across
several files — so they must have ONE canonical Rust definition that every
module builds against.

For each shared C type below, produce an idiomatic, safe Rust definition:
- Owned types (String, Vec<T>), derive Debug/Clone/PartialEq where sensible.
- C idioms map to Rust idioms: char arrays/pointers -> String, {{ptr,len}} or
  flexible arrays -> Vec<T>, int-as-bool -> bool, tagged unions -> enum with
  payloads, C enums -> Rust enums (preserve the VARIANT MEANINGS and, where
  descriptions pin numeric wire values, note them with explicit
  discriminants).
- Intrusive/self-referential structures (parent/child/sibling pointers) are
  redesigned as ownership trees: Vec<Child> or Box, never raw pointers.
- ONE project error enum covering the failure modes the descriptions mention.
  Make it comprehensive — every module of the crate will use this enum and
  ONLY this enum. Never define a success/failure status enum (`Ok`/`Failed`/
  `Success` variants) — that is a C idiom; the error enum plus `Result`
  covers it. The error enum MUST also get a hand-written
  `impl std::fmt::Display` and `impl std::error::Error` (the one exception
  to the no-methods rule below).
- A one-line doc comment per type stating what it is.
- Visibility: `pub(crate)` on every type (struct fields may be `pub`) — the
  crate's only `pub` items are the program entry point and FFI exports.
- No raw pointers, no unsafe, no methods (impls come later, per module).

Begin the rust block with an `// API CONTRACT` comment block stating the
crate-wide conventions every module must follow. Pin at minimum, adapted to
this project's vocabulary:
- Operations on a shared type are `impl` methods on that type, never free
  functions taking it as the first parameter.
- One canonical byte/string type per concept (e.g. hashing takes `&[u8]`;
  callers convert once at the boundary) — name the concrete choices.
- Fallible operations return `Result<T, <the project error enum>>`.
- Method names carry no `<type>_<verb>` C-style prefixes.

Also produce a GLOSSARY mapping each C type name AND each recurring concept
phrase from the descriptions to its Rust type name.

THE SHARED C TYPES (header declarations are the exact fields; the behavioral
descriptions say what they mean):
{types_block}

Reply in exactly this layout:
```rust
<the shared type definitions and the error enum>
```
GLOSSARY:
```json
{{"<C name or concept phrase>": "<RustTypeName>", ...}}
```
"""


def _decl_text(header_src: str, type_name: str) -> str:
    """The declaration block for `type_name` in a header: the typedef/struct/
    enum statement containing it, extracted by brace matching around the
    first occurrence."""
    at = re.search(rf"\b{re.escape(type_name)}\b", header_src)
    if at is None:
        return ""
    # widen to the enclosing statement: back up to the previous ';' or file
    # start, forward through balanced braces to the closing ';'
    start = header_src.rfind(";", 0, at.start()) + 1
    depth = 0
    i = start
    while i < len(header_src):
        ch = header_src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ";" and depth == 0:
            return header_src[start:i + 1].strip()
        i += 1
    return header_src[start:].strip()


def types_block(index, unit_descriptions: dict[str, list[str]]) -> str:
    """Render the shared-types prompt block. `unit_descriptions` maps
    file -> its MTU description strings (from each file's completed MTU
    run); files without a completed run contribute no descriptions —
    honest degradation, the header decl still anchors the fields."""
    parts = []
    for tname, header in sorted(index.shared_types.items()):
        decl = _decl_text((index.c_root / header).read_text(), tname)
        users = sorted(index.type_refs[tname])
        mentions = []
        for f in users:
            for d in unit_descriptions.get(f, []):
                if re.search(rf"\b{re.escape(tname)}\b", d) or _concept_match(tname, d):
                    mentions.append(f"[{f}] {d}")
        block = (f"### {tname} (declared in {header}, used by {', '.join(users)})\n"
                 f"```c\n{decl}\n```")
        if mentions:
            block += "\nBehavioral descriptions mentioning it:\n" + \
                     "\n".join(f"- {m}" for m in mentions[:6])
        parts.append(block)
    return "\n\n".join(parts)


def _concept_match(type_name: str, description: str) -> bool:
    """CamelCase C type names appear in descriptions as spaced concepts
    ("BuildJob" -> "build job")."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", type_name).lower()
    return spaced in description.lower()


async def synthesize_project_types(llm: LLM, index,
                                   unit_descriptions: dict[str, list[str]],
                                   max_tokens: int) -> tuple[str, dict]:
    """(shared_types_rs, project_glossary)."""
    prompt = PROJECT_TYPES_PROMPT.format(
        types_block=types_block(index, unit_descriptions))
    reply = await llm.ask(prompt, max_tokens=max_tokens)
    rust = extract_rust(reply)
    glossary = {}
    m = re.search(r"GLOSSARY:\s*```json\s*(\{.*?\})\s*```", reply, re.DOTALL)
    if m:
        try:
            glossary = json.loads(m.group(1))
        except json.JSONDecodeError:
            glossary = {}
    return rust, glossary
