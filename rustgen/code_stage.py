"""Stage C — code generation: one call per MTU, in parallel.

Implements each unit against its spec, the shared types, and every sibling's
signatures. The invariants are the acceptance criteria.
"""

from __future__ import annotations

import asyncio

from llm import LLM
from state import Explanation
from rustgen.common import extract_rust, render_spec, unit_block

CODE_PROMPT = """\
Implement ONE behavioral unit of a program in safe, idiomatic Rust.

Rules:
- Implement exactly the signature(s) given for this unit — no changes to them.
- Use only: the shared types below, the sibling signatures below (call them
  freely; their bodies exist elsewhere), `deps::` stubs, and Rust std.
- Functions listed under "SIBLING MODULE FUNCTIONS" in the shared-types
  context (multi-file projects) are implemented elsewhere in this same crate:
  call them directly by name, never re-implement or stub them.
- If the shared types begin with an `// API CONTRACT` comment block, its
  conventions are binding for this unit's code.
- Do NOT redefine shared types, sibling functions, or deps stubs. Private
  helpers are allowed: nest a single-use helper as a `fn` inside its caller;
  a module-level helper gets a short descriptive snake_case name. Never embed
  a unit id in an identifier.
- Visibility: `pub(crate)` on every item (fn, struct, trait, impl-block
  methods, const). Only the program's designated entry point and
  `#[no_mangle]` FFI exports are `pub`.
- Idiom specifics: `match` (never `if`/`else if` chains) for string->variant
  and variant->value dispatch; `&'static str` (not `String`) from fixed-text
  lookups like name/label functions; `writeln!`/`println!` instead of
  `write!`/`print!` with a trailing `\\n`; propagate errors with `?` — never
  `.map_err(|_| ...)` that throws away the underlying error unless the
  contract pins a specific variant; infallible operations return the value
  directly, never a Result that can only be Ok.
- API-SHAPE FIREWALL: C source or C-derived descriptions never dictate API
  shape. A C comparator returning int becomes a function returning
  `std::cmp::Ordering` (or an `Ord`/`PartialOrd` impl); a C print function
  for a type becomes an `impl Display` (callers print with `println!("{{}}",
  x)`); an int-as-bool predicate returns `bool`; a C out-parameter becomes
  the return value.
- OWNERSHIP ORDER: when the C stores an object (appends it to a list,
  installs a pointer) and then keeps mutating it through the stored pointer,
  restructure: populate the object FULLY first, insert it LAST. Never insert
  a `.clone()` and continue mutating the local original — the stored copy
  silently misses every later mutation.
- ARGV: an `args: &[String]` parameter mirrors C argv — args[0] is the
  program path; subcommands and flags start at args[1]. Never match a
  command against args[0] (or `args.first()`/`args.get(0)`).
- Every invariant listed for the unit MUST be honored. Invariants are
  acceptance criteria, not a comment checklist. Cite an invariant in a
  `// invariant:` comment ONLY where the code would look wrong or arbitrary
  without it. If an invariant is satisfied structurally by Rust itself
  (ownership/RAII handles a free, a `Vec` bounds-checks, a type makes a state
  unrepresentable), it is honored — do NOT write a comment about it, and do
  NOT write code (e.g. an empty `Drop` impl) whose only purpose is to have a
  place to put that comment.
- Comment discipline: inside the code block, the ONLY comments allowed are
  `///` docs on items and `// invariant:` citations. Never write comments
  about the spec, the siblings, these instructions, design alternatives, or
  your reasoning — if you must explain a decision, do it OUTSIDE the code
  block.
- Where C idioms were described (zero-byte terminators, manual buffers), use
  the Rust-idiomatic equivalent unless the invariant pins the wire format —
  wire formats must be preserved byte-for-byte.
- If some part is genuinely unimplementable from the information given, write
  `todo!("<what is missing>")` for that part only.
- ALLOCATOR OWNERSHIP: Rust collections (Vec/String/Box) own their buffers —
  NEVER pass a collection's buffer to an allocator function (malloc/realloc/
  free, `*alloc*`-style deps, `Vec::from_raw_parts` over such a pointer):
  that mixes allocators and corrupts memory. Growth/shrink/copy of a
  collection uses its own methods (push/insert/reserve/truncate/clone).
  Descriptions of C realloc/capacity machinery are C idioms — implement the
  *behavior* with collection methods; capacity invariants may be tracked as
  plain numbers if the contract needs them.

SHARED TYPES:
```rust
{types_rs}
```

SIBLING SIGNATURES (implemented elsewhere — do not implement these):
{sibling_sigs}

UNIT TO IMPLEMENT:
{unit}

ITS RUST DESIGN SPEC (the contract — follow every section):
{spec}

Reply with ONLY a ```rust code block containing the implementation.
"""


async def generate_code(llm: LLM, units: list[Explanation],
                        specs: dict[str, dict], types_rs: str,
                        max_tokens: int,
                        extras: dict[str, str] | None = None) -> dict[str, str]:
    """Returns {unit_id: rust_code}. `extras` (common.unit_extras) appends
    caller/C-source context per unit."""
    extras = extras or {}

    def sibling_sigs_for(uid: str) -> str:
        lines = []
        for other_id, spec in specs.items():
            if other_id == uid:
                continue
            lines.extend(f"- {s}" for s in spec.get("signatures", []))
        return "\n".join(lines) or "(none)"

    async def code(u: Explanation) -> tuple[str, str]:
        spec = specs.get(u.id, {})
        reply = await llm.ask(CODE_PROMPT.format(
            types_rs=types_rs,
            sibling_sigs=sibling_sigs_for(u.id),
            unit=unit_block(u, extras.get(u.id, "")),
            spec=render_spec(spec)),
            max_tokens=max_tokens)
        return u.id, extract_rust(reply)

    results = await asyncio.gather(*(code(u) for u in units))
    return dict(results)
