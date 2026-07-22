"""Stage S — spec generation: one call per MTU, in parallel.

Two modes (cfg.rustgen_spec_mode):

- "thin": signatures + a one-sentence note. Cheap plumbing.
- "rich": a full Rust design spec per MTU — ownership/borrowing decisions,
  error mapping, per-invariant implementation obligations, responsibilities.
  Pins the design decisions codegen and repairs would otherwise make
  independently (and incompatibly — spec/impl drift, cross-MTU duplicates,
  void* leakage were all observed consequences of the thin mode).

Rich mode runs a single global RESPONSIBILITY ASSIGNMENT call first: concerns
that exactly one unit must own (Drop impls, shared helpers, trait impls) are
assigned to specific MTUs, so no two units implement the same item.

In both modes the spec ADDS to the MTU, never replaces it: codegen always
receives the MTU description + invariants alongside — the invariants remain
the acceptance criteria.
"""

from __future__ import annotations

import asyncio

from config import Config
from llm import LLM
from state import Explanation
from rustgen.common import unit_block

RESPONSIBILITY_PROMPT = """\
A C file has been decomposed into the behavioral units below, to be
reimplemented in Rust against the shared types below. Some concerns must be
implemented by EXACTLY ONE unit or the crate will not compile: Drop/Default/
trait impls for a shared type, a helper several units need, a constructor.

List each such single-owner concern and assign it to the one unit whose
behavior it belongs to (resource-release behavior owns Drop; construction
behavior owns the constructor). Only list genuinely shared concerns — a
function obviously private to one unit needs no assignment.

NEVER create a Drop concern for a type whose cleanup is plain RAII (fields
are String/Vec/Box/Option that free themselves) — that describes C's free()
and is already satisfied by Rust; an `impl Drop` exists ONLY for behavior
beyond dropping fields (flushing, closing an external handle, ordering).
Most types need no Drop impl at all.

DO create trait-impl concerns where the file's behavior implies them, each
assigned to the unit owning that behavior: a comparator for a type ->
`impl Ord`/`PartialOrd`; a print/format function for a type ->
`impl std::fmt::Display`; a parse-from-text function -> `impl FromStr`.
These are the idiomatic Rust surface for compare/print/parse behavior.

SHARED TYPES (may begin with an `// API CONTRACT` block — its conventions are binding on every signature you design; may end with a SIBLING MODULE FUNCTIONS list — those are implemented elsewhere in this crate; design signatures that CALL them, never re-specify them):
```rust
{types_rs}
```

UNITS:
{units}

Reply with ONLY a JSON object:
{{"concerns": [{{"concern": "<e.g. 'impl Drop for TreeCache'>", "owner": "<unit id>"}}, ...]}}
"""

RICH_SPEC_PROMPT = """\
You are writing the RUST DESIGN SPEC for one behavioral unit of a program —
the contract a separate implementer will code against. The shared data model
already exists (below); use its types, define no new ones. Pin every design
decision the implementer would otherwise have to guess:

- "signatures": the signature(s). Methods in `impl` blocks where the
  unit operates on a shared type, free functions otherwise; Result<_, the
  error enum> for fallible operations; borrowed parameters (&str, &[u8], &T /
  &mut T) unless the behavior requires ownership. If the unit's data is an
  opaque payload the program never interprets, prefer a generic parameter over
  raw byte buffers. Visibility is `pub(crate)` — the crate's only `pub` items
  are the program entry point and FFI exports. Infallible operations return
  the value directly, never a Result that can only be Ok. C shapes never
  survive into signatures: comparator-returning-int -> `Ordering` (or an
  `Ord`/`PartialOrd` impl), print-function-for-a-type -> `impl Display`,
  int-as-bool -> `bool`, out-parameter -> return value. An `args: &[String]`
  parameter mirrors C argv: args[0] is the program path; real arguments
  (subcommands, flags) start at args[1].
- "ownership": one entry per parameter/return worth deciding: who owns it,
  borrowed or moved, lifetimes of returned references.
- "error_mapping": for each failure the unit's invariants describe, which
  error variant it produces and what state is left behind.
- "invariant_obligations": for EACH invariant, the concrete Rust obligation
  that satisfies it (e.g. "capacity doubles" -> "grow via checked_mul(2);
  on overflow return Err(...)"). Keep 1:1 with the invariants.
- "idioms": std APIs/patterns to use (iterator over index loop, entry API,
  slice::from_raw_parts NOT allowed here — no unsafe outside FFI).
- "owns": concerns assigned to this unit (from the list given), which it MUST
  implement.
- "must_not_implement": concerns owned by OTHER units — never emit these
  items, call them instead.
- "behavior_note": one sentence the implementer must not forget.

OUTPUT REACHES OUTPUT: when the unit's behavior is to display/print/report,
the design must guarantee emission — either the signature prints directly
(returning () or Result<(), _>), or it returns a displayable value AND the
"behavior_note" names which caller (from the unit's CALLED BY context) is
responsible for printing it. Never design a report/display unit whose
computed output has no printing consumer — a report nobody prints is wrong
even though it compiles.

SHARED TYPES (may begin with an `// API CONTRACT` block — its conventions are binding on every signature you design; may end with a SIBLING MODULE FUNCTIONS list — those are implemented elsewhere in this crate; design signatures that CALL them, never re-specify them):
```rust
{types_rs}
```

GLOSSARY: {glossary}

CONCERN ASSIGNMENTS (single-owner items across the whole file):
{assignments}

UNIT TO SPECIFY:
{unit}

Reply with ONLY a JSON object with exactly the keys:
{{"signatures": [...], "ownership": [...], "error_mapping": [...],
  "invariant_obligations": [...], "idioms": [...], "owns": [...],
  "must_not_implement": [...], "behavior_note": "..."}}
"""

SPEC_PROMPT = """\
You are writing the Rust SIGNATURE(S) for one behavioral unit of a program.
The shared data model already exists (below) — use its types; do not define
new types. Choose idiomatic Rust: methods in `impl` blocks where the unit
operates on one of the shared types, free functions otherwise; Result<_,
<the error enum>> for fallible operations; borrowed parameters (&str, &[u8],
&T / &mut T) where the behavior doesn't require ownership. Visibility is
`pub(crate)` — the crate's only `pub` items are the program entry point and
FFI exports. C shapes never survive into signatures: comparator-returning-int
-> `Ordering`, print-function-for-a-type -> `impl Display`, int-as-bool ->
`bool`, out-parameter -> return value. Display/print/report behavior must
guarantee emission: print directly, or note which caller prints the returned
value — never a report nobody prints.

SHARED TYPES (may begin with an `// API CONTRACT` block — its conventions are binding on every signature you design; may end with a SIBLING MODULE FUNCTIONS list — those are implemented elsewhere in this crate; design signatures that CALL them, never re-specify them):
```rust
{types_rs}
```

GLOSSARY: {glossary}

UNIT TO SPECIFY:
{unit}

Reply with ONLY a JSON object:
{{"signatures": ["<one signature per public function, e.g. 'impl TreeCache {{ pub fn get(&self, path: &str) -> Option<&TreeCache> }}' or 'pub fn parse(...) -> Result<X, Error>'>", ...],
  "behavior_note": "<one sentence: what the implementer must not forget>"}}
"""


async def generate_specs(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, cfg: Config,
                         extras: dict[str, str] | None = None) -> dict[str, dict]:
    """Returns {unit_id: spec dict}. Every spec has at least "signatures" and
    "behavior_note"; rich mode adds the design-decision fields. `extras`
    (common.unit_extras) appends caller/C-source context per unit."""
    if cfg.rustgen_spec_mode == "rich":
        return await _generate_rich(llm, units, types_rs, glossary, cfg,
                                    extras or {})
    return await _generate_thin(llm, units, types_rs, glossary,
                                cfg.rustgen_spec_max_tokens, extras or {})


async def _generate_thin(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, max_tokens: int,
                         extras: dict[str, str]) -> dict[str, dict]:
    async def spec(u: Explanation) -> tuple[str, dict]:
        resp = await llm.ask_json(SPEC_PROMPT.format(
            types_rs=types_rs, glossary=glossary,
            unit=unit_block(u, extras.get(u.id, ""))),
            max_tokens=max_tokens)
        if not isinstance(resp, dict):
            resp = {}
        resp.setdefault("signatures", [])
        resp.setdefault("behavior_note", "")
        return u.id, resp

    results = await asyncio.gather(*(spec(u) for u in units))
    return dict(results)


async def _generate_rich(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, cfg: Config,
                         extras: dict[str, str]) -> dict[str, dict]:
    # global pass: single-owner concerns, so no two units emit the same item
    resp = await llm.ask_json(RESPONSIBILITY_PROMPT.format(
        types_rs=types_rs,
        units="\n\n".join(unit_block(u) for u in units)),
        max_tokens=2000)
    concerns = resp.get("concerns", []) if isinstance(resp, dict) else []
    concerns = [c for c in concerns
                if isinstance(c, dict) and c.get("concern") and c.get("owner")]

    def assignments_for(uid: str) -> str:
        owns = [c["concern"] for c in concerns if c["owner"] == uid]
        others = [f"{c['concern']} (owned by {c['owner']})"
                  for c in concerns if c["owner"] != uid]
        lines = []
        lines.append("This unit OWNS: " + ("; ".join(owns) if owns else "(nothing shared)"))
        if others:
            lines.append("Owned by OTHER units (never implement these): " + "; ".join(others))
        return "\n".join(lines)

    async def spec(u: Explanation) -> tuple[str, dict]:
        r = await llm.ask_json(RICH_SPEC_PROMPT.format(
            types_rs=types_rs, glossary=glossary,
            assignments=assignments_for(u.id),
            unit=unit_block(u, extras.get(u.id, ""))),
            max_tokens=cfg.rustgen_rich_spec_max_tokens)
        if not isinstance(r, dict):
            r = {}
        r.setdefault("signatures", [])
        r.setdefault("behavior_note", "")
        # authoritative assignment beats whatever the model echoed back
        r["owns"] = [c["concern"] for c in concerns if c["owner"] == u.id]
        r["must_not_implement"] = [c["concern"] for c in concerns if c["owner"] != u.id]
        return u.id, r

    results = await asyncio.gather(*(spec(u) for u in units))
    return dict(results)
