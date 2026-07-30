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


from config import Config
from llm import LLM
from state import Explanation
from rustgen.common import gather_units, unit_block

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
{symbol_map_bullet}
OUTPUT REACHES OUTPUT: when the unit's behavior is to display/print/report,
the design must guarantee emission — either the signature prints directly
(returning () or Result<(), _>), or it returns a displayable value AND the
"behavior_note" names which caller (from the unit's CALLED BY context) is
responsible for printing it. Never design a report/display unit whose
computed output has no printing consumer — a report nobody prints is wrong
even though it compiles.

EXIT STATUS IS OUTPUT: the "C shapes never survive" rule above has ONE
exception. A C function whose return value reaches the process exit status —
`main`, whatever `main` returns, and anything feeding them — has that value as
OBSERVABLE BEHAVIOR, not as an internal success flag. If such a function can
return more than one distinct non-zero value (e.g. `return 1` for a usage error
and `return 2` for a script failure), each distinct code MUST be recoverable
from what your signature returns: either return the status directly, or give
the error enum a variant per code and say in "error_mapping" which variant is
which number. `Result<(), E>` collapsed to `Err(_) => 1` at the entry point is
WRONG — it silently turns exit code 2 into 1, which no compiler and no type
error will ever catch. Read the unit's C for the set of codes it can produce;
do not assume 0/1.

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
  "must_not_implement": [...], "behavior_note": "..."{symbol_map_key}}}
"""

# The symbol_map contract, held separately so cfg.rustgen_symbol_map can drop
# it from the prompt entirely for an ablation run (an empty bullet would still
# leave the model primed by the key name in the output schema).
SYMBOL_MAP_BULLET = """\
- "symbol_map": one entry for EVERY C function listed as defined in this unit,
  saying where it went. Renaming and reshaping into idiomatic Rust is expected
  and fine — this only records WHERE, so nothing is lost silently:
    {"c": "scheduler_spawn", "rust": "Scheduler::spawn"}
    {"c": "scheduler_free", "rust": null, "reason": "drop_glue"}
    {"c": "text_equals", "rust": null, "reason": "stdlib_equivalent"}
    {"c": "scheduler_print_task", "rust": null, "reason": "inlined_into:Scheduler::print_report"}
  `reason` is required when "rust" is null and must be one of: drop_glue (the
  C frees/destroys and Rust ownership handles it), stdlib_equivalent (a std
  method replaces it), inlined_into:<rust path> (its body is folded into that
  function — which then INHERITS its output text and its calls), dead_code
  (never reached). Anything this unit's C printed must still be printed by
  whatever absorbed it.
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

ONE exception to "C shapes never survive": a return value that reaches the
process exit status (`main`, what `main` returns, anything feeding them) is
observable behavior. If the C can return more than one distinct non-zero code,
every code must be recoverable from your signature — return the status, or use
one error variant per code. `Result<(), E>` collapsed to `Err(_) => 1` turns
exit code 2 into 1, and nothing in the type system catches it.

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


def _str_array() -> dict:
    return {"type": "array", "items": {"type": "string"}}


# Schemas for constrained decoding. They mirror the "reply with exactly these
# keys" instruction in each prompt — the prompt still carries the MEANING of
# each field, the schema only makes the shape unfalsifiable. Kept permissive
# (no additionalProperties:false) so a model volunteering an extra key is not
# a hard failure; the stages already ignore unknown keys.
RESPONSIBILITY_SCHEMA = {
    "type": "object",
    "properties": {"concerns": {"type": "array", "items": {
        "type": "object",
        "properties": {"concern": {"type": "string"}, "owner": {"type": "string"}},
        "required": ["concern", "owner"]}}},
    "required": ["concerns"],
}

THIN_SPEC_SCHEMA = {
    "type": "object",
    "properties": {"signatures": _str_array(), "behavior_note": {"type": "string"}},
    "required": ["signatures", "behavior_note"],
}


def rich_spec_schema(with_symbol_map: bool) -> dict:
    props = {
        "signatures": _str_array(), "ownership": _str_array(),
        "error_mapping": _str_array(), "invariant_obligations": _str_array(),
        "idioms": _str_array(), "owns": _str_array(),
        "must_not_implement": _str_array(), "behavior_note": {"type": "string"},
    }
    required = ["signatures", "behavior_note"]
    if with_symbol_map:
        props["symbol_map"] = {"type": "array", "items": {
            "type": "object",
            "properties": {"c": {"type": "string"},
                           "rust": {"type": ["string", "null"]},
                           "reason": {"type": "string"}},
            "required": ["c"]}}
        required.append("symbol_map")
    return {"type": "object", "properties": props, "required": required}


async def generate_specs(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, cfg: Config,
                         extras: dict[str, str] | None = None,
                         skip: set[str] | None = None,
                         failures: list[dict] | None = None,
                         on_result=None) -> dict[str, dict]:
    """Returns {unit_id: spec dict} for the units actually generated. Every spec
    has at least "signatures" and "behavior_note"; rich mode adds the
    design-decision fields. `extras` (common.unit_extras) appends caller/C-source
    context per unit.

    `skip` names units whose specs are already persisted, so a resume redraws
    only what is missing. They stay in `units` regardless: rich mode's
    responsibility assignment is a single global pass over the whole file, and
    running it on a subset would hand the remaining units a different ownership
    split from the one the persisted specs were written against.

    `failures` and `on_result` are passed through to common.gather_units — a
    unit that raises is dropped and recorded rather than killing the stage.
    """
    if cfg.rustgen_spec_mode == "rich":
        return await _generate_rich(llm, units, types_rs, glossary, cfg,
                                    extras or {}, skip or set(), failures,
                                    on_result)
    return await _generate_thin(llm, units, types_rs, glossary,
                                cfg.rustgen_spec_max_tokens, extras or {},
                                skip or set(), failures, on_result)


async def _generate_thin(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, max_tokens: int,
                         extras: dict[str, str], skip: set[str],
                         failures: list[dict] | None,
                         on_result) -> dict[str, dict]:
    async def spec(u: Explanation) -> tuple[str, dict]:
        resp = await llm.ask_json(SPEC_PROMPT.format(
            types_rs=types_rs, glossary=glossary,
            unit=unit_block(u, extras.get(u.id, ""))),
            max_tokens=max_tokens, schema=THIN_SPEC_SCHEMA)
        if not isinstance(resp, dict):
            resp = {}
        resp.setdefault("signatures", [])
        resp.setdefault("behavior_note", "")
        resp.setdefault("symbol_map", [])
        return u.id, resp

    return await gather_units("spec", [u for u in units if u.id not in skip],
                              spec, failures, on_result)


async def _generate_rich(llm: LLM, units: list[Explanation], types_rs: str,
                         glossary: dict, cfg: Config,
                         extras: dict[str, str], skip: set[str],
                         failures: list[dict] | None,
                         on_result) -> dict[str, dict]:
    # global pass: single-owner concerns, so no two units emit the same item.
    # Guarded because it runs before any unit does: an unparseable reply here
    # used to abort the whole project at the start of its most expensive stage.
    # Losing the assignment costs cross-unit coordination, not correctness —
    # every unit still gets a spec — so it degrades rather than aborts.
    try:
        resp = await llm.ask_json(RESPONSIBILITY_PROMPT.format(
            types_rs=types_rs,
            units="\n\n".join(unit_block(u) for u in units)),
            max_tokens=2000, schema=RESPONSIBILITY_SCHEMA)
    except Exception as e:
        print(f"[spec] responsibility assignment FAILED "
              f"({type(e).__name__}: {e}) — proceeding unassigned, "
              f"run marked degraded")
        if failures is not None:
            failures.append({"stage": "spec", "unit": "(responsibility pass)",
                             "error": f"{type(e).__name__}: {e}"})
        resp = None
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

    want_symbols = cfg.rustgen_symbol_map

    async def spec(u: Explanation) -> tuple[str, dict]:
        r = await llm.ask_json(RICH_SPEC_PROMPT.format(
            types_rs=types_rs, glossary=glossary,
            assignments=assignments_for(u.id),
            symbol_map_bullet=SYMBOL_MAP_BULLET if want_symbols else "",
            symbol_map_key=', "symbol_map": [...]' if want_symbols else "",
            unit=unit_block(u, extras.get(u.id, ""))),
            max_tokens=cfg.rustgen_rich_spec_max_tokens,
            schema=rich_spec_schema(want_symbols))
        if not isinstance(r, dict):
            r = {}
        r.setdefault("signatures", [])
        r.setdefault("behavior_note", "")
        r.setdefault("symbol_map", [])
        # authoritative assignment beats whatever the model echoed back
        r["owns"] = [c["concern"] for c in concerns if c["owner"] == u.id]
        r["must_not_implement"] = [c["concern"] for c in concerns if c["owner"] != u.id]
        return u.id, r

    return await gather_units("spec", [u for u in units if u.id not in skip],
                              spec, failures, on_result)
