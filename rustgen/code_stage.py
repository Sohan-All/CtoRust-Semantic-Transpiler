"""Stage C — code generation: one call per MTU, in parallel.

Implements each unit against its spec, the shared types, and every sibling's
signatures. The invariants are the acceptance criteria.
"""

from __future__ import annotations


from llm import LLM
from state import Explanation
from rustgen.common import (extract_rust, gather_units, illegal_stubs,
                            render_spec, unbalanced_delimiters, unit_block)

# Regeneration attempts for a section that fails a pre-flight check (delimiters
# that do not balance, or an unexplained `todo!()`). Two: both failures are
# single-sample accidents that a fresh draw almost always fixes, and each extra
# call costs a full section generation.
CODE_RETRIES = 2

STUB_RETRY_NOTE = """\
Your previous reply left work unfinished: {problem}. A `todo!()` type-checks,
so no compiler error will ever flag it — at RUN TIME it panics and takes the
whole program down. Nothing downstream fills these in: this unit is the only
thing that implements its own signatures. Write every body now, using the
shared types, the sibling signatures above, and Rust std. Only if a capability
is genuinely impossible with what you have been given may a stub remain, and
then its message MUST be a full sentence naming what is missing and why — e.g.
`todo!("no sibling provides tag storage and Task has no field for it")`. A
placeholder, or a bare function name in angle brackets, is not an
explanation."""

PARSE_RETRY_NOTE = """\
Your previous reply did not parse: {problem}. Emit the COMPLETE section with
every block closed."""

CODE_PROMPT = """\
Implement ONE behavioral unit of a program in safe, idiomatic Rust.

The rules below come in three parts. Part 1 always applies. Part 2 settles
which rule wins when two of them could apply. Part 3 is a list of conditions —
check each against this unit, and apply only the ones whose IF matches.

=== PART 1: ALWAYS ===

- Implement exactly the signature(s) given for this unit — no changes to them.
- Use only: the shared types below, the sibling signatures below (call them
  freely; their bodies exist elsewhere), `deps::` stubs, and Rust std.
- Do NOT redefine shared types, sibling functions, or deps stubs. Private
  helpers are allowed: nest a single-use helper as a `fn` inside its caller;
  a module-level helper gets a short descriptive snake_case name. Never embed
  a unit id in an identifier.
- A callback the caller supplies is a PARAMETER. If your signature takes a
  generic bounded by `Fn`/`FnMut`, call that parameter directly — `pred(item)`.
  Never route it through a `deps::` stub, never define a free function to stand
  in for it, and never invent a body for it: each caller passes different
  behavior, so any fixed implementation is wrong and compiles anyway.
- Visibility: `pub(crate)` on every item (fn, struct, trait, impl-block
  methods, const). Only the program's designated entry point and
  `#[no_mangle]` FFI exports are `pub`.
- Every invariant listed for the unit MUST be honored — invariants are
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
- If some part is genuinely unimplementable from the information given,
  replace only that part with a `todo!` whose message is a full sentence
  naming what is missing and why — e.g. `todo!("no sibling provides tag
  storage and Task has no field for it")`. A placeholder, or a bare function
  name in angle brackets, is not an explanation.

=== PART 2: PRECEDENCE ===

When two rules could apply to the same code, the earlier one wins:

  1. ORPHAN RULE — a violation does not compile, so nothing else can matter.
  2. REPRESENTATION vs BEHAVIOR — decides which invariants are real behavior.
     It OVERRIDES "every invariant MUST be honored" in Part 1: an invariant
     that merely restates C's in-memory representation is NOT honored by
     reproducing that representation.
  3. Everything else.

=== PART 3: CONDITIONAL RULES ===

Three of these are marked OBSERVABLE. A mistake there changes what the program
prints or returns, and no compiler will flag it.

--- IF the shared-types context lists "SIBLING MODULE FUNCTIONS"
    THEN those functions are implemented elsewhere in this same crate: call
    them directly by name, never re-implement or stub them.

--- IF the shared types begin with an `// API CONTRACT` comment block
    THEN its conventions are binding for this unit's code.

--- IF an invariant describes how C stored the data in memory
    THEN [OBSERVABLE] REPRESENTATION vs BEHAVIOR applies. C's in-memory
    representation is not observable behavior. A Rust `String`/`Vec` carries
    its own length, so a terminator is never needed and a literal 0 byte
    inside one is a BUG that shows up in the program's output. Never write
    `.push('\\0')`, `\\0` in a string literal, or reserve "+1 for the
    terminator".
      "must be terminated by a zero byte"   -> just build the String; drop it
      C `*end = '\\0'` (truncate in place)   -> `s.truncate(n)` / `&s[..n]`
      C `strlen(buf)`                       -> `s.len()` (do NOT add or subtract 1)
      C `char buf[N]` + manual copies        -> `String`/`Vec<u8>`, no capacity
                                               bookkeeping
    The same applies to how C arranged its COLLECTIONS, and there the mistake
    is louder: when C keeps two views of the same elements (a heap array and a
    linear list, a list plus a count, a map plus a parallel index) the shared
    types here have usually collapsed them into ONE Rust collection. An
    invariant like "must be present in both the queue and the task list" then
    describes C's bookkeeping, not behavior — it is satisfied by a single
    insert into the one collection that exists. Inserting twice to satisfy it
    literally duplicates every element and every count the program prints.
    Before writing a second insert, remove or push, check the shared types for
    whether the two containers the invariant names are actually one field.
    Satisfy such an invariant by producing the same OBSERVABLE bytes — the
    same printed text, the same counts and lengths reported to the user —
    never by reproducing C's storage arrangement. The single exception is a
    genuine external wire format (a file or socket the program reads/writes,
    where a real reader depends on the byte layout); reproduce that
    byte-for-byte, and only that.

--- IF this unit mirrors a C `printf`/`fprintf`/`puts`
    THEN [OBSERVABLE] TRAILING NEWLINE applies — exactly one newline total,
    counted, not guessed. The `ln` in `println!`/`eprintln!`/`writeln!`
    appends a newline of its own, so `println!("done\\n")` emits TWO and every
    byte-comparison against the C fails. Count the trailing `\\n` in C's
    format string and keep the total the same:
      C ends with one `\\n`  -> `println!` with the `\\n` REMOVED (or `print!`
                               keeping it — pick one, never both)
      C ends with no `\\n`   -> `print!` / `eprint!`
      C ends with two `\\n`  -> `println!` with ONE `\\n` left in the string
    C `puts(s)` appends its own newline -> `println!("{{}}", s)`. This applies
    to every line of a multi-line banner or usage text: reproduce the interior
    newlines exactly and add none at the end. Same stream as the C —
    `fprintf(stderr, ...)` -> `eprint!`/`eprintln!`, `printf` -> stdout.

--- IF this unit's C returns a value that becomes the process exit status
    THEN [OBSERVABLE] the specific numbers are output. If the C can produce
    more than one non-zero code (`return 1` here, `return 2` there), every one
    must survive to the entry point — an `Err(_) => 1` catch-all that reports
    2 as 1 is a behavior change no compiler will flag. Honor whatever the
    spec's signature and error_mapping chose for carrying the code.

--- IF the spec asks for a trait impl
    THEN ORPHAN RULE applies: a trait impl is only possible when this crate
    owns the trait or the type. `impl FromStr for i32`,
    `impl Display for String`, `impl Ord for u32` do not compile — no body, no
    workaround, and one of them fails the whole crate. For a trait impl on a
    primitive or another std type, write a plain function instead
    (`parse_priority(s: &str) -> Result<i32, E>`) and implement nothing else.
    Never emit such an `impl` with a `todo!()` inside "explaining" that it is
    impossible — delete the block.

--- IF the C stores an object (appends it to a list, installs a pointer) and
    then keeps mutating it through the stored pointer
    THEN OWNERSHIP ORDER applies: populate the object FULLY first, insert it
    LAST. Never insert a `.clone()` and continue mutating the local original —
    the stored copy silently misses every later mutation.

--- IF this unit takes an `args: &[String]` parameter
    THEN it mirrors C argv — args[0] is the program path; subcommands and
    flags start at args[1]. Never match a command against args[0] (or
    `args.first()`/`args.get(0)`).

--- IF the C source or a C-derived description suggests an API shape
    THEN the API-SHAPE FIREWALL applies: it never dictates API shape. A C
    comparator returning int becomes a function returning `std::cmp::Ordering`
    (or an `Ord`/`PartialOrd` impl); a C print function for a type becomes an
    `impl Display` (callers print with `println!("{{}}", x)`); an int-as-bool
    predicate returns `bool`; a C out-parameter becomes the return value.

--- IF the unit describes other C idioms (manual buffers, index bookkeeping)
    THEN likewise implement the behavior with Rust collections.

--- IF allocator functions are in play (malloc/realloc/free, `*alloc*`-style
    deps, `Vec::from_raw_parts`)
    THEN ALLOCATOR OWNERSHIP applies: Rust collections (Vec/String/Box) own
    their buffers — NEVER pass a collection's buffer to an allocator function;
    that mixes allocators and corrupts memory. Growth/shrink/copy of a
    collection uses its own methods (push/insert/reserve/truncate/clone).
    Descriptions of C realloc/capacity machinery are C idioms — implement the
    *behavior* with collection methods; capacity invariants may be tracked as
    plain numbers if the contract needs them.

--- IF this unit dispatches string->variant or variant->value
    THEN use `match`, never `if`/`else if` chains.

--- IF this unit returns fixed text (a name/label lookup)
    THEN return `&'static str`, not `String`.

--- IF this unit can fail
    THEN propagate errors with `?` — never `.map_err(|_| ...)` that throws
    away the underlying error unless the contract pins a specific variant.

--- IF an operation cannot fail
    THEN return the value directly, never a Result that can only be Ok.

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
                        extras: dict[str, str] | None = None,
                        skip: set[str] | None = None,
                        failures: list[dict] | None = None,
                        on_result=None) -> dict[str, str]:
    """Returns {unit_id: rust_code} for the units actually generated. `extras`
    (common.unit_extras) appends caller/C-source context per unit.

    `skip` names units whose code is already persisted, so a resume redraws only
    what is missing; sibling signatures still come from the full `specs` map, so
    a skipped unit is as visible to its siblings as a freshly drawn one.

    `failures` and `on_result` are passed through to common.gather_units.
    """
    extras = extras or {}
    skip = skip or set()

    def sibling_sigs_for(uid: str) -> str:
        lines = []
        for other_id, spec in specs.items():
            if other_id == uid:
                continue
            lines.extend(f"- {s}" for s in spec.get("signatures", []))
        return "\n".join(lines) or "(none)"

    async def code(u: Explanation) -> tuple[str, str]:
        spec = specs.get(u.id, {})
        prompt = CODE_PROMPT.format(
            types_rs=types_rs,
            sibling_sigs=sibling_sigs_for(u.id),
            unit=unit_block(u, extras.get(u.id, "")),
            spec=render_spec(spec))
        # Two pre-flight checks, both regenerating the section on failure.
        #
        # Delimiters that do not balance make the section a PARSE error, and
        # rustc reports one of those for the whole crate however much else is
        # wrong — so the compile loop reads "1 error", repairs, still reads
        # "1 error", calls that no improvement and reverts, forever. One
        # observed run sat at `final: 1` for five rounds and never built.
        #
        # An unexplained `todo!()` is the mirror image: it type-checks, so the
        # compile loop sees a CLEAN crate and reports success while the unit's
        # behaviour is a runtime panic. The compile loop has a repair pass for
        # these, but it is strictly cheaper and more reliable to redraw the
        # section here, while the unit's own spec and call graph are still the
        # prompt, than to reconstruct that context later.
        base_prompt = prompt
        rust, problem = "", ""
        for attempt in range(CODE_RETRIES + 1):
            reply = await llm.ask(prompt, max_tokens=max_tokens)
            rust = extract_rust(reply)
            problem = unbalanced_delimiters(rust)
            note = PARSE_RETRY_NOTE
            if not problem:
                problem = illegal_stubs(rust)
                note = STUB_RETRY_NOTE
            if not problem:
                return u.id, rust
            if attempt < CODE_RETRIES:
                prompt = base_prompt + "\n\n" + note.format(problem=problem)
        # exhausted: hand back the last attempt rather than nothing, and say so
        print(f"[code] {u.id}: {problem} — still present after "
              f"{CODE_RETRIES} retries")
        return u.id, rust

    return await gather_units("code", [u for u in units if u.id not in skip],
                              code, failures, on_result)
