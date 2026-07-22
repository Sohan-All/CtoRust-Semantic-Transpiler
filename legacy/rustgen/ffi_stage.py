"""FFI shim stage — the cando2 contract.

Equivalence testing binds ONLY at the file's public C functions (static
internals are decomposition, not contract — see SPEC.md). For each public
function, generate a `#[no_mangle] pub unsafe extern "C" fn` with the exact C
symbol name and ABI, adapting raw C-shaped arguments onto the idiomatic Rust
underneath. The crate builds as a cdylib so cando2 can record against the C
.so and replay against ours.

The public-function list and C signatures come from re-chunking the C source —
deterministic, no LLM, no re-run of the MTU pipeline needed.
"""

from __future__ import annotations

import re

from chunker import chunk
from llm import LLM
from rustgen.common import extract_rust

FFI_PROMPT = """\
Write ONE C-ABI export shim for a Rust library that reimplements a C file.
Produce a single `#[no_mangle] pub unsafe extern "C" fn` with EXACTLY the same
symbol name, arity, and C-compatible parameter/return types as the C function
below (raw pointers; core::ffi::c_int/c_char etc. via fully-qualified paths;
usize for size_t).
{abi_block}

The shim: convert the raw inputs into the idiomatic types (std::ffi::CStr for
C strings, std::slice::from_raw_parts for ptr+len pairs), call the
corresponding idiomatic function (signatures below), convert the result back
(0/-1 style return codes where the C returned int; out-parameters written
through their pointers). Null-pointer checks on every pointer argument,
returning the C convention's error value (or doing nothing for void). Opaque C
handles (pointers to this file's own structs) are Box-allocated idiomatic
values: constructors Box::into_raw, destructors Box::from_raw, accessors
&*ptr / &mut *ptr. Raw pointers and unsafe are allowed HERE ONLY; do not
re-implement any behavior. Use fully-qualified paths (core::ffi::…,
std::ffi::…) instead of `use` statements. The shim will live inside `pub mod
ffi {{ use super::*; … }}`.

SHARED IDIOMATIC TYPES:
```rust
{types_rs}
```

IDIOMATIC FUNCTION SIGNATURES (implemented elsewhere — call these):
{idiomatic_sigs}

C FUNCTION TO EXPORT:
{c_signature}

Reply with ONLY a ```rust code block containing this one shim function.
"""

ABI_BLOCK = """
ABI AUTHORITY — this signature was derived by bindgen from the C headers and
is what the equivalence harness will call. Your shim MUST have exactly this
arity and these ABI types (you may rename parameters). Where it names a C
struct type this crate doesn't define, declare that pointer as
`*mut core::ffi::c_void` / `*const core::ffi::c_void` (same ABI). Keep
`Option<unsafe extern "C" fn(...)>` callback parameters with this exact shape
(struct pointers inside them become c_void pointers too); if the idiomatic
layer cannot accept the callback, the shim may ignore it only when the C
semantics allow a default, otherwise return the C error value.

    {abi_sig}
"""

# ---------------------------------------------------------------- boundary --
# The lib_swap differential proved function-signature ABI is not sufficient:
# C code embeds/reads the boundary STRUCTS directly, so persistent state must
# live in C-layout memory. Design decision (July 10): the idiomatic core stays
# untouched; C layout is confined to the FFI layer via bindgen mirror structs
# (rustgen/layout_stage.py) + per-struct conversion functions + a sync-in /
# call-core / sync-out shim discipline.

ALLOC_PRELUDE = """\
// C-allocator helpers: any buffer C retains must come from malloc — C frees
// it with free(). Rust-side allocation must never leak into C-visible memory.
extern "C" {
    #[link_name = "malloc"]
    pub fn c_abi_malloc(n: usize) -> *mut core::ffi::c_void;
    #[link_name = "realloc"]
    pub fn c_abi_realloc(p: *mut core::ffi::c_void, n: usize) -> *mut core::ffi::c_void;
    #[link_name = "free"]
    pub fn c_abi_free(p: *mut core::ffi::c_void);
}"""

CONVERSION_PROMPT = """\
A Rust crate reimplements ONE C file idiomatically, but untranslated C code
still accesses this C struct's memory directly (embeds it by value, reads its
fields). Persistent state therefore lives in the C layout; the FFI shims
convert at the boundary. Write the three conversion functions:

pub unsafe fn {c_name}_sync_in(c: *const c_abi::{c_name}) -> <IdiomaticType>
pub unsafe fn {c_name}_sync_out(v: &<IdiomaticType>, c: *mut c_abi::{c_name})
pub unsafe fn {c_name}_sync_init(v: &<IdiomaticType>, c: *mut c_abi::{c_name})

Rules:
- First line of your reply: `TYPE: <IdiomaticTypeName>` naming the type from
  SHARED IDIOMATIC TYPES that represents this C struct. If none does, reply
  with the single line `OPAQUE` and nothing else.
- sync_in reads a VALID C struct into an owned idiomatic value; it must NOT
  take ownership of or free any C memory.
- sync_out UPDATES a valid C struct in place: buffers C retains are
  (re)allocated with `c_abi_malloc`/`c_abi_realloc` (declared in this module;
  C frees them with free()), lengths/capacities updated. C fields with no
  idiomatic counterpart (function pointers, flags the Rust type doesn't
  model) must be PRESERVED, never zeroed.
- sync_init writes into a struct whose memory may be UNINITIALIZED GARBAGE
  (C initializer semantics: `git_x v; x_init(&v);`). It must READ NOTHING
  from the target — no realloc/free of old pointers, no preserving old
  fields — every field is written fresh (fresh c_abi_malloc buffers;
  no-counterpart fields set to their C zero-values).
- Never store a Rust-only value (Vec/String/Box) into C-visible memory.
- Fully-qualified paths (core::ffi::…); no `use` statements. The functions
  live inside `pub mod ffi` next to `mod c_abi`.

C MIRROR STRUCT (bindgen authority — field layout is exact):
```rust
{mirror}
```

SHARED IDIOMATIC TYPES:
```rust
{types_rs}
```

Reply with the TYPE line, then ONLY a ```rust code block with the three
functions.
"""

BOUNDARY_BLOCK = """
EXACT SIGNATURE (bindgen, from the C headers — use VERBATIM as your
`#[no_mangle] pub unsafe extern "C" fn`; parameter names may change, types
may not; `::std::os::raw::*` paths are fine; the mirror struct types are in
scope as `c_abi::<name>`, qualify them so):

    {bindgen_sig}

BOUNDARY DATA CONTRACT — C code allocates/embeds/reads the mirror structs
(mod c_abi, below) directly; persistent state MUST live in the C struct
across calls:
- sync-in: read the C struct / arguments into idiomatic values (use the
  CONVERSION HELPERS below when one exists for the type),
- call the idiomatic function,
- sync-out: write results back into the C struct BEFORE returning (again via
  the helpers). Any buffer C retains must come from `c_abi_malloc`; never
  leave a Rust-only value (Vec/String/Box) in C-visible memory; never clobber
  C fields that have no idiomatic counterpart. No statics holding state
  between calls.
- INITIALIZER SEMANTICS: when the C function INITIALIZES the struct (init /
  the target of a dup / an out-struct C passes in uninitialized), the
  incoming memory is GARBAGE — do NOT sync-in from it and do NOT call
  sync_out on it (sync_out reads old fields); use `<name>_sync_init`, which
  writes every field fresh. Destructor-like functions (dispose/free/clear)
  must leave the struct in the C-idiomatic emptied state (buffers freed with
  c_abi_free, pointers nulled, lengths zeroed) — sync-in first, then write
  the emptied state directly.

MIRROR STRUCTS (bindgen authority):
```rust
{mirror}
```

CONVERSION HELPERS (already defined in this module — call them, do not
redefine):
{conversion_sigs}
"""


def public_functions(source: str, split_over: int) -> list[tuple[str, str]]:
    """[(name, c_signature)] for non-static function definitions, file order."""
    seen: dict[str, str] = {}
    for b in chunk(source, split_over).blocks:
        if b.function and b.is_public and b.function not in seen:
            seen[b.function] = b.c_signature
    return list(seen.items())


async def generate_conversions(llm: LLM, structs: dict[str, str],
                               types_rs: str, max_tokens: int
                               ) -> dict[str, str]:
    """{c_struct_name: conversion fns code} for each boundary struct that has
    an idiomatic counterpart; structs the model declares OPAQUE are omitted
    (their shims fall back to opaque-pointer handling). Reply shape is
    re-validated mechanically (both fns present) with one retry."""
    import asyncio

    from rustgen.surface import fn_decls

    async def convert(c_name: str, mirror: str) -> tuple[str, str]:
        prompt = CONVERSION_PROMPT.format(c_name=c_name, mirror=mirror,
                                          types_rs=types_rs)
        for attempt in range(2):
            reply = await llm.ask(prompt, max_tokens=max_tokens)
            if reply.strip().splitlines()[0].strip() == "OPAQUE":
                return c_name, ""
            code = extract_rust(reply)
            names = set(fn_decls(code))
            if {f"{c_name}_sync_in", f"{c_name}_sync_out",
                f"{c_name}_sync_init"} <= names:
                return c_name, code
            prompt += (f"\nYour previous reply was invalid: it must define "
                       f"ALL of {c_name}_sync_in, {c_name}_sync_out and "
                       f"{c_name}_sync_init "
                       f"(found: {sorted(names)}). Emit the corrected reply.")
        return c_name, ""

    results = await asyncio.gather(*(convert(n, m) for n, m in structs.items()))
    return {n: code for n, code in results if code}


async def generate_shims(llm: LLM, pubs: list[tuple[str, str]], types_rs: str,
                         specs: dict[str, dict], max_tokens: int,
                         abi_decls: dict | None = None,
                         boundary: tuple[str, dict, dict] | None = None,
                         conversions: dict[str, str] | None = None
                         ) -> dict[str, str]:
    """{name: shim code} — one call per public function. `abi_decls` (from
    rustgen.surface.abi_decls) makes bindgen's header-derived signature the
    authority; without it the tree-sitter C signature stands alone."""
    import asyncio

    from rustgen.surface import compare_decl, fn_decls, render_abi_decl

    idiomatic = "\n".join(f"- {s}" for spec in specs.values()
                          for s in spec.get("signatures", [])) or "(none)"

    def _unwrap(code: str) -> str:
        """Strip a `pub mod ffi { ... }` wrapper the model may emit despite
        instructions — the shim is assembled into that module by us."""
        m = re.search(r"pub mod ffi\s*\{(?:\s*use super::\*;)?(.*)\}\s*$",
                      code, re.DOTALL)
        return m.group(1).strip() if m else code

    def _drift(name: str, code: str) -> list[str]:
        got = fn_decls(code).get(name)
        if got is None:
            return ["shim signature unparseable"]
        return compare_decl(got, abi_decls[name])

    mirror_mod, fn_sigs = (boundary[0], boundary[1]) if boundary else ("", {})
    conv_sigs = "\n".join(
        f"- {c}_sync_in / {c}_sync_out / {c}_sync_init (for c_abi::{c})"
        for c in (conversions or {})) or "(none — treat boundary structs opaquely)"

    async def shim(name: str, sig: str) -> tuple[str, str]:
        abi_block = ""
        if fn_sigs.get(name) and mirror_mod:
            # boundary mode: bindgen signature verbatim + sync discipline
            abi_block = BOUNDARY_BLOCK.format(
                bindgen_sig=fn_sigs[name], mirror=mirror_mod,
                conversion_sigs=conv_sigs)
        elif abi_decls and name in abi_decls:
            abi_block = ABI_BLOCK.format(abi_sig=render_abi_decl(name, abi_decls[name]))
        prompt = FFI_PROMPT.format(types_rs=types_rs, idiomatic_sigs=idiomatic,
                                   c_signature=sig, abi_block=abi_block)
        reply = await llm.ask(prompt, max_tokens=max_tokens)
        code = _unwrap(extract_rust(reply))
        if not (code and abi_decls and name in abi_decls):
            return name, code
        # the ABI decl is verifiable — enforce it, don't hope: one retry with
        # the exact mismatch quoted back
        problems = _drift(name, code)
        if problems:
            reply = await llm.ask(
                prompt + "\nYour previous shim violated the ABI AUTHORITY: "
                + "; ".join(problems) + "\nEmit the corrected shim.",
                max_tokens=max_tokens)
            fixed = _unwrap(extract_rust(reply))
            if fixed and not _drift(name, fixed):
                code = fixed
        return name, code

    results = await asyncio.gather(*(shim(n, s) for n, s in pubs))
    return {n: code for n, code in results if code}


def assemble_ffi(shims: dict[str, str], order: list[str],
                 boundary: tuple[str, dict, dict] | None = None,
                 conversions: dict[str, str] | None = None) -> str:
    prelude = ""
    if boundary and boundary[0]:
        parts = [boundary[0], ALLOC_PRELUDE] + list((conversions or {}).values())
        prelude = "\n\n".join(parts) + "\n\n"
    body = prelude + "\n\n".join(shims[n] for n in order if n in shims)
    indented = "\n".join(("    " + line if line.strip() else line)
                         for line in body.split("\n"))
    return "pub mod ffi {\n    use super::*;\n\n" + indented + "\n}"


async def generate_ffi(llm: LLM, source: str, split_over: int, types_rs: str,
                       specs: dict[str, dict], max_tokens: int,
                       abi_decls: dict | None = None,
                       boundary: tuple[str, dict, dict] | None = None
                       ) -> tuple[str, list[str]]:
    """Returns (ffi_rs, exported_symbol_names). Empty if no public functions.

    One call per public function — a whole-module call truncates on files with
    large public surfaces (26 shims blew a 6000-token budget mid-function,
    yielding unclosed delimiters).

    `boundary` (from rustgen.layout_stage.boundary_surface) switches shims to
    the sync-in/call/sync-out discipline over bindgen mirror structs, and
    extends the export list with header-declared functions the tree-sitter
    scan missed (macro-generated definitions like GIT_COMMIT_GETTER)."""
    pubs = public_functions(source, split_over)
    if boundary:
        missed = [n for n in boundary[1] if n not in dict(pubs)]
        pubs += [(n, "(macro-generated definition — no visible C body; "
                     "implement from the bindgen signature and the idiomatic "
                     "functions)") for n in missed]
    if not pubs:
        return "", []
    conversions = None
    if boundary and boundary[2]:
        conversions = await generate_conversions(llm, boundary[2], types_rs,
                                                 max_tokens)
    shims = await generate_shims(llm, pubs, types_rs, specs, max_tokens,
                                 abi_decls, boundary, conversions)
    return (assemble_ffi(shims, [n for n, _ in pubs], boundary, conversions),
            [name for name, _ in pubs])
