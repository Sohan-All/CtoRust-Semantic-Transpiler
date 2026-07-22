"""Boundary layout authority — bindgen-derived repr(C) mirror structs.

The lib_swap differential showed function-signature ABI is necessary but not
sufficient: C code accesses the STRUCTS the translated code fills — embedded
by value (git_vector inside revwalk), fields read directly (sig->name), or
C-allocated buffers handed over to fill (sizeof(git_commit) in object.c).
The persistent state must therefore live in C-layout memory.

Design decision (Sohan, July 10): the idiomatic Rust core stays untouched —
the C layout is confined to a conversion layer at the FFI boundary. This
module supplies that layer's raw material, deterministically:

  boundary_surface(c_root, source_name, ...) ->
      module_text : a `mod c_abi` with the repr(C) structs / aliases / consts
                    reachable from the file's exported function signatures,
                    verbatim from bindgen (bindgen IS the authority; nothing
                    here is model output)
      fn_sigs     : {name: full bindgen signature} — shims adopt these
                    verbatim, so pointer types are the real struct types,
                    not erased c_void
      structs     : {name: struct definition} — the boundary types that need
                    sync-in/sync-out conversion functions

Everything is parsed from bindgen's machine-generated output, which is
line-regular: `pub const`/`pub type` one-liners, `#[repr(C)] ... pub
struct/union N {...}` blocks, and one-fn `unsafe extern "C" {...}` blocks.
"""
from __future__ import annotations

import re
from pathlib import Path

from rustgen.surface import header_for_source, run_bindgen

_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_KEYWORDS = frozenset(
    "pub const type struct union enum fn unsafe extern mut usize isize bool "
    "u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 std os raw core ffi c_char c_int "
    "c_uint c_long c_ulong c_uchar c_schar c_short c_ushort c_void c_longlong "
    "c_ulonglong Option Copy Clone Debug derive repr C".split())


def parse_items(bindings: str) -> dict[str, tuple[str, str]]:
    """{name: (kind, item_text)} over bindgen output. kind is one of
    const|type|struct|union|fn."""
    items: dict[str, tuple[str, str]] = {}
    lines = bindings.splitlines()
    i, pending_attrs = 0, []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#["):
            pending_attrs.append(line)
            i += 1
            continue
        m = re.match(r"pub (const|type) (\w+)", stripped)
        if m:
            block = pending_attrs + [line]
            while not block[-1].rstrip().endswith(";"):
                i += 1
                block.append(lines[i])
            items[m.group(2)] = (m.group(1), "\n".join(block))
            pending_attrs = []
            i += 1
            continue
        m = re.match(r"pub (struct|union) (\w+)", stripped)
        if m:
            block = pending_attrs + [line]
            if not stripped.endswith(";"):          # unit structs end in ;
                depth = line.count("{") - line.count("}")
                while depth > 0:
                    i += 1
                    block.append(lines[i])
                    depth += lines[i].count("{") - lines[i].count("}")
            items[m.group(2)] = (m.group(1), "\n".join(block))
            pending_attrs = []
            i += 1
            continue
        if stripped.startswith(("unsafe extern", "extern")) and "{" in line:
            block = [line]
            depth = line.count("{") - line.count("}")
            while depth > 0:
                i += 1
                block.append(lines[i])
                depth += lines[i].count("{") - lines[i].count("}")
            body = "\n".join(block)
            fm = re.search(r"pub fn (\w+)", body)
            if fm:
                items[fm.group(1)] = ("fn", body)
            pending_attrs = []
            i += 1
            continue
        pending_attrs = []
        i += 1
    return items


def _referenced(item_text: str, own_name: str) -> set[str]:
    names = set(_IDENT.findall(item_text)) - _KEYWORDS - {own_name}
    return names


def type_closure(items: dict[str, tuple[str, str]],
                 roots: list[str]) -> list[str]:
    """Names of every item transitively referenced from `roots`, in
    deterministic (discovery, then alpha) order, excluding fn items —
    the closure is the TYPE surface the mirror module must carry."""
    seen: list[str] = []
    frontier = list(roots)
    while frontier:
        name = frontier.pop(0)
        if name not in items:
            continue
        for ref in sorted(_referenced(items[name][1], name)):
            if ref in items and ref not in seen and ref not in frontier:
                if items[ref][0] != "fn":
                    seen.append(ref)
                    frontier.append(ref)
    return seen


def boundary_surface(c_root: Path, source_name: str, exported: list[str],
                     out_dir: Path, clang_args: str = "",
                     bindgen_bin: str = "bindgen"
                     ) -> tuple[str, dict[str, str], dict[str, str]]:
    """(module_text, fn_sigs, structs) for one translated file — see module
    docstring. Empty results when bindgen/headers are unavailable (the FFI
    stage then falls back to its pre-layout behavior, honestly)."""
    header = header_for_source(c_root, source_name)
    if header is None:
        return "", {}, {}
    bindings = run_bindgen(c_root, [header], out_dir, clang_args, bindgen_bin)
    if bindings is None:
        return "", {}, {}
    items = parse_items(bindings.read_text())

    # A second, header-scoped pass enumerates the functions THIS header
    # declares — that catches macro-generated definitions the tree-sitter
    # function scan cannot see (they have no visible C body in the .c file).
    own_fns: set[str] = set(exported)
    header_path = next(c_root.rglob(header), None)
    if header_path is not None:
        scoped = run_bindgen(c_root, [header], out_dir / "own", clang_args,
                             bindgen_bin, allow=re.escape(str(header_path)))
        if scoped is not None:
            own_fns |= {n for n, (k, _) in parse_items(scoped.read_text()).items()
                        if k == "fn"}

    fn_sigs: dict[str, str] = {}
    for name in sorted(own_fns):
        if name in items and items[name][0] == "fn":
            m = re.search(r"pub fn .*?;", items[name][1], re.S)
            if m:
                fn_sigs[name] = m.group(0)

    needed = type_closure(items, list(fn_sigs))
    structs = {n: items[n][1] for n in needed
               if items[n][0] in ("struct", "union")}
    if not needed:
        return "", fn_sigs, {}

    body = "\n".join(items[n][1] for n in needed)
    module_text = (
        "/// LAYOUT AUTHORITY (bindgen) — C-identical mirror types. C code\n"
        "/// reads/embeds these structs directly; persistent state crossing\n"
        "/// the FFI boundary must live in THIS layout. Do not edit fields.\n"
        "#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals,\n"
        "        dead_code, unused)]\n"
        "pub mod c_abi {\n"
        + "\n".join("    " + l if l.strip() else l for l in body.splitlines())
        + "\n}\n")
    return module_text, fn_sigs, structs
