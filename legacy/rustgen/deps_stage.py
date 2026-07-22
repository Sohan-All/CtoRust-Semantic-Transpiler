"""Deps linkage stage — make the generated crate's external dependencies real.

The stage-T `pub mod deps` contains idiomatic stub signatures with `todo!()`
bodies: the crate compiles but panics the moment a shim call reaches a
dependency. This stage rewrites the module for the incremental-port pattern:

- each stub KEEPS its exact idiomatic signature (all generated code calls it),
- its body becomes an adapter that marshals to a raw `extern "C"` declaration
  of the real C symbol (declared in an inner `mod c`), resolved at LOAD time
  when the C library is present — cdylibs tolerate undefined symbols, so the
  crate still builds here; `nm` shows them as U,
- dependencies that are C MACROS (PQUEUE_PARENT_OF, git_pqueue_size aliases)
  cannot be linked — they are implemented natively in Rust from the macro
  body,
- dependencies whose C declaration cannot be found keep their `todo!()` stub,
  honestly reported.

Real declarations are located deterministically in the C source tree
(headers preferred over .c definitions).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from llm import LLM
from rustgen.common import extract_rust

# --------------------------------------------------------------------------- #
# C source lookup (deterministic)
# --------------------------------------------------------------------------- #

def _grep(pattern: str, root: Path, include: str,
          ignore_case: bool = False) -> list[tuple[Path, int]]:
    cmd = ["grep", "-rn", f"--include={include}", "-E", pattern, str(root)]
    if ignore_case:
        cmd.insert(1, "-i")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    hits = []
    for line in proc.stdout.splitlines():
        path, n, *_ = line.split(":", 2)
        hits.append((Path(path), int(n)))
    return hits


def _read_macro(path: Path, line_no: int) -> str:
    """#define plus its backslash continuations."""
    lines = path.read_text(errors="replace").split("\n")
    out = []
    i = line_no - 1
    while i < len(lines):
        out.append(lines[i])
        if not lines[i].rstrip().endswith("\\"):
            break
        i += 1
    return "\n".join(out)


def _read_decl(path: Path, line_no: int, name: str) -> str | None:
    """The declaration statement containing `name(` at line_no: from the
    statement start (may begin a line or two earlier — return type / GIT_*
    qualifiers) to the terminating ';' or the '{' of a definition."""
    lines = path.read_text(errors="replace").split("\n")
    start = line_no - 1
    # walk back while previous line doesn't end a statement/block
    while start > 0:
        prev = lines[start - 1].strip()
        if (not prev or prev.endswith((";", "}", "{", "*/"))
                or prev.startswith(("#", "//", "/*"))):
            break
        start -= 1
    buf = []
    for i in range(start, min(start + 12, len(lines))):
        line = lines[i]
        brace = line.find("{")
        semi = line.find(";")
        if brace != -1 and (semi == -1 or brace < semi):
            buf.append(line[:brace])
            return " ".join(" ".join(buf).split()) + ";"
        buf.append(line)
        if semi != -1:
            return " ".join(" ".join(buf).split())
    return None


_SYSTEM_HEADER_ROOT = Path("/usr/include")


def find_c_dep(name: str, root: Path) -> tuple[str, str] | None:
    """('macro', text) | ('fn', decl) | None. Macro lookup falls back to
    case-insensitive (stage T lowercases names like PQUEUE_PARENT_OF).
    Names absent from the project tree fall back to the system headers —
    libc deps (mktime, time) are LINK-trivial, the loader always has them."""
    pattern = rf"#\s*define\s+{re.escape(name)}\b"
    for ignore_case in (False, True):
        for include in ("*.h", "*.c"):
            hits = _grep(pattern, root, include, ignore_case=ignore_case)
            if hits:
                return "macro", _read_macro(*hits[0])
    roots = [root] + ([_SYSTEM_HEADER_ROOT] if _SYSTEM_HEADER_ROOT.is_dir() else [])
    for search_root in roots:
        for include in ("*.h", "*.c") if search_root == root else ("*.h",):
            hits = _grep(rf"\b{re.escape(name)}\s*\(", search_root, include)
            for path, line_no in hits:
                text = path.read_text(errors="replace").split("\n")[line_no - 1]
                if re.search(rf"#\s*define|//", text.split(name)[0]):
                    continue
                decl = _read_decl(path, line_no, name)
                if not decl:
                    continue
                m = re.search(rf"\b{re.escape(name)}\s*\(", decl)
                if not m:
                    continue
                # a CALL SITE also matches — a declaration's prefix is only
                # type/qualifier tokens, never operators or control keywords
                prefix = decl[:m.start()]
                if re.search(r"[=<>!?+\-/]|\b(if|while|for|return|switch|do|"
                             r"else|sizeof)\b", prefix):
                    continue
                return "fn", decl
    return None


# --------------------------------------------------------------------------- #
# stub extraction + module splicing
# --------------------------------------------------------------------------- #

def _match_braces(text: str, open_idx: int) -> int:
    """Index just past the brace matching text[open_idx] ('{')."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def extract_deps_module(types_rs: str) -> tuple[str, int, int] | None:
    """(module body text, start, end) of `pub mod deps { ... }` in types_rs."""
    m = re.search(r"pub mod deps\s*\{", types_rs)
    if not m:
        return None
    end = _match_braces(types_rs, m.end() - 1)
    return types_rs[m.start():end], m.start(), end


_STUB_FN = re.compile(r"pub fn (\w+)[^{]*")  # `;` is legal inside `[T; N]`


def stub_signatures(deps_module: str) -> dict[str, str]:
    """{fn name: full stub signature text (up to the body)}."""
    return {m.group(1): " ".join(m.group(0).split())
            for m in _STUB_FN.finditer(deps_module)}


# --------------------------------------------------------------------------- #
# adapter / native generation
# --------------------------------------------------------------------------- #

ADAPTER_PROMPT = """\
You are resolving ONE external dependency of a Rust reimplementation of a C
file. The Rust module exposes this idiomatic stub, and all generated code
calls it — its signature must stay EXACTLY as-is:

    {stub_sig}

The original C declaration:

    {c_decl}

Shared Rust types of this crate:
```rust
{types_rs}
```

FIRST DECIDE which resolution is semantically correct:

- NATIVE — the stub operates on THIS CRATE'S OWN idiomatic state (its
  parameters or return use the crate's shared types, Vec, String, ...). That
  state does NOT have the C struct's memory layout, so calling the C function
  on it would corrupt memory. Implement the behavior natively in safe Rust
  against the idiomatic types (a vector op becomes a Vec op, a string parse
  becomes str parsing). This is the right choice for data-structure and
  utility dependencies.

- LINK — the stub passes through OPAQUE HANDLES to state that lives on the C
  side (the crate never inspects it), or is a pure C-side effect (error
  registration, queries about C-side objects). Declare the C symbol and
  marshal: opaque handles are `*mut/*const core::ffi::c_void`, C strings via
  CString/CStr, out-params via pointers, `unsafe` only around the call. Never
  guess the memory layout of a non-opaque C struct — if marshaling would
  require that, choose NATIVE or `todo!("<why>")`.

Reply in EXACTLY ONE of these layouts:

NATIVE:
```rust
<the full function, stub signature verbatim, safe native body>
```

or

EXTERN:
```rust
<one C-ABI declaration line for `mod c`, ending with ;>
```
ADAPTER:
```rust
<the full adapter function, stub signature verbatim, calling c::{name}>
```
"""

NATIVE_PROMPT = """\
An external dependency of a Rust reimplementation turns out to be a C MACRO,
so it cannot be linked — implement it natively. The Rust module exposes this
stub, called by generated code; its signature must stay EXACTLY as-is:

    {stub_sig}

The C macro definition:

```c
{macro_text}
```

If the macro aliases another C function (e.g. `#define a b`), implement the
aliased behavior; if it is an expression macro, translate the expression.
Shared Rust types for context:
```rust
{types_rs}
```

Reply with ONLY a ```rust code block containing the full function (same
signature, native body — no extern, no unsafe unless unavoidable).
"""


def _unwrap_extern_block(decl: str) -> str:
    """Models emit anything from a bare decl to a whole `extern "C" { ... }`
    block with struct definitions mixed in. Re-parse the fn declarations and
    re-render them sanitized (struct pointers -> c_void — `mod c` defines no
    types) — anything unmappable (callback params, by-value structs) is
    dropped, demoting its adapter to an honest stub downstream."""
    from rustgen.surface import extern_decl_rust, fn_decls

    decls = [extern_decl_rust(name, d) for name, d in fn_decls(decl).items()]
    return "\n        ".join(d for d in decls if d)


def _clean_item(code: str) -> str:
    """Strip reply-layout artifacts that leak into item code: bare
    NATIVE:/EXTERN:/ADAPTER: label lines and stray code-fence markers."""
    return "\n".join(l for l in code.split("\n")
                     if not re.fullmatch(r"\s*(NATIVE:|EXTERN:|ADAPTER:|```\w*)\s*", l))


def _parse_adapter_reply(reply: str) -> tuple[str, str | None, str] | None:
    """('native', None, code) | ('extern', decl, adapter) | None."""
    if "NATIVE:" in reply and "EXTERN:" not in reply:
        body = _clean_item(extract_rust(reply.split("NATIVE:", 1)[1]))
        return ("native", None, body) if body else None
    if "EXTERN:" in reply and "ADAPTER:" in reply:
        extern_part, adapter_part = reply.split("ADAPTER:", 1)
        if "EXTERN:" not in extern_part:  # sections out of order -> retry
            return None
        extern_decl = extract_rust(extern_part.split("EXTERN:", 1)[1])
        adapter = _clean_item(extract_rust(adapter_part))
        if extern_decl and adapter:
            return "extern", _unwrap_extern_block(extern_decl), adapter
    return None


def _decl_headers(names: list[str], root: Path) -> list[str]:
    """Bare header names declaring any of `names` (for a bindgen wrapper —
    resolved via surface.discover_include_dirs -I paths)."""
    headers = []
    for name in names:
        for path, _ in _grep(rf"\b{re.escape(name)}\s*\(", root, "*.h"):
            if path.name not in headers:
                headers.append(path.name)
            break
    return headers


def _bindgen_backfill(names: list[str], c_root: Path,
                      clang_args: str, bindgen_bin: str) -> dict[str, str]:
    """{name: compilable extern decl} for every name bindgen can map soundly.
    Adapters reference `mod c` helpers the model never declared; their decls
    are derivable deterministically — demotion is the last resort, not the
    first response."""
    from rustgen.surface import extern_decl_rust, fn_decls, run_bindgen

    headers = _decl_headers(names, c_root)
    if not headers:
        return {}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = run_bindgen(c_root, headers, Path(td), clang_args, bindgen_bin)
        if out is None:
            return {}
        decls = fn_decls(out.read_text(errors="replace"))
    filled = {}
    for name in names:
        if name in decls:
            decl = extern_decl_rust(name, decls[name])
            if decl:
                filled[name] = decl
    return filled


def _todo_stubs(deps_module: str) -> dict[str, tuple[int, int, str]]:
    """{name: (fn start, fn end, signature)} for deps fns whose body is a
    todo!() stub. Per-head scan to the fn's OWN `{` or `;` — regex greed here
    once swallowed sibling fn heads and matched the wrong body (extern decls
    in `mod c` are bodiless)."""
    out = {}
    for m in re.finditer(r"pub fn (\w+)", deps_module):
        j, brackets = m.end(), 0
        while j < len(deps_module):
            ch = deps_module[j]
            if ch == "[":
                brackets += 1
            elif ch == "]":
                brackets -= 1
            elif ch == "{" or (ch == ";" and brackets == 0):
                break  # `;` inside `[T; N]` is part of the type, not an end
            j += 1
        if j >= len(deps_module) or deps_module[j] == ";":
            continue  # bodiless extern decl
        end = _match_braces(deps_module, j)
        if "todo!(" in deps_module[j:end]:
            sig = " ".join(deps_module[m.start():j].split())
            out[m.group(1)] = (m.start(), end, sig)
    return out


async def relink_todo_stubs(llm: LLM, types_rs: str, c_root: Path,
                            max_tokens: int) -> tuple[str, dict]:
    """Targeted repair: re-adapt ONLY the deps stubs still carrying todo!()
    bodies, splicing successes into the otherwise-untouched module (re-linking
    a converged crate wholesale is a measured regression risk). New extern
    decls merge into the existing `mod c`. Returns (new_types_rs, report)."""
    found = extract_deps_module(types_rs)
    if found is None:
        return types_rs, {"note": "no deps module", "fixed": []}
    module, mod_start, mod_end = found
    stubs = _todo_stubs(module)
    if not stubs:
        return types_rs, {"note": "no todo stubs", "fixed": [],
                          "still_todo": [], "new_externs": 0}
    sigs = {name: sig for name, (_, _, sig) in stubs.items()}

    existing_decls = set()
    c_block = re.search(r'pub\(super\) mod c\s*\{\s*extern "C"\s*\{', module)
    if c_block:
        c_end = _match_braces(module, c_block.end() - 1)
        existing_decls = set(re.findall(r"\bfn\s+(\w+)",
                                        module[c_block.end():c_end]))

    async def build(name: str) -> tuple[str, str | None, str | None]:
        hit = find_c_dep(name, c_root)
        if hit is None:
            return name, None, None
        kind, text = hit
        if kind == "macro":
            reply = await llm.ask(NATIVE_PROMPT.format(
                stub_sig=sigs[name], macro_text=text, types_rs=types_rs),
                max_tokens=max_tokens)
            body = _clean_item(extract_rust(reply))
            return name, None, (body or None)
        prompt = ADAPTER_PROMPT.format(
            stub_sig=sigs[name], c_decl=text, name=name, types_rs=types_rs)
        for _ in range(2):
            reply = await llm.ask(prompt, max_tokens=max_tokens)
            parsed = _parse_adapter_reply(reply)
            if parsed:
                _, extern_decl, item = parsed
                return name, extern_decl, item
        return name, None, None

    results = await asyncio.gather(*(build(n) for n in stubs))

    new_declared = existing_decls | {
        n for _, decl, _ in results if decl
        for n in re.findall(r"\bfn\s+(\w+)", decl)}
    fixed, new_externs = {}, []
    for name, extern_decl, item in results:
        if not item:
            continue
        undeclared = [n for n in re.findall(r"\bc::(\w+)", item)
                      if n not in new_declared]
        if undeclared:
            continue
        fixed[name] = item
        if extern_decl:
            for line in extern_decl.split("\n"):
                fn = re.search(r"\bfn\s+(\w+)", line)
                if fn and fn.group(1) not in existing_decls:
                    new_externs.append(line.strip())

    # splice fn replacements (descending offsets so earlier spans stay valid)
    for name in sorted(fixed, key=lambda n: -stubs[n][0]):
        start, end, _ = stubs[name]
        body = "\n".join(("    " + l if l.strip() else l)
                         for l in fixed[name].strip().split("\n")).lstrip()
        module = module[:start] + body + module[end:]

    if new_externs:
        if c_block := re.search(r'(extern "C"\s*\{)', module):
            at = c_block.end()
            module = (module[:at] + "\n            "
                      + "\n            ".join(new_externs) + module[at:])
        else:
            block = ("    #[allow(non_snake_case, dead_code, improper_ctypes)]\n"
                     "    pub(super) mod c {\n        extern \"C\" {\n            "
                     + "\n            ".join(new_externs)
                     + "\n        }\n    }\n\n")
            after_use = module.find("use super::*;")
            at = module.find("\n", after_use) + 1 if after_use != -1 \
                else module.find("{") + 1
            module = module[:at] + "\n" + block + module[at:]

    report = {"fixed": sorted(fixed),
              "still_todo": sorted(set(stubs) - set(fixed)),
              "new_externs": len(new_externs)}
    return types_rs[:mod_start] + module + types_rs[mod_end:], report


async def link_deps(llm: LLM, types_rs: str, c_root: Path,
                    max_tokens: int, clang_args: str = "",
                    bindgen_bin: str = "bindgen") -> tuple[str, dict]:
    """Rewrite `pub mod deps` in types_rs with real linkage.
    Returns (new_types_rs, report)."""
    found = extract_deps_module(types_rs)
    if found is None:
        return types_rs, {"note": "no deps module", "linked": 0}
    module, start, end = found
    stubs = stub_signatures(module)
    if not stubs:
        return types_rs, {"note": "deps module empty", "linked": 0}

    lookups = {name: find_c_dep(name, c_root) for name in stubs}

    async def build(name: str) -> tuple[str, str, str | None, str | None]:
        """(name, status, extern_decl, item_code)"""
        hit = lookups[name]
        if hit is None:
            return name, "unresolved", None, None
        kind, text = hit
        if kind == "macro":
            reply = await llm.ask(NATIVE_PROMPT.format(
                stub_sig=stubs[name], macro_text=text, types_rs=types_rs),
                max_tokens=max_tokens)
            body = extract_rust(reply)
            return (name, "native", None, body) if body else (name, "unresolved", None, None)
        prompt = ADAPTER_PROMPT.format(
            stub_sig=stubs[name], c_decl=text, name=name, types_rs=types_rs)
        parsed = None
        for _ in range(2):  # reply-layout flakiness: one retry
            reply = await llm.ask(prompt, max_tokens=max_tokens)
            parsed = _parse_adapter_reply(reply)
            if parsed is not None:
                break
        if parsed is None:
            return name, "unresolved", None, None
        kind_out, extern_decl, item = parsed
        return name, kind_out, extern_decl, item

    results = await asyncio.gather(*(build(n) for n in stubs))

    declared = {n for n, status, decl, _ in results
                if status == "extern" and decl
                for n in re.findall(r"\bfn\s+(\w+)", decl)}
    # adapters may call `mod c` helpers the model never declared — derive
    # those decls deterministically from bindgen before resorting to demotion
    missing = sorted({n for _, status, _, item in results if status != "unresolved"
                      for n in re.findall(r"\bc::(\w+)", item or "")
                      if n not in declared})
    backfilled = _bindgen_backfill(missing, c_root, clang_args, bindgen_bin)
    declared |= set(backfilled)

    externs = ["        " + d for d in backfilled.values()]
    items, statuses = [], {}
    for name, status, extern_decl, item in results:
        # a body still referencing undeclared `mod c` symbols cannot compile —
        # demote to an honest stub
        if status != "unresolved":
            undeclared = [n for n in re.findall(r"\bc::(\w+)", item or "")
                          if n not in declared]
            if undeclared:
                status = "unresolved"
        statuses[name] = status
        if status == "unresolved":
            # keep the original stub body (todo!) for this name — reconstruct
            # a minimal stub from its signature
            items.append(f"{stubs[name]} {{\n        todo!(\"C declaration for "
                         f"{name} not found in source tree\")\n    }}")
            continue
        if extern_decl:
            externs.append("        " + extern_decl.strip())
        items.append(item)

    extern_block = ""
    if externs:
        extern_block = ("    #[allow(non_snake_case, dead_code, improper_ctypes)]\n"
                        "    pub(super) mod c {\n"
                        "        extern \"C\" {\n"
                        + "\n".join("    " + e for e in externs)
                        + "\n        }\n    }\n\n")
    body = "\n\n".join(items)
    indented = "\n".join(("    " + l if l.strip() else l) for l in body.split("\n"))
    new_module = "pub mod deps {\n    use super::*;\n\n" + extern_block + indented + "\n}"

    report = {
        "linked": sum(1 for s in statuses.values() if s == "extern"),
        "native": sum(1 for s in statuses.values() if s == "native"),
        "backfilled": sorted(backfilled),
        "unresolved": sorted(n for n, s in statuses.items() if s == "unresolved"),
    }
    return types_rs[:start] + new_module + types_rs[end:], report
