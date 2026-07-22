"""Sibling-deps resolution — the multi-file analog of deps_stage.

Per-file Stage T stubs the functions a file needs from elsewhere under C
names (`*_deps` modules, todo!() bodies). In a single-file translation those
became extern "C"; in a multi-file translation the implementations exist IN
THE SAME CRATE under idiomatic names — the stub just doesn't know which one.
This stage closes that gap: one Haiku call per stub, given the stub's
signature/doc and the full project signature registry, returning either

    DELEGATE:  <body calling the sibling idiomatic function>
    NATIVE:    <body implementing a trivial libc-ism directly>
    KEEP       (genuinely missing — stays an honest todo!())

Replies are spliced mechanically (never trusted for layout): the stub's
todo!() body is replaced, nothing else moves.
"""
from __future__ import annotations

import asyncio
import re

from llm import LLM
from rustgen.common import extract_rust

RESOLVE_PROMPT = """\
A multi-file C project was redesigned as ONE Rust crate. The stub below was
generated from a C-style dependency name; its real implementation exists in
this same crate under an idiomatic name (listed in the REGISTRY), or it is a
trivial memory/lifecycle operation Rust ownership already handles.

Write the stub's BODY (replacing todo!()):
- If a registry function implements this behavior: call it, adapting
  arguments/returns (e.g. Option<->Result, &str<->String, Box wrapping).
  Methods appear as `Type::method(&self, ...)` — call them method-style.
- If it is a free/dealloc operation: `drop(...)` or empty (ownership).
- If NOTHING in the registry matches, reply with the single word KEEP.
- The body must use only: the stub's parameters, registry functions/methods,
  shared types, and Rust std. No unsafe, no new helper functions.

SHARED TYPES (abbreviated):
```rust
{types_summary}
```

REGISTRY (functions and methods implemented in this crate):
{registry}

THE STUB (in module `{module}`):
```rust
{stub}
```

Reply with ONLY a ```rust code block containing the COMPLETE stub function
with its real body (same signature), or the single word KEEP.
"""


def find_stubs(types_rs: str) -> list[dict]:
    """Every fn with a todo!() body inside a `*_deps` module:
    [{module, name, text, start, end}] with offsets into types_rs."""
    stubs = []
    for mm in re.finditer(r"pub mod (\w+_deps)\s*\{", types_rs):
        module = mm.group(1)
        # module body span by brace matching
        i = mm.end() - 1
        d = 0
        for j in range(i, len(types_rs)):
            if types_rs[j] == "{":
                d += 1
            elif types_rs[j] == "}":
                d -= 1
                if d == 0:
                    break
        body_start, body_end = mm.end(), j
        for fm in re.finditer(r"pub fn (\w+)", types_rs[body_start:body_end]):
            fstart = body_start + fm.start()
            bi = types_rs.index("{", body_start + fm.end())
            d = 0
            for k in range(bi, body_end):
                if types_rs[k] == "{":
                    d += 1
                elif types_rs[k] == "}":
                    d -= 1
                    if d == 0:
                        break
            text = types_rs[fstart:k + 1]
            if "todo!()" in text:
                stubs.append({"module": module, "name": fm.group(1),
                              "text": text, "start": fstart, "end": k + 1})
    return stubs


def _signature_of(fn_text: str) -> str:
    return fn_text[:fn_text.index("{")].strip()


async def resolve_sibling_stubs(llm: LLM, types_rs: str, registry: list[str],
                                types_summary: str, max_tokens: int
                                ) -> tuple[str, dict]:
    """Returns (patched types_rs, report). Splices resolved bodies by
    descending offset so earlier offsets stay valid."""
    stubs = find_stubs(types_rs)
    reg = "\n".join(f"- {s}" for s in registry)

    async def resolve(stub: dict) -> tuple[dict, str | None]:
        prompt = RESOLVE_PROMPT.format(types_summary=types_summary,
                                       registry=reg, module=stub["module"],
                                       stub=stub["text"])
        for attempt in range(2):
            reply = await llm.ask(prompt, max_tokens=max_tokens)
            if reply.strip() == "KEEP":
                return stub, None
            code = extract_rust(reply)
            # mechanical validation: same fn name, no todo, single fn
            if (code and f"fn {stub['name']}" in code
                    and "todo!()" not in code
                    and code.count("pub fn") <= 1):
                return stub, code.strip()
            prompt += ("\nYour previous reply was invalid (must be the one "
                       "complete function, same name, no todo!()). "
                       "Reply KEEP if it cannot be implemented.")
        return stub, None

    results = await asyncio.gather(*(resolve(s) for s in stubs))
    resolved = [(s, c) for s, c in results if c]
    kept = [s["name"] for s, c in results if not c]
    for stub, code in sorted(resolved, key=lambda x: -x[0]["start"]):
        types_rs = types_rs[:stub["start"]] + code + types_rs[stub["end"]:]
    return types_rs, {"resolved": [s["name"] for s, _ in resolved],
                      "kept": kept, "total": len(stubs)}
