"""Every prompt template and retry note must survive `.format()`.

    PYTHONPATH=. venv/bin/python test_prompts.py

Why this exists. `types_stage.STUB_RETRY_NOTE` contained a literal
`impl SomeType { ... }`, which `str.format` reads as a placeholder named " ... ".
It raised `KeyError: ' '` and killed a translation run outright — but ONLY on the
path where the stub gate actually fires, so it sat latent through a full trial
and every unit test of `illegal_stubs` before taking down trial 4.

The lesson is that testing the checker is not testing the response to the
checker. These tests format every template, and drive the retry paths with a
fake LLM that returns bad output, so the notes are exercised rather than merely
inspected.
"""

from __future__ import annotations

import asyncio
import string
import sys

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


def test_templates_format() -> None:
    """Every module-level prompt string formats with its own placeholders."""
    print("\n=== templates survive .format() ===")
    import rustgen.code_stage as cs
    import rustgen.spec_stage as ss
    import rustgen.types_stage as ts
    import rustgen.compile_loop as cl

    mods = {"code_stage": cs, "spec_stage": ss, "types_stage": ts,
            "compile_loop": cl}
    for mname, mod in mods.items():
        for attr in sorted(dir(mod)):
            if not (attr.endswith("_PROMPT") or attr.endswith("_NOTE")):
                continue
            tmpl = getattr(mod, attr)
            if not isinstance(tmpl, str):
                continue
            keys = {f[1] for f in string.Formatter().parse(tmpl) if f[1]}
            try:
                tmpl.format(**{k: "X" for k in keys})
                check(f"{mname}.{attr} ({len(keys)} placeholder(s))", True)
            except (KeyError, IndexError, ValueError) as e:
                check(f"{mname}.{attr}", False, f"{type(e).__name__}: {e}")


class _FakeLLM:
    """Returns canned replies so the retry paths run for real.

    `replies` is consumed in order; the last one repeats once exhausted.
    """

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls = 0

    async def ask(self, prompt: str, **kw) -> str:
        self.calls += 1
        i = min(self.calls - 1, len(self.replies) - 1)
        return self.replies[i]

    async def ask_json(self, prompt: str, **kw):
        return {}


def test_types_stage_retry() -> None:
    """The path that crashed trial 4: stage T emits a stubbed impl block, the
    gate rejects it, and the retry note gets formatted."""
    print("\n=== types_stage retry path (the trial-4 crash) ===")
    from rustgen.types_stage import synthesize_types
    from state import Explanation

    stubbed = ("```rust\npub struct S { pub a: i32 }\n"
               "impl S { pub fn f(&self) -> i32 { todo!() } }\n```\n"
               'GLOSSARY:\n```json\n{"thing": "S"}\n```')
    clean = ("```rust\npub struct S { pub a: i32 }\n```\n"
             'GLOSSARY:\n```json\n{"thing": "S"}\n```')
    unit = Explanation(id="exp_0001", text="does a thing", invariants=[],
                       ranges=[(1, 5)], status="locked")

    # bad twice then good: exercises the note, then succeeds
    llm = _FakeLLM([stubbed, stubbed, clean])
    try:
        rs, gloss = asyncio.run(synthesize_types(llm, [unit], 100))
        check("retry note formats and a later clean draw is accepted",
              "todo!()" not in rs and llm.calls == 3, f"calls={llm.calls}")
    except Exception as e:
        check("retry note formats", False, f"{type(e).__name__}: {e}")

    # bad every time: exhausts retries, must return the last draw, not raise
    llm = _FakeLLM([stubbed])
    try:
        rs, gloss = asyncio.run(synthesize_types(llm, [unit], 100))
        check("exhausted retries return last draw instead of raising",
              "todo!()" in rs and llm.calls == 3, f"calls={llm.calls}")
    except Exception as e:
        check("exhausted retries do not raise", False, f"{type(e).__name__}: {e}")


def test_code_stage_retry() -> None:
    """Same for stage C's two pre-flight checks (stubs and delimiters)."""
    print("\n=== code_stage retry paths ===")
    from rustgen.code_stage import generate_code
    from state import Explanation

    unit = Explanation(id="exp_0001", text="does a thing", invariants=[],
                       ranges=[(1, 5)], status="locked")
    specs = {"exp_0001": {"signatures": ["pub(crate) fn f() -> i32"]}}

    stubbed = "```rust\npub(crate) fn f() -> i32 { todo!() }\n```"
    unbalanced = "```rust\npub(crate) fn f() -> i32 { 1\n```"
    clean = "```rust\npub(crate) fn f() -> i32 { 1 }\n```"

    for label, bad, want in [("stub", stubbed, "todo!()"),
                             ("unbalanced delimiters", unbalanced, None)]:
        llm = _FakeLLM([bad, clean])
        try:
            out = asyncio.run(generate_code(llm, [unit], specs, "", 100))
            code = out.get("exp_0001", "")
            check(f"{label}: note formats and clean retry is accepted",
                  "todo!()" not in code and llm.calls == 2, f"calls={llm.calls}")
        except Exception as e:
            check(f"{label}: note formats", False, f"{type(e).__name__}: {e}")


def main() -> int:
    test_templates_format()
    test_types_stage_retry()
    test_code_stage_retry()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
