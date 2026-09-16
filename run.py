#!/usr/bin/env python3
"""MTU discovery pipeline — CLI entry point.

Usage:
    python3 run.py <file.c> --strategy diffusion
    python3 run.py <file.c> --strategy whole_file
    python3 run.py <file.c> --strategy both        # A/B: separate output dirs

Output lands in out/<file-stem>[.<strategy>]/{state.jsonl, explanations.md}.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from config import Config, load_config
from llm import LLM
from state import Store, LOCKED, IRREDUCIBLE
from strategies.diffusion import DiffusionStrategy
from strategies.whole_file import WholeFileStrategy

STRATEGIES = {
    "diffusion": DiffusionStrategy,
    "whole_file": WholeFileStrategy,
}


async def run_one(strategy_name: str, source: str, source_path: Path, out_root: Path,
                  cfg: Config, suffix: bool) -> None:
    stem = source_path.stem + (f".{strategy_name}" if suffix else "")
    # Function-pointer typedefs from the whole directory, not just this file:
    # every corpus project that declares a callback as `ArrayListCompareFunc
    # cmp` keeps the typedef in a HEADER, so a .c file cannot resolve its own
    # callback parameters. Without this the chunker files `compare_func` under
    # `calls_external` -> `external_deps`, and stage T stubs it in
    # `<stem>_deps` as a global no caller can ever supply. Set before the
    # config record is written so the record shows what the run actually saw.
    if not cfg.fn_ptr_typedefs:
        from chunker import fn_pointer_typedefs
        seen: set[str] = set()
        for f in sorted(source_path.parent.rglob("*.h")) + \
                 sorted(source_path.parent.rglob("*.c")):
            try:
                seen |= fn_pointer_typedefs(f.read_text(errors="replace"))
            except OSError:
                continue
        cfg.fn_ptr_typedefs = sorted(seen)
    store = Store(out_root / stem)
    store.write_record(cfg.to_record())
    store.event("start", strategy=strategy_name, file=str(source_path))
    llm = LLM(cfg, role="worker")
    strategy = STRATEGIES[strategy_name](cfg, llm, store)

    t0 = time.time()
    await strategy.run(source, source_path.name)
    store.write_record(llm.usage_record())
    elapsed = time.time() - t0

    final = store.final_units()
    locked = [e for e in final if e.status == LOCKED]
    irr = [e for e in final if e.status == IRREDUCIBLE]
    print(f"\n[{strategy_name}] {source_path.name}: "
          f"{len(locked)} locked MTU(s), {len(irr)} irreducible, "
          f"{llm.calls} calls, {llm.input_tokens}+{llm.output_tokens} tokens, "
          f"{elapsed:.1f}s")
    for e in final:
        ranges = ", ".join(f"{s}-{en}" for s, en in e.ranges)
        marker = "LOCK" if e.status == LOCKED else "IRRD"
        print(f"  [{marker}] lines {ranges}: {e.text[:100]}")
    print(f"  -> {store.out_dir / 'explanations.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MTU discovery pipeline")
    ap.add_argument("file", type=Path, help="C source file")
    ap.add_argument("--strategy", choices=[*STRATEGIES, "both"], default="diffusion")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file overriding Config defaults, e.g. "
                         '{"worker_model": "gemma-4-31b"} (also read from '
                         "the DIFFUSIONMTUS_CONFIG env var)")
    ap.add_argument("--model", default=None, help="override worker model")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.worker_model = args.model
    if args.concurrency:
        cfg.concurrency = args.concurrency

    source = args.file.read_text()
    names = list(STRATEGIES) if args.strategy == "both" else [args.strategy]

    async def go():
        for name in names:
            await run_one(name, source, args.file, args.out, cfg,
                          suffix=(args.strategy == "both"))

    asyncio.run(go())


if __name__ == "__main__":
    main()
