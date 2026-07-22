#!/usr/bin/env python3
"""MTU -> Rust generation (v1, validation-free).

Usage:
    python3 -m rustgen.run_rust out/tree-cache

Consumes out/<stem>/state.jsonl from a completed MTU run; writes
out/<stem>/rust_crate/ and appends rust_spec / rust_code records for
provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import time
from pathlib import Path

from config import Config, load_config
from llm import LLM
from state import Store
from rustgen.types_stage import synthesize_types
from rustgen.spec_stage import generate_specs
from rustgen.code_stage import generate_code
from rustgen.assemble import assemble
from rustgen.compile_loop import compile_loop
from rustgen.deps_stage import link_deps
from rustgen.ffi_stage import generate_ffi


def _latest_records(store: Store):
    """Most recent rust_types / rust_spec / rust_code / rust_ffi records from
    state.jsonl (later records win — the file is a log)."""
    types_rs, glossary, ffi = None, {}, ("", [])
    specs: dict[str, dict] = {}
    code: dict[str, str] = {}
    for r in store.records:
        t = r.get("type")
        if t == "rust_types":
            types_rs, glossary = r.get("types_rs"), r.get("glossary", {})
        elif t == "rust_spec":
            specs[r["unit"]] = {k: v for k, v in r.items()
                                if k not in ("type", "ts", "unit")}
        elif t == "rust_code":
            code[r["unit"]] = r.get("code", "")
        elif t == "rust_ffi":
            ffi = (r.get("code", ""), r.get("exported", []))
    return types_rs, glossary, specs, code, ffi


async def run(out_dir: Path, compile_rounds: int | None = None,
              resume: bool = False, spec_mode: str | None = None,
              deps_mode: str | None = None, c_root: Path | None = None,
              bindgen_args: str | None = None,
              regen_ffi: bool = False, config: Path | None = None) -> None:
    store = Store.load(out_dir)
    units = store.final_units()
    if not units:
        raise SystemExit(f"{out_dir}: no locked/irreducible units in state.jsonl")

    source_name = next((r.get("file", "?") for r in store.records
                        if r.get("type") == "event" and r.get("event") == "start"), "?")

    base = load_config(config)
    cfg = dataclasses.replace(base, worker_model=base.rustgen_model)
    if compile_rounds is not None:
        cfg = dataclasses.replace(cfg, rustgen_compile_rounds=compile_rounds)
    if spec_mode is not None:
        cfg = dataclasses.replace(cfg, rustgen_spec_mode=spec_mode)
    if deps_mode is not None:
        cfg = dataclasses.replace(cfg, rustgen_deps_mode=deps_mode)
    if c_root is not None:
        cfg = dataclasses.replace(cfg, c_source_root=str(c_root))
    if bindgen_args is not None:
        cfg = dataclasses.replace(cfg, bindgen_clang_args=bindgen_args)
    llm = LLM(cfg)
    t0 = time.time()

    prev = _latest_records(store) if resume else (None, {}, {}, {}, ("", []))
    types_rs, glossary, specs, code, (ffi_rs, exported) = prev
    if regen_ffi:
        ffi_rs, exported = "", []

    if types_rs is None:
        types_rs, glossary = await synthesize_types(llm, units, cfg.rustgen_types_max_tokens)
        store.write_record({"type": "rust_types", "types_rs": types_rs, "glossary": glossary})

    src_path = Path(source_name)
    if not src_path.exists():  # start event stores path relative to run cwd
        src_path = Path(__file__).parent.parent / source_name

    # per-unit context: reverse call graph (who calls what this unit defines)
    # + the unit's raw C lines when cfg.rustgen_c_source_context is on
    extras: dict[str, str] = {}
    if src_path.exists():
        from rustgen.common import unit_extras
        extras = unit_extras(units, src_path.read_text(),
                             cfg.split_function_over_lines,
                             cfg.rustgen_c_source_context)

    if not specs:
        specs = await generate_specs(llm, units, types_rs, glossary, cfg,
                                     extras=extras)
        for uid, spec in specs.items():
            store.write_record({"type": "rust_spec", "unit": uid, **spec})

    if not code:
        code = await generate_code(llm, units, specs, types_rs,
                                   cfg.rustgen_code_max_tokens, extras=extras)
        for uid, rust in code.items():
            store.write_record({"type": "rust_code", "unit": uid, "code": rust})

    # FFI shim layer: C-ABI exports for the file's public (non-static)
    # functions — the binding points for cando2 equivalence. Needs the C
    # source; re-chunked deterministically, no MTU re-run.
    if not ffi_rs and cfg.rustgen_ffi_enabled and src_path.exists():
        # bindgen over the C headers is the ABI authority for shim signatures
        # (the same decls the cando2 value.spec is generated from); {} when
        # bindgen/headers are unavailable — tree-sitter signature fallback.
        # The boundary surface adds the DATA contract: mirror structs +
        # conversion layer, because C reads the structs directly (lib_swap).
        abi, boundary = {}, None
        if cfg.c_source_root:
            from rustgen.ffi_stage import public_functions
            from rustgen.layout_stage import boundary_surface
            from rustgen.surface import abi_decls
            abi = abi_decls(Path(cfg.c_source_root), src_path.name,
                            cfg.bindgen_clang_args, cfg.bindgen_bin)
            names = [n for n, _ in public_functions(
                src_path.read_text(), cfg.split_function_over_lines)]
            boundary = boundary_surface(
                Path(cfg.c_source_root), src_path.name, names,
                out_dir / "bindgen", cfg.bindgen_clang_args, cfg.bindgen_bin)
            if not boundary[0]:
                boundary = None
        ffi_rs, exported = await generate_ffi(
            llm, src_path.read_text(), cfg.split_function_over_lines,
            types_rs, specs, cfg.rustgen_ffi_max_tokens, abi, boundary)
        store.write_record({"type": "rust_ffi", "exported": exported, "code": ffi_rs})

    # deps linkage: swap todo!() stubs for extern "C" adapters against the
    # real C symbols (resolved at load time — the incremental-port pattern).
    # NEVER re-linked on an already-linked crate (joint relink + FFI regen on
    # a converged crate is a documented regression: commit 1 error -> 121);
    # targeted repair is deps_stage.relink_todo_stubs.
    already_linked = types_rs is not None and 'extern "C"' in types_rs
    if cfg.rustgen_deps_mode == "extern" and cfg.c_source_root and not already_linked:
        types_rs, deps_report = await link_deps(
            llm, types_rs, Path(cfg.c_source_root), cfg.rustgen_deps_max_tokens,
            cfg.bindgen_clang_args, cfg.bindgen_bin)
        store.write_record({"type": "rust_types", "types_rs": types_rs,
                            "glossary": glossary, "deps_mode": "extern"})
        store.write_record({"type": "event", "event": "deps_linked", **deps_report})
        print(f"[deps] {out_dir.name}: {deps_report['linked']} extern, "
              f"{deps_report['native']} native, "
              f"{len(deps_report['unresolved'])} unresolved"
              + (f" ({', '.join(deps_report['unresolved'][:6])})"
                 if deps_report["unresolved"] else ""))

    crate_name = out_dir.name.replace(".", "_").replace("-", "_")
    src = Path(source_name).name
    crate = assemble(out_dir, crate_name, src, types_rs, units, code, ffi_rs)

    if cfg.rustgen_compile_rounds > 0:
        def reassemble(t_rs: str, c: dict, f_rs: str) -> None:
            assemble(out_dir, crate_name, src, t_rs, units, c, f_rs)

        pre_types, pre_ffi = types_rs, ffi_rs
        types_rs, code, ffi_rs, report = await compile_loop(
            cfg, llm, crate, units, specs, types_rs, code, reassemble, ffi_rs)
        for uid, rust in code.items():
            store.write_record({"type": "rust_code", "unit": uid, "round": "repaired",
                                "code": rust})
        # shared-section and FFI repairs must persist too, or the next
        # --resume silently reverts them (re-assembling old types with new
        # code once resurrected 40 already-fixed errors)
        if types_rs != pre_types:
            # label by what the module actually is, not the run's flag — a
            # stub-mode resume of a linked crate must not relabel it
            mode = "extern" if 'extern "C"' in types_rs else "stub"
            store.write_record({"type": "rust_types", "types_rs": types_rs,
                                "glossary": glossary, "deps_mode": mode,
                                "note": "compile-loop repaired"})
        if ffi_rs != pre_ffi:
            store.write_record({"type": "rust_ffi", "exported": exported,
                                "code": ffi_rs, "note": "compile-loop repaired"})
        store.write_record({"type": "compile_report", "rounds": report.rounds,
                            "final_errors": report.final_errors,
                            "clean_units": report.clean_units})
        print(f"[compile] {out_dir.name}: {report.summary()}")

    store.write_record(llm.usage_record())
    n_lines = (crate / "src" / "lib.rs").read_text().count("\n")
    ffi_note = f", {len(exported)} C symbols exported" if exported else ""
    print(f"[rustgen] {out_dir.name}: {len(units)} MTUs -> {crate / 'src/lib.rs'} "
          f"({n_lines} lines{ffi_note}), {llm.calls} calls, "
          f"{llm.input_tokens}+{llm.output_tokens} tokens, {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="MTU -> Rust generation")
    ap.add_argument("out_dir", type=Path, help="a completed run dir, e.g. out/tree-cache")
    ap.add_argument("--compile-rounds", type=int, default=None,
                    help="cargo check repair rounds (0 disables the loop)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse recorded stage outputs from state.jsonl; only "
                         "regenerate missing stages and run the compile loop")
    ap.add_argument("--spec-mode", choices=["thin", "rich"], default=None,
                    help="stage S mode (default: config)")
    ap.add_argument("--deps-mode", choices=["stub", "extern"], default=None,
                    help="deps stubs (todo!()) or extern C linkage")
    ap.add_argument("--c-root", type=Path, default=None,
                    help="C source tree for deps declaration lookup")
    ap.add_argument("--bindgen-args", default=None,
                    help="extra clang args for bindgen (ABI shim authority)")
    ap.add_argument("--regen-ffi", action="store_true",
                    help="with --resume: regenerate the FFI layer even if one "
                         "is recorded (e.g. to adopt the boundary conversion "
                         "layer); MTU code and types are untouched")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file overriding Config defaults, e.g. "
                         '{"rustgen_model": "gemma-4-31b"} (also read from '
                         "the DIFFUSIONMTUS_CONFIG env var)")
    args = ap.parse_args()
    asyncio.run(run(args.out_dir, compile_rounds=args.compile_rounds,
                    resume=args.resume, spec_mode=args.spec_mode,
                    deps_mode=args.deps_mode, c_root=args.c_root,
                    bindgen_args=args.bindgen_args, regen_ffi=args.regen_ffi,
                    config=args.config))


if __name__ == "__main__":
    main()
