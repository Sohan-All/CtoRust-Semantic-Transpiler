#!/usr/bin/env python3
"""Multi-file project translation orchestrator (cross-file bridging).

    python3 run_project.py <c_src_dir> [--phase index|mtu|types|all]

Phases (each resumable; state lives in out/<file-stem>/ per file plus
out/_project_<name>/ for project-level records):

  index   — deterministic: project symbol index, shared types, file-SCC
            translation order (project_index.py). Printed + recorded.
  mtu     — per-file MTU discovery (existing run.py pipeline), all files in
            parallel; skips files whose out/<stem>/state.jsonl is complete.
  types   — project Stage T: ONE canonical Rust definition per shared type
            (header decls = exact fields, MTU descriptions = meaning),
            recorded in the project state.
  (rustgen phase lands next: per-file generation in SCC-group order with
   shared types + sibling signatures injected; siblings become plain Rust
   calls, extern "C" is reserved for the project's true public boundary.)
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from config import Config, CONFIG_ENV_VAR, load_config
from llm import LLM
from project_index import ProjectIndex, build_index
from state import Store

ROOT = Path(__file__).parent
# DIFFUSIONMTUS_OUT lets a sandboxed run (repo mounted read-only) redirect all
# state/artifacts to a writable volume; default stays repo-local.
OUT = Path(os.environ.get("DIFFUSIONMTUS_OUT") or (ROOT / "out"))

# Exit status for an unhandled exception in a phase, distinct from argparse's 2
# and from a deliberate SystemExit, so the harness can report CRASHED rather
# than folding every nonzero rc into one verdict.
CRASH_EXIT = 3


def stem_of(fname: str) -> str:
    return Path(fname).stem


def project_dir(c_root: Path) -> Path:
    d = OUT / f"_project_{c_root.parent.parent.name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(pdir: Path, rec: dict) -> None:
    rec = {"ts": time.time(), **rec}
    with (pdir / "project.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _latest(pdir: Path, rtype: str) -> dict | None:
    path = pdir / "project.jsonl"
    if not path.exists():
        return None
    out = None
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == rtype:
            out = r
    return out


def files_dir(c_root: Path) -> Path:
    d = project_dir(c_root) / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mtu_complete(c_root: Path, fname: str) -> bool:
    d = files_dir(c_root) / stem_of(fname)
    if not (d / "state.jsonl").exists():
        return False
    try:
        return bool(Store.load(d).final_units())
    except Exception:
        return False


def phase_index(c_root: Path) -> ProjectIndex:
    idx = build_index(c_root)
    pdir = project_dir(c_root)
    _record(pdir, idx.to_record())
    print(f"[index] {len(idx.files)} files, "
          f"{len(idx.shared_types)} shared types, "
          f"{len(idx.groups)} translation groups")
    for i, g in enumerate(idx.groups):
        print(f"  group {i}: {', '.join(g)}"
              + ("  (CO-TRANSLATE)" if len(g) > 1 else ""))
    return idx


def phase_mtu(c_root: Path, idx: ProjectIndex) -> None:
    """Per-file MTU discovery, in parallel (discovery is file-independent —
    ordering only matters from rustgen onward)."""
    todo = [f for f in idx.files if not mtu_complete(c_root, f)]
    if not todo:
        print("[mtu] all files already discovered")
        return
    print(f"[mtu] discovering {len(todo)} files: {', '.join(todo)}")
    procs = {}
    for f in todo:
        log = project_dir(c_root) / f"mtu_{stem_of(f)}.log"
        procs[f] = (subprocess.Popen(
            [sys.executable, str(ROOT / "run.py"), str(c_root / f),
             "--strategy", "whole_file", "--out", str(files_dir(c_root))],
            stdout=log.open("w"), stderr=subprocess.STDOUT, cwd=ROOT), log)
    for f, (p, log) in procs.items():
        rc = p.wait()
        status = "ok" if rc == 0 and mtu_complete(c_root, f) else f"FAILED (rc={rc}, see {log})"
        print(f"[mtu] {f}: {status}")


async def phase_types(c_root: Path, idx: ProjectIndex) -> tuple[str, dict]:
    from rustgen.project_types import synthesize_project_types
    descriptions: dict[str, list[str]] = {}
    for f in idx.files:
        d = files_dir(c_root) / stem_of(f)
        try:
            units = Store.load(d).final_units()
        except Exception:
            units = []
        descriptions[f] = [u.text for u in units]

    base = load_config()
    cfg = dataclasses.replace(base, worker_model=base.rustgen_model)
    llm = LLM(cfg)
    shared_rs, glossary = await synthesize_project_types(
        llm, idx, descriptions, cfg.rustgen_types_max_tokens)
    pdir = project_dir(c_root)
    _record(pdir, {"type": "project_types", "shared_types_rs": shared_rs,
                   "glossary": glossary})
    _record(pdir, llm.usage_record())
    n = len(re.findall(r"\bpub(?:\(crate\))? (?:struct|enum) ", shared_rs))
    print(f"[types] {n} shared definitions, glossary {len(glossary)} entries "
          f"-> {pdir / 'project.jsonl'}")
    return shared_rs, glossary




async def phase_rustgen(c_root: Path, idx: ProjectIndex) -> None:
    """Per-file generation in SCC-group order with cross-file bridging:
    shared project types prepended to every file's type context, sibling
    signatures from already-generated files injected as callable context,
    then one integrated crate + one compile loop over all units.

    Sibling deps are NOT extern "C" — they are direct Rust calls (the
    project's only real boundary is the CLI binary the vectors drive)."""
    from rustgen.types_stage import synthesize_types
    from rustgen.spec_stage import generate_specs
    from rustgen.code_stage import generate_code
    from rustgen.assemble import assemble_project
    from rustgen.compile_loop import compile_loop
    import dataclasses as dc

    pdir = project_dir(c_root)
    ptypes = _latest(pdir, "project_types")
    if ptypes is None:
        raise SystemExit("run --phase types first")
    shared_rs = ptypes["shared_types_rs"]
    pglossary = ptypes.get("glossary", {})

    base = load_config()
    cfg = dataclasses.replace(base, worker_model=base.rustgen_model)
    llm = LLM(cfg)

    # entry function: the file defining C main is the binary wrapper's
    # target; its MTU code is generated like any other unit
    main_file = idx.defines.get("main")

    registry: list[str] = []          # rendered sibling sigs, growing
    all_units, all_specs, all_code = [], {}, {}
    c_lines_by_unit: dict[str, list[str]] = {}   # prefixed unit id -> its file's lines
    proj_types_parts = [shared_rs]
    # Every stage-level fallback appends here; a non-empty list becomes a
    # `degraded` record and the harness refuses to score the run. Surviving a
    # failed unit is only safe if the resulting crate cannot be mistaken for a
    # complete one — assemble() fills a missing unit with a *documented*
    # todo!(), which illegal_stubs() deliberately permits, so the stub gate
    # would pass it. This record is what makes the loss visible.
    degraded: list[dict] = []

    xcalls = project_call_texts(c_root, idx, cfg.split_function_over_lines)
    xstatus = (project_exit_status(c_root, idx) if cfg.rustgen_exit_status
               else {})

    order = [f for g in idx.groups for f in g]
    for fname in order:
        stem = stem_of(fname)
        store = Store.load(files_dir(c_root) / stem)
        units = store.final_units()
        if not units:
            print(f"[rustgen] {fname}: no units, skipped")
            continue

        prev_types = None
        for r in store.records:
            if r.get("type") == "rust_types" and r.get("project") == True:
                prev_types = (r.get("types_rs"), r.get("glossary", {}))
        if prev_types:
            file_types, glossary = prev_types
        else:
            file_types, glossary = await synthesize_types(
                llm, units, cfg.rustgen_types_max_tokens,
                project_block=_project_block(shared_rs, pglossary, registry,
                                             idx, fname))
            store.write_record({"type": "rust_types", "types_rs": file_types,
                                "glossary": glossary, "project": True})

        # deterministic namespacing: every file's `pub mod deps` becomes
        # `pub mod <stem>_deps` and its units' `deps::` refs follow — flat
        # assembly would otherwise collide every file's deps module (E0428
        # churn previously absorbed nondeterministically by the compile loop)
        mod_ns = stem.replace("-", "_") + "_deps"
        file_types = re.sub(r"\bpub(?:\(crate\))? mod deps\b",
                            f"pub mod {mod_ns}", file_types)
        # same reason as the rename: deterministic cleanup of stage T's text
        # before it reaches a compile loop that has no write access to it
        from rustgen.common import demote_dangling_docs
        file_types = demote_dangling_docs(file_types)

        # context for spec/code: shared types + this file's types + sibling
        # signatures (rendered as comments — the stages treat types_rs as
        # opaque context, so this needs no stage-signature changes)
        ctx_types = shared_rs + "\n\n" + file_types
        if registry:
            ctx_types += ("\n\n// ===== SIBLING MODULE FUNCTIONS (implemented"
                          " in this crate — call directly) =====\n"
                          + "\n".join(f"// {s}" for s in registry))

        if fname == main_file:
            ctx_types += (
                "\n\n// NOTE: this file is the PROGRAM ENTRY. Emit exactly one"
                "\n// function: `pub fn app_main(args: &[String]) -> i32` —"
                "\n// convert the args and delegate to the sibling entry"
                "\n// function listed above; implement no other behavior."
                "\n// ARGV CONVENTION (binding on the whole arg-handling call"
                "\n// chain): args mirrors C argv — args[0] is the program"
                "\n// path, the first real argument (subcommand/flag) is"
                "\n// args[1]. Never match a command against args[0].")

        # per-unit context: reverse call graph — within-file edges from the
        # chunker, cross-file edges from the project index (a unit that knows
        # cli.c calls it designs a signature cli.c can call, and doesn't end
        # up as dead code) — plus raw C lines when rustgen_c_source_context
        from rustgen.common import unit_extras
        ext_callers = {fn: sorted(g for g in idx.files if g != fname
                                  and fn in idx.references.get(g, ()))
                       for fn, owner in idx.defines.items() if owner == fname}
        file_source = (c_root / fname).read_text()
        extras = unit_extras(units, file_source,
                             cfg.split_function_over_lines,
                             cfg.rustgen_c_source_context,
                             external_callers={fn: fs for fn, fs
                                               in ext_callers.items() if fs},
                             external_call_texts=xcalls.get(fname),
                             exit_status=xstatus.get(fname),
                             call_sites=cfg.rustgen_call_sites,
                             symbol_map=cfg.rustgen_symbol_map,
                             output_formats=cfg.rustgen_output_formats)

        # Both stages persist per unit as it lands (the on_result callbacks
        # below) rather than after the parallel barrier. Writing after the
        # barrier meant one unit raising discarded every sibling that had
        # already succeeded, and the resume path — keyed on "did this stage
        # produce anything at all" — then redrew the whole file. `skip` is what
        # makes the saved work count: a resume regenerates only the units with
        # no record, instead of all of them.
        specs = {}
        for r in store.records:
            if r.get("type") == "rust_spec" and r.get("project") == True:
                specs[r["unit"]] = {k: v for k, v in r.items()
                                    if k not in ("type", "ts", "unit", "project")}
        if set(specs) != {u.id for u in units}:
            fresh = await generate_specs(
                llm, units, ctx_types, glossary, cfg, extras=extras,
                skip=set(specs), failures=degraded,
                on_result=lambda uid, spec: store.write_record(
                    {"type": "rust_spec", "unit": uid, "project": True, **spec}))
            specs.update(fresh)

        code = {}
        for r in store.records:
            if r.get("type") == "rust_code" and r.get("project") == True:
                code[r["unit"]] = r.get("code", "")
        # only units that actually have a spec are codeable
        want_code = {u.id for u in units} & set(specs)
        if set(code) != want_code:
            fresh = await generate_code(
                llm, [u for u in units if u.id in want_code], specs, ctx_types,
                cfg.rustgen_code_max_tokens, extras=extras,
                skip=set(code), failures=degraded,
                on_result=lambda uid, rust: store.write_record(
                    {"type": "rust_code", "unit": uid, "project": True,
                     "code": rust}))
            code.update(fresh)

        # a unit whose spec or code never materialised is missing behaviour the
        # crate will not obviously lack — siblings stub it and it still builds
        missing = [u.id for u in units if u.id not in code]
        if missing:
            degraded.append({"stage": "assemble", "file": fname,
                             "error": f"{len(missing)} unit(s) with no code: "
                                      f"{', '.join(sorted(missing))}"})

        for spec in specs.values():
            registry.extend(f"[{stem}] {s}" for s in spec.get("signatures", []))
        proj_types_parts.append(f"// ----- {fname} local types -----\n{file_types}")

        code = {k: re.sub(r"(?<![A-Za-z0-9_])deps::", f"{mod_ns}::", v)
                for k, v in code.items()}
        prefix = stem.replace("-", "_") + "__"
        file_lines = file_source.split("\n")
        for u in units:
            all_units.append(dc.replace(u, id=prefix + u.id))
            # a unit's ranges index ITS OWN file, so the semantic check needs
            # that file's lines kept alongside the prefixed id
            c_lines_by_unit[prefix + u.id] = file_lines
        all_specs.update({prefix + k: v for k, v in specs.items()})
        all_code.update({prefix + k: v for k, v in code.items()})
        print(f"[rustgen] {fname}: {len(units)} units, "
              f"{len(specs)} specs, {len(code)} code blocks")

    entry_fn = None
    if main_file and any(u.startswith(stem_of(main_file).replace("-", "_") + "__")
                         and "pub fn app_main" in all_code.get(u, "")
                         for u in all_code):
        entry_fn = "app_main"
    elif main_file:
        stem = stem_of(main_file).replace("-", "_")
        for uid, spec in all_specs.items():
            if uid.startswith(stem + "__"):
                for s in spec.get("signatures", []):
                    m = re.search(r"pub(?:\(crate\))? fn (\w+)", s)
                    if m:
                        entry_fn = m.group(1)
                        break
    repaired = _latest(pdir, "project_types_all")
    types_all = (repaired["types_rs"] if repaired
                 else "\n\n".join(proj_types_parts))

    from rustgen.sibling_deps import resolve_sibling_stubs
    types_all, sib_report = await resolve_sibling_stubs(
        llm, types_all, registry, shared_rs, cfg.rustgen_deps_max_tokens)
    if sib_report["total"]:
        _record(pdir, {"type": "project_types_all", "types_rs": types_all,
                       "note": "sibling stubs resolved"})
        _record(pdir, {"type": "event", "event": "sibling_deps", **sib_report})
        print(f"[siblings] {len(sib_report['resolved'])} resolved, "
              f"{len(sib_report['kept'])} kept as todo "
              f"({', '.join(sib_report['kept'][:5])})" if sib_report["kept"]
              else f"[siblings] {len(sib_report['resolved'])} resolved, 0 kept")
    crate_name = c_root.parent.parent.name
    crate = assemble_project(pdir, crate_name, str(c_root), types_all,
                             all_units, all_code, entry_fn=entry_fn)
    print(f"[assemble] {crate}/src/lib.rs "
          f"({(crate / 'src/lib.rs').read_text().count(chr(10))} lines), "
          f"entry: {entry_fn}")

    if cfg.rustgen_compile_rounds > 0:
        def reassemble(t_rs, c, f_rs):
            assemble_project(pdir, crate_name, str(c_root), t_rs,
                             all_units, c, entry_fn=entry_fn)
        types_all, all_code, _, report = await compile_loop(
            cfg, llm, crate, all_units, all_specs, types_all, all_code,
            reassemble, "")
        # persist repairs to the per-file stores or the next resume silently
        # reverts them (run_rust learned this the hard way; same rule here)
        for uid, rust in all_code.items():
            stem, _, base = uid.partition("__")
            sdir = files_dir(c_root) / stem
            if not sdir.exists():
                sdir = files_dir(c_root) / stem.replace("_", "-")
            Store.load(sdir).write_record(
                {"type": "rust_code", "unit": base, "project": True,
                 "code": rust, "note": "compile-loop repaired (project)"})
        _record(pdir, {"type": "project_compile", "rounds": report.rounds,
                       "final_errors": report.final_errors,
                       "stub_sections": report.stub_sections})
        # the loop repairs the TYPES section too — must persist or every
        # resume replays ~170 type errors and re-repairs them differently
        _record(pdir, {"type": "project_types_all", "types_rs": types_all})
        _record(pdir, {"type": "project_assembly", "types_rs": types_all,
                       "code": all_code, "entry_fn": entry_fn})
        print(f"[compile] project: {report.summary()}")

    # Semantic checks run AFTER the compile loop, on the crate as shipped.
    # They only REPORT: the crate is not touched, so nothing here can
    # reintroduce a type error and no re-compile is needed.
    if cfg.rustgen_semantic_check:
        from rustgen.semantic_check import semantic_report

        sem = semantic_report(cfg, all_units, all_specs, all_code,
                              c_lines_by_unit, types_all)
        _record(pdir, {"type": "project_semantic", "findings": sem.findings,
                       "hints": sem.hints, "total": sem.total})
        print(f"[semantic] project: {sem.summary()}")
    _record(pdir, llm.usage_record())

    # Last, so it covers every stage above. Written only when something was
    # actually lost — an absent record means a clean run, which is what the
    # harness gate reads.
    if degraded:
        _record(pdir, {"type": "degraded", "count": len(degraded),
                       "failures": degraded})
        print(f"[degraded] {len(degraded)} unit(s)/stage(s) failed and were "
              f"dropped — this crate is INCOMPLETE and must not be scored:")
        for d in degraded:
            print(f"  - {d['stage']} {d.get('unit') or d.get('file')}: "
                  f"{d['error']}")


def project_call_texts(c_root: Path, idx: ProjectIndex,
                       split_over: int) -> dict[str, dict[str, list[str]]]:
    """{defining file: {C function: call expressions used by OTHER files}}.

    One pass over the project (~8ms for 5-7 files) so each file's units can be
    told how their siblings actually invoke them. Within a file the chunker
    already supplies this; across files the caller was previously reduced to
    its filename, which drops the arguments — and the arguments are the
    contract (C's parse_options(argc, argv, 2, &o) vs (argc, argv, 3, &o)).
    """
    from chunker import chunk

    per_file: dict[str, dict[str, set[str]]] = {}
    for fname in idx.files:
        sites: dict[str, set[str]] = {}
        for b in chunk((c_root / fname).read_text(), split_over).blocks:
            for callee, texts in (getattr(b, "call_sites", {}) or {}).items():
                sites.setdefault(callee, set()).update(texts)
        per_file[fname] = sites

    out: dict[str, dict[str, list[str]]] = {f: {} for f in idx.files}
    for fn, owner in idx.defines.items():
        for caller_file, sites in per_file.items():
            if caller_file != owner and fn in sites:
                out[owner].setdefault(fn, []).extend(sites[fn])
    return {f: {fn: sorted(set(t)) for fn, t in m.items()} for f, m in out.items()}


def project_exit_status(c_root: Path, idx: ProjectIndex) -> dict[str, dict]:
    """{defining file: {C function: ExitStatus}} for functions whose return
    value becomes the process exit status.

    One pass over the project, like project_call_texts, because the flow
    crosses files: `main` in main.c returns `cli_run`'s value from cli.c. See
    exit_status.py for why this is return-flow and not the call graph.

    Only "interesting" functions are kept — those able to return a code outside
    {0, 1}. A function returning just 0/1 is already served correctly by the
    idiomatic `Result` -> `Err(_) => 1` mapping, so telling its unit anything
    would be prompt noise. Across B03_organic this keeps 21 of 1162 C functions.
    """
    from exit_status import analyze_project, interesting

    sources = {f: (c_root / f).read_text(errors="replace") for f in idx.files}
    out: dict[str, dict] = {f: {} for f in idx.files}
    for (fname, fn), info in analyze_project(sources).items():
        if interesting(info) and fname in out:
            out[fname][fn] = info
    return out


def _project_block(shared_rs: str, pglossary: dict, registry: list[str],
                   idx: ProjectIndex, fname: str) -> str:
    sibling_fns = sorted(n for n in idx.references.get(fname, ())
                         if idx.dep_class(fname, n) == "SIBLING")
    block = ("SHARED PROJECT TYPES (already defined at project level — build "
             "against these, do NOT redefine them; define ONLY types local to "
             "this file):\n```rust\n" + shared_rs + "\n```\n"
             "The project error enum above is the ONLY error type in this "
             "crate. Do NOT define a new error enum for this file, even where "
             "the instructions below ask for one — all fallible operations "
             "return `Result<T, <the project error enum>>`. If this file has "
             "a failure mode with no matching variant, use the closest "
             "existing variant and mark the line with a "
             "`// MISSING ERROR VARIANT: <name>` comment.\n")
    if pglossary:
        block += "PROJECT GLOSSARY: " + json.dumps(pglossary) + "\n"
    if sibling_fns:
        block += ("These functions are implemented by SIBLING FILES of this "
                  "project and will be directly callable — do NOT stub them "
                  "in `pub mod deps`: " + ", ".join(sibling_fns) + "\n")
    return block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("c_root", type=Path)
    ap.add_argument("--phase", choices=["index", "mtu", "types", "rustgen", "all"],
                    default="all")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file overriding Config defaults, e.g. "
                         '{"worker_model": "gemma-4-31b", "rustgen_model": '
                         '"gemma-4-31b"}. Propagated to the mtu phase\'s '
                         "per-file run.py subprocesses via DIFFUSIONMTUS_CONFIG.")
    args = ap.parse_args()
    if args.config:
        # resolved: the mtu phase passes this to run.py subprocesses that run
        # with cwd=ROOT, so a relative path works for the parent and breaks
        # every child. The failure then surfaced three layers away as
        # "no state.jsonl", with nothing pointing at the config path.
        if not args.config.exists():
            raise SystemExit(f"No such config file: {args.config}")
        os.environ[CONFIG_ENV_VAR] = str(args.config.resolve())
    c_root = args.c_root.resolve()

    # Each phase is wrapped so an unhandled exception leaves a record and a
    # distinguishable exit status instead of just a traceback on stdout. The
    # harness previously saw only a nonzero rc and filed every cause — a crash,
    # a missing arg, a dead server — as one TRANSLATE_FAILED.
    def phase(name: str, fn):
        try:
            return fn()
        except SystemExit:
            raise                      # deliberate abort (e.g. "run --phase types first")
        except KeyboardInterrupt:
            print(f"\n[{name}] interrupted")
            raise
        except BaseException as e:
            _record(project_dir(c_root),
                    {"type": "crash", "phase": name,
                     "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()})
            print(f"[{name}] CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(CRASH_EXIT)

    idx = phase("index", lambda: phase_index(c_root))
    if args.phase == "index":
        return
    if args.phase in ("mtu", "all"):
        phase("mtu", lambda: phase_mtu(c_root, idx))
    if args.phase in ("types", "all"):
        phase("types", lambda: asyncio.run(phase_types(c_root, idx)))
    if args.phase in ("rustgen", "all"):
        phase("rustgen", lambda: asyncio.run(phase_rustgen(c_root, idx)))


if __name__ == "__main__":
    main()
