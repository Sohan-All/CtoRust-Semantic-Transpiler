#!/usr/bin/env python3
"""cando2 preflight — how much of our exported surface the value lane can test.

Runs the SAME machinery forclift's value.spec producer uses (bindgen over the
C headers, then forclift.engine.value_spec.generate_specs) and intersects the
resulting amenable catalog with the symbols our crate actually exports
(`rust_ffi` record). Three buckets per crate:

  AMENABLE     — the value lane will record/replay these symbols
  OUT_OF_SCOPE — declared, but excluded by the amenability rules (reason shown)
  NO_DECL      — exported but not declared in any bindgen-reachable header
                 (deprecated API / helper wrongly exported): invisible to the
                 value lane entirely

Usage:
    python3 tools/surface_report.py out/tree-cache out/pqueue ... \
        --c-root <tree> [--clang-args "..."] [--forclift <uwisc-docker dir>]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import Store
from rustgen.surface import header_for_source, run_bindgen


def catalog_for(c_root: Path, source_name: str, clang_args: str,
                bindgen_bin: str, generate_specs) -> dict | None:
    includes = []
    own = header_for_source(c_root, source_name)
    if own:
        includes.append(own)
    inc = c_root / "include"
    if inc.is_dir():
        includes += sorted(p.name for p in inc.glob("*.h"))
    with tempfile.TemporaryDirectory() as td:
        out = run_bindgen(c_root, includes, Path(td), clang_args, bindgen_bin)
        if out is None and own:
            out = run_bindgen(c_root, includes[1:], Path(td), clang_args, bindgen_bin)
        if out is None:
            return None
        return generate_specs(out.read_text(errors="replace"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", type=Path, nargs="+")
    ap.add_argument("--c-root", type=Path, required=True)
    ap.add_argument("--clang-args", default="")
    ap.add_argument("--bindgen-bin", default="bindgen")
    ap.add_argument("--forclift", type=Path,
                    default=Path.home() / "ctorust" / "uwisc-docker")
    args = ap.parse_args()

    sys.path.insert(0, str(args.forclift))
    from forclift.engine.value_spec import generate_specs

    grand = {"exported": 0, "amenable": 0, "out_of_scope": 0, "no_decl": 0}
    for out_dir in args.out_dirs:
        store = Store.load(out_dir)
        source_name, exported = "?", []
        for r in store.records:
            if r.get("type") == "event" and r.get("event") == "start":
                source_name = r.get("file", "?")
            elif r.get("type") == "rust_ffi":
                exported = r.get("exported", [])
        cat = catalog_for(args.c_root, Path(source_name).name, args.clang_args,
                          args.bindgen_bin, generate_specs)
        if cat is None:
            print(f"[{out_dir.name}] bindgen failed — no catalog")
            continue
        amen = {e["symbol"] for e in cat["amenable"]}
        oos = {e["symbol"]: e["reason"] for e in cat["out_of_scope"]}
        ours_amen = sorted(s for s in exported if s in amen)
        ours_oos = sorted(s for s in exported if s in oos)
        ours_nodecl = sorted(s for s in exported if s not in amen and s not in oos)
        grand["exported"] += len(exported)
        grand["amenable"] += len(ours_amen)
        grand["out_of_scope"] += len(ours_oos)
        grand["no_decl"] += len(ours_nodecl)
        print(f"[{out_dir.name}] {len(exported)} exported: "
              f"{len(ours_amen)} amenable, {len(ours_oos)} out-of-scope, "
              f"{len(ours_nodecl)} no-decl")
        for s in ours_oos:
            print(f"    OUT_OF_SCOPE {s}: {oos[s]}")
        for s in ours_nodecl:
            print(f"    NO_DECL      {s}")
    n = grand["exported"] or 1
    print(f"\nTOTAL {grand['exported']} exported — {grand['amenable']} amenable "
          f"({100 * grand['amenable'] // n}%), {grand['out_of_scope']} out-of-scope, "
          f"{grand['no_decl']} without header decl")


if __name__ == "__main__":
    main()
