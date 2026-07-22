# legacy/ — archived, not on the live path

These files belonged to the **single-file rustgen driver** (`run_rust.py`) and its
**FFI / C-ABI equivalence-testing** machinery. The current translator is
`run_project.py` (multi-file, the Docker entrypoint); its `phase_rustgen`
supersedes `run_rust.py`, and `rustgen/sibling_deps.py` supersedes the
single-file `deps_stage.py`. Nothing in `run.py` or `run_project.py` imports any
of these.

Archived here:

| file | role |
|------|------|
| `rustgen/run_rust.py` | old single-file MTU→Rust driver (`python3 -m rustgen.run_rust out/<stem>`) |
| `rustgen/deps_stage.py` | external-dep stubbing/linking (superseded by `sibling_deps.py`) |
| `rustgen/ffi_stage.py` | `extern "C"` shim / C-ABI export layer |
| `rustgen/layout_stage.py` | bindgen-derived struct-layout boundary surface |
| `rustgen/surface.py` | bindgen ABI authority (header parsing, decl rendering) |
| `tools/surface_report.py` | cando2 preflight (exported-surface coverage report) |

Config keys these consumed (`rustgen_ffi_enabled`, `rustgen_ffi_max_tokens`,
`rustgen_deps_mode`, `bindgen_bin`, `bindgen_clang_args`, `c_source_root`) were
removed from `config.Config` and are now accepted-but-ignored by `load_config`
(see `DEPRECATED_CONFIG_KEYS`).

## To restore
Move a file back to its original path, e.g. `mv legacy/rustgen/surface.py rustgen/`.
The modules import each other via `from rustgen.X import ...`, so they run again
only once returned to the `rustgen/` package (and the relevant config keys would
need un-deprecating).
