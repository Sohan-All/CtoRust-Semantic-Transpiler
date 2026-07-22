"""Assembly — deterministic, no LLM. Emits the crate."""

from __future__ import annotations

from pathlib import Path

from state import Explanation, IRREDUCIBLE

CARGO_TOML = """\
[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
crate-type = ["cdylib", "rlib"]
"""


def assemble(out_dir: Path, crate_name: str, source_name: str, types_rs: str,
             units: list[Explanation], code: dict[str, str],
             ffi_rs: str = "") -> Path:
    crate = out_dir / "rust_crate"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(CARGO_TOML.format(name=crate_name))

    parts = [
        f"//! Generated from {source_name} via the MTU pipeline (validation-free v1).",
        "//! Each section maps to one MTU; provenance ranges refer to the C source.",
        "",
        "// ===== shared data model (stage T) =====",
        types_rs,
        "",
    ]
    for u in sorted(units, key=Explanation.sort_key):
        ranges = ", ".join(f"{s}-{e}" for s, e in u.ranges)
        parts.append(f"// ===== MTU {u.id} (C lines {ranges}) =====")
        if u.status == IRREDUCIBLE:
            parts.append("// LOW CONFIDENCE (irreducible MTU): the language-agnostic")
            parts.append("// description of this unit did not pass the lock check.")
        parts.append(code.get(u.id, f'todo!("no code generated for {u.id}");'))
        parts.append("")

    if ffi_rs:
        parts.append("// ===== FFI shims (C-ABI export layer for equivalence testing) =====")
        parts.append(ffi_rs)
        parts.append("")

    lib = crate / "src" / "lib.rs"
    lib.write_text(_dedupe_top_level_uses("\n".join(parts)))
    return crate


def _dedupe_top_level_uses(lib_rs: str) -> str:
    """Sections are generated independently, so two of them importing the
    same item (`use std::cmp::Ordering;` twice at module scope) is an E0252
    the compile loop then has to burn a round on. Drop exact repeats of
    column-0 `use` lines, keeping the first occurrence. Indented `use` lines
    (inside mods/fns) are separate scopes and left alone."""
    seen: set[str] = set()
    out = []
    for line in lib_rs.split("\n"):
        if line.startswith("use ") and line.rstrip().endswith(";"):
            key = line.rstrip()
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


BIN_CARGO_TOML = """\
[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[[bin]]
name = "{bin_name}"
path = "src/main.rs"
"""

MAIN_RS = """\
//! Binary entry point — the project's real public boundary is this CLI
//! (equivalence = this binary vs the C binary over the test vectors).
fn main() {{
    let args: Vec<String> = std::env::args().collect();
    std::process::exit({crate_name}::{entry_fn}(&args));
}}
"""


def assemble_project(out_dir: Path, crate_name: str, source_name: str,
                     types_rs: str, units: list[Explanation],
                     code: dict[str, str], bin_name: str = "driver",
                     entry_fn: str | None = None) -> Path:
    """Integrated multi-file crate: flat lib (shared types + every file's
    units as sections; sibling calls are direct — C's extern namespace is
    already flat) + a bin wrapper when `entry_fn` names the CLI entry."""
    crate = assemble(out_dir, crate_name, source_name, types_rs, units, code)
    if entry_fn:
        (crate / "Cargo.toml").write_text(
            BIN_CARGO_TOML.format(name=crate_name, bin_name=bin_name))
        (crate / "src" / "main.rs").write_text(
            MAIN_RS.format(crate_name=crate_name, entry_fn=entry_fn))
        # items default to pub(crate), but the bin target is a separate crate
        # — the entry point must be genuinely pub for main.rs to reach it
        lib = crate / "src" / "lib.rs"
        lib.write_text(lib.read_text().replace(
            f"pub(crate) fn {entry_fn}(", f"pub fn {entry_fn}("))
    return crate
