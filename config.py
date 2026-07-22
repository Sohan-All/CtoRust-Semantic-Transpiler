"""Pipeline configuration. Every threshold from SPEC.md lives here so A/B runs
can vary one knob at a time."""

import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields, replace
from pathlib import Path

# Env var a --config flag sets so subprocesses (run_project.py's per-file
# `run.py` workers) inherit the same model choice without re-passing --config.
CONFIG_ENV_VAR = "DIFFUSIONMTUS_CONFIG"

# Keys that configured the FFI / single-file rustgen path now archived under
# legacy/. Accepted-but-ignored in load_config so an old config JSON that still
# names them is a warning, not a hard error.
DEPRECATED_CONFIG_KEYS = {
    "rustgen_ffi_enabled", "rustgen_ffi_max_tokens", "rustgen_deps_mode",
    "bindgen_bin", "bindgen_clang_args", "c_source_root",
}

# served-model-name -> which vLLM server instance serves it. Add an entry
# here whenever a new model is stood up so worker_model/rustgen_model in a
# config file resolve to the right endpoint automatically.
MODEL_SERVERS = {
    "gemma-4-26b-a4b": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_file": "/nobackup2/alleshwaram/gemma4-vllm/api_key.txt",
    },
    "gemma-4-31b": {
        "base_url": "http://127.0.0.1:8001/v1",
        "api_key_file": "/nobackup2/alleshwaram/gemma4-31b-vllm/api_key.txt",
    },
}


@dataclass
class Config:
    # --- models ---
    worker_model: str = "gemma-4-26b-a4b"  # --served-model-name of the local vLLM server
    max_tokens: int = 1024
    concurrency: int = 4  # vLLM batches concurrent requests; bounded by GPU KV cache

    # --- syntax seeding ---
    split_function_over_lines: int = 40  # functions longer than this split one level down

    # --- diffusion strategy termination ---
    max_merge_passes: int = 6           # safety net; fixpoint is the primary stop
    staleness_limit: int = 2            # failed lock attempts w/o change -> irreducible
    size_guard_lines: int = 300         # open unit larger than this that won't lock -> irreducible
    merge_soft_cap_lines: int = 200     # don't create merged units larger than this
    max_invariant_rephrases: int = 2    # per invariant
    max_repairs: int = 2                # per unit: answer probe questions from source

    # --- whole-file strategy ---
    whole_file_max_lines: int = 2000    # hard cap for strategy B's single-call partition
    whole_file_max_retries: int = 3     # coverage-validator retries

    # --- rustgen (MTU -> Rust) ---
    rustgen_model: str = "gemma-4-26b-a4b"
    rustgen_types_max_tokens: int = 4000   # stage T: shared type definitions
    rustgen_spec_max_tokens: int = 1500    # stage S thin mode: signatures per MTU
    rustgen_spec_mode: str = "rich"        # "thin" (signatures only) | "rich" (full Rust spec)
    rustgen_rich_spec_max_tokens: int = 3000
    rustgen_code_max_tokens: int = 4000    # stage C: function bodies per MTU
    rustgen_compile_rounds: int = 5        # cargo check -> repair iterations
                                           # (loop exits early at 0 errors; 5
                                           # absorbs the richer v3 API surface)
    rustgen_repair_max_tokens: int = 4000
    rustgen_max_errors_per_section: int = 6  # errors quoted back per repair call
    rustgen_surgical_max_errors: int = 2     # sections at/below this get tier-1 edits first
    rustgen_deps_max_tokens: int = 1500      # sibling-stub resolution (sibling_deps.py)
    # How much of the MTU's original C the spec/code stages may see:
    #   "off"      — description + invariants only (the MTU philosophy)
    #   "literals" — just the string/char/numeric literals from the unit's C
    #                lines (seed data, output formats, constants) — data
    #                fidelity without exposing C control flow or API shape
    #   "full"     — the raw C lines, labeled reference-only
    rustgen_c_source_context: str = "off"

    # --- lock check ---
    lock_regex_enabled: bool = True
    # c_mention uses per-item ALLOWED/FLAGGED classification with worked
    # examples — detection-style prompting made Haiku over-flag; classification
    # scores 0 FP / 100% recall on tests/judge_eval.py
    lock_c_mention_enabled: bool = True
    lock_round_trip_enabled: bool = True
    lock_round_trip_max_open: int = 3   # OPEN behavior/format questions the probe
                                        # tolerates before failing a unit (lower =
                                        # stricter; also gates whether arbitration runs)

    def to_record(self) -> dict:
        return {"type": "config", **asdict(self)}


DEFAULT = Config()


def load_config(path: str | Path | None = None) -> Config:
    """DEFAULT, optionally overlaid with a JSON file's fields (e.g. to pick
    worker_model/rustgen_model). `path` falls back to the DIFFUSIONMTUS_CONFIG
    env var, so a --config flag set once at the top of a run propagates into
    subprocesses spawned along the way."""
    path = path or os.environ.get(CONFIG_ENV_VAR)
    if not path:
        return Config()
    data = json.loads(Path(path).read_text())
    for k in DEPRECATED_CONFIG_KEYS & set(data):
        print(f"{path}: ignoring deprecated config key {k!r} "
              "(FFI / single-file rustgen path moved to legacy/)", file=sys.stderr)
        data.pop(k)
    unknown = set(data) - {f.name for f in fields(Config)}
    if unknown:
        raise SystemExit(f"{path}: unknown config key(s): {', '.join(sorted(unknown))}")
    # legacy bool form of rustgen_c_source_context (pre-"literals" mode)
    if isinstance(data.get("rustgen_c_source_context"), bool):
        data["rustgen_c_source_context"] = (
            "full" if data["rustgen_c_source_context"] else "off")
    mode = data.get("rustgen_c_source_context")
    if mode is not None and mode not in ("off", "literals", "full"):
        raise SystemExit(f"{path}: rustgen_c_source_context must be "
                         f"off|literals|full, got {mode!r}")
    return replace(Config(), **data)
