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
    # Second instance of the SAME weights, tensor-parallel over the other two
    # GPUs. Two servers rather than two runs sharing one: vLLM interleaves
    # concurrent requests into the same batches, so runs sharing an instance
    # are coupled through batch composition — a confound when the quantity
    # being measured is run-to-run variance.
    "gemma-4-31b-b": {
        "base_url": "http://127.0.0.1:8002/v1",
        "api_key_file": "/nobackup2/alleshwaram/gemma4-31b-b-vllm/api_key.txt",
    },
    # Anthropic models via Google Vertex AI. NOT a vLLM server — `backend`
    # switches llm.py onto the Anthropic SDK, and base_url/api_key_file do not
    # apply. Auth is GCP ADC: point GOOGLE_APPLICATION_CREDENTIALS at the
    # service-account JSON.
    #
    # READ THIS BEFORE COMPARING A CLAUDE RUN TO A GEMMA ONE. The sampling
    # regime is NOT the same and cannot be made the same: this pipeline sends
    # temperature 1.0 / top_p 0.95 / top_k 64 (Gemma's own defaults), and
    # Claude 5-family models REJECT non-default top_p/top_k with a 400. They
    # are dropped for this backend, and adaptive thinking — which Gemma has no
    # equivalent for — is on by default. So a Claude-vs-Gemma run is each model
    # at ITS OWN defaults, not one variable changed. Say so in any write-up.
    "claude-sonnet-5": {
        "backend": "anthropic-vertex",
        "project_id": "cs-trustworthy-ai-43a8",
        "region": "global",
    },
    "claude-opus-5": {
        "backend": "anthropic-vertex",
        "project_id": "cs-trustworthy-ai-43a8",
        "region": "global",
    },
}


@dataclass
class Config:
    # --- models ---
    worker_model: str = "gemma-4-26b-a4b"  # --served-model-name of the local vLLM server
    max_tokens: int = 1024
    concurrency: int = 4  # vLLM batches concurrent requests; bounded by GPU KV cache

    # --- transport ---
    # Per-request ceiling. Generous, because a 31B model emitting a 16k-token
    # reply (LLM.MAX_TOKENS_CEILING, reached by the doubling path) under a
    # queued batch legitimately runs into the minutes — but bounded, because
    # the previous value of 3600 meant one wedged request stalled a whole run
    # for an hour before failing it.
    # FLOOR, not the bound: llm._request_timeout scales the actual per-request
    # wall clock with the token budget, because one fixed value cannot serve
    # both a 512-token call and a 16000-token one. At the slow end of the
    # observed decode rate the ceiling needs ~1070s, so 900 was unsatisfiable
    # for it — the request could never finish, whatever the server did.
    request_timeout: int = 900
    # The ONLY retry layer. The OpenAI client is built with max_retries=0: it
    # used to be 5, which silently multiplied request_timeout by six and
    # reported nothing, and is what wedged `array_list` t4 srvA. These retries
    # exist for a vLLM server being restarted — unreachable for minutes, then
    # fine. Backoff is exponential from `retry_backoff`, and every retry is
    # printed and counted into the run's `transport_retries`.
    transport_retries: int = 3
    retry_backoff: float = 5.0
    # Backstop on ONE logical LLM call end to end — every request it issues,
    # including budget doublings and transport retries. Each of those is
    # individually bounded; nothing bounded their product, which is how a
    # single call sat for 38 minutes writing nothing to the log. Generous
    # enough that only a pathological call reaches it.
    call_deadline: int = 2700

    # --- sampling ---
    # Sent explicitly on every request. Left unset these fall back to the
    # SERVER's generation_config.json (gemma-4-31b: temperature 1.0, top_p
    # 0.95, top_k 64), which means a run's record does not describe what was
    # actually sampled — and temperature 1.0 is why a single configuration
    # produced divergence counts of 1, 4, 6 and 9 on the same project.
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64

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
    # Ablation only. False restores the behaviour that shipped between the
    # project-scoped responsibility pass and 2026-07-31: every unit receives
    # EVERY concern it does not own as `must_not_implement`, which on
    # `array_list` was 17-29 entries per unit against 1.9 under the old
    # per-file pass. `array_list` reached `final: 0` in 0 of 8 arms with the
    # flood and 3 of 4 without it, but that is mechanism plus timing, never a
    # measurement — this knob is what makes the paired comparison possible.
    rustgen_scoped_prohibitions: bool = True
    # Ablation only, like the four rustgen_* knobs above: route a missing-item
    # error (E0599 `no method/associated function named X for T`) to the
    # section owning `impl T`, instead of leaving the cluster as the caller
    # alone. Suspected of causing `array_list base_srvB_t6`'s ten emptied-block
    # rejections on owner sections; set False to measure that. Ships True.
    rustgen_owner_routing: bool = True
    # Ablation only. When stage T's gates exhaust their retries, repair the
    # block instead of giving up on it. That branch is a death sentence as it
    # stands: over 255 recorded run logs every run reaching it died (4
    # BUILD_FAILED, 3 STUB_CRATE, none scored), because a stubbed shared block
    # hands every unit a phantom API to defer to. False restores the previous
    # behaviour — print, record the failure, continue with the bad block.
    # The repair can only turn a dead run into a live one; it returns the
    # original block on any failure. Ships True.
    rustgen_types_repair: bool = True
    # Ablation only. Resolve remaining `todo!()` in the finished crate rather
    # than letting `remaining_stubs` void the run. 21 runs have been voided by
    # that gate and 13 of them died on ONE OR TWO stubs, so the leverage is in
    # the shape of the failures, not in the share of stubs fixed. False skips
    # the pass entirely. Ships True.
    rustgen_stub_repair: bool = True
    # The CONTROL arm for the above, and the one that decides whether the loop
    # is worth its tokens. False makes the handler answer every question with
    # "no information available", turning the loop into a plain retry. If the
    # two score the same, the retrieval is doing nothing and the loop should
    # go. Ships True; only an ablation should set it False.
    rustgen_stub_context: bool = True
    rustgen_max_errors_per_section: int = 6  # errors quoted back per repair call
    rustgen_surgical_max_errors: int = 2     # sections at/below this get tier-1 edits first
    rustgen_deps_max_tokens: int = 1500      # sibling-stub resolution (sibling_deps.py)
    # oracle-free omission checks after the compile loop (semantic_check.py):
    # lost printf output text, unmapped/vanished C symbols. Diagnostic only —
    # recorded, never repaired. The repair loop that used to follow them cost
    # ~40% of a run's wall clock and produced byte-identical output, so only
    # the checks survive; they add ~20ms and no LLM calls.
    rustgen_semantic_check: bool = True
    # Ablation switches. All default True = the shipped behaviour; each one
    # removes exactly one prompt input so an A/B run can attribute its share
    # of the measured gain. They are knobs for experiments, not tuning:
    # nothing downstream should ever ship with one of these False.
    rustgen_call_sites: bool = True       # CALL SITES block (chunker.call_sites)
    rustgen_symbol_map: bool = True       # symbol_map in stage S + its checks
    rustgen_output_formats: bool = True   # OUTPUT FORMATS block
    rustgen_exit_status: bool = True      # EXIT STATUS block (exit_status.py)
    rustgen_callback_params: bool = True  # CALLBACK PARAMETERS block (bug class 7)
    # How much of the MTU's original C the spec/code stages may see:
    #   "off"      — description + invariants only (the MTU philosophy)
    #   "literals" — just the string/char/numeric literals from the unit's C
    #                lines (seed data, output formats, constants) — data
    #                fidelity without exposing C control flow or API shape
    #   "full"     — the raw C lines, labeled reference-only
    # Default "full": measured across 7 translations of 2 projects, "off" kept
    # 31-38% of the C's printable output fragments vs 73-90% for
    # literals/full. The MTU philosophy is about not transliterating CONTROL
    # FLOW; withholding the C's literal output text just loses data.
    rustgen_c_source_context: str = "full"

    # Names typedef'd to a function-pointer type, over the whole project.
    # DERIVED, not a setting: `run.py` fills it from the source file's
    # directory before discovery, and `run_project.py` recomputes it for the
    # rustgen stages. It lives here because the discovery strategies see only
    # a source STRING and a filename, never a path, and they need it to keep
    # `ArrayListCompareFunc compare_func` out of `external_deps`. Recorded with
    # the rest of the config, which also documents what each run actually saw.
    fn_ptr_typedefs: list[str] = field(default_factory=list)

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
