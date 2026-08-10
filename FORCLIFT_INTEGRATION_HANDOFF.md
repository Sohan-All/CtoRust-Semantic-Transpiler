# Handoff — Integrating the diffusionMTUs translator into `forclift/`

**Goal:** make the diffusionMTUs MTU pipeline a first-class *translator* inside
`uwisc-docker/forclift/` (the v2 architecture), reusing forclift's existing
structure — its Artifact/Producer/Scheduler/CheckSet machinery — rather than
bolting a second orchestrator alongside it.

**Status when this was written:** greenfield on branch `p02-updates-generalization`.
Nothing in `forclift/` references diffusionMTUs yet. (A *different* branch had a
`c2rust`-side `TRANSLATOR=mtu` → `c2rust/translators/mtu_translate` hook — useful
prior art, but that's the older c2rust PhaseFn world, **not** the forclift v2 target
here.)

Paths below are relative to `/nobackup2/alleshwaram/CtoRust/uwisc-docker/` unless noted.
The pipeline itself lives at `/nobackup2/alleshwaram/CtoRust/diffusionMTUs/`.

---

## 1. The one-sentence design

**diffusionMTUs becomes an alternative producer of the `rust.src` artifact** — it
*generates* `translated_rust/` from the C source instead of reading a
pre-existing tree — and forclift's existing checks (`value`, `bineq`, `abi`,
`quality`) validate it unchanged.

Everything else follows from that sentence.

## 2. Why this is the right seam (and why it's safe)

- **Translation output in forclift is the `rust.src` artifact.** Today it's a
  *declared-input* producer that just reads `ctx.workspace.rust_dir`
  (`forclift/producers/inputs.py:138-155`, kind `"rust.src"`, validated by
  `validate_cargo_workspace`). `ctx.workspace.rust_dir` = `<project>/translated_rust/`
  (`forclift/core/context.py:24-46`).
- **Overriding a producer kind is the sanctioned extension point** —
  `standard_producers()` says so verbatim: *"targets/plans may override individual
  kinds (the sanctioned swap point for alternative producers)"*
  (`forclift/producers/__init__.py:1-3`). So we register a producer that builds
  `translated_rust/` and swap it in for a `--translator mtu` run.
- **Trust model makes an alternative translator safe by construction.** Per
  `forclift/producers/agentic.py:5-12` and `docs/plans/09-translator-integration.md`:
  a generator *shapes the surface, never encodes the answer — C is the oracle*, so a
  bad translation can only UNDER-perform (checks read N/A / DIVERGE), never forge a
  green. This is exactly why we can drop in a wholly different translator and trust
  the result: the signed-signal checks, not the translator, earn trust.

## 3. What forclift already gives us for free (do NOT re-port these)

diffusionMTUs currently reimplements decomposition internally
(`diffusionMTUs/project_index.py` — Tarjan SCC over a C symbol graph, leaf-first
order). forclift already has all of it, better-integrated:

| diffusionMTUs today | forclift equivalent to reuse |
|---|---|
| `project_index.build_index` (symbol graph) | `defn_graph` producer (`forclift/producers/defn_graph.py:165`), real cslicer/libclang extractor |
| `tarjan_scc` + SCC groups | `graph.cluster_modules` → `ModulePlan` (`forclift/core/graph.py:168-195`) |
| the per-file leaf-first loop in `run_project.phase_rustgen` | the `dag` combinator (`forclift/core/decompose.py:59`) — leaf-first, parallel, worktree-isolated |
| `run_project.py` orchestration + resumability | forclift `Scheduler` + artifact disk cache + `--resume` |

The "maintain the existing structure" mandate = **map diffusionMTUs' MTU-discovery
and rustgen stages onto these seams; delete the orchestration layer, keep the
brains** (chunker, lock-check, the Stage T/S/C rustgen prompts, the compile loop).

## 4. The LLM-seam decision (important)

diffusionMTUs talks to a **local vLLM server** via its own `llm.py` / `config.py`
(async, concurrency-bounded, `MODEL_SERVERS`). forclift has its own agent layer
(`forclift/core/agents.py`, key-scrubbed `child_env`) built for one-shot
`agent.run(prompt, cwd, env)` calls.

**Recommendation: keep diffusionMTUs' vLLM client; do NOT route through forclift's
agent seam.** The MTU pipeline is a multi-stage async graph, not a single prompt —
`agentic_producer` (`forclift/producers/agentic.py:43`, one prompt → files) is the
wrong shape. Instead the producer's `build()` invokes the diffusionMTUs pipeline
directly (as a library call or subprocess), writing the crate into the output dir.
This matches the trust model (the translator is untrusted regardless of which model
backs it) and keeps the concurrency/config work already done in diffusionMTUs intact.

Consequence: the forclift image/run env must be able to reach the vLLM endpoints
(or run them). That's an infra note for the Dockerfile, not a code dependency.

## 5. Concrete plan — two milestones

### Milestone 1 — monolithic producer (fastest path to a green run)
Get diffusionMTUs producing `translated_rust/` and passing forclift checks on a
single project, WITHOUT the module DAG yet.

1. **New producer module** `forclift/producers/mtu_translate.py`, exporting
   `register_mtu_translate_producer(env)`. Mirror the structure of
   `agentic.py`/`inputs.py`: a `Producer(kind="rust.src", version=..., build=...,
   fingerprint=..., validator=path_contract(validate_cargo_workspace))`.
   - `build(ref, deps, ctx)`: read C from `ctx.workspace.c_source`; run the
     diffusionMTUs pipeline (`run_project.py`'s phases: index → mtu → types →
     rustgen); emit the assembled crate into `ctx.workspace.rust_dir`.
   - `fingerprint(ref, ctx)`: content-address by the C source tree
     (`tree_fingerprint([ctx.workspace.c_source])`, see `producers/fingerprint.py`)
     — so a resume regenerates only when the C changes. This is what earns
     `--resume` honesty.
2. **Crate-shape reconciliation.** diffusionMTUs emits its own `Cargo.toml` +
   `src/lib.rs` (+ `src/main.rs`) via `rustgen/assemble.py:assemble_project`
   (crate-type `["cdylib","rlib"]`, bin when there's an entry fn). forclift's
   `validate_cargo_workspace` expects the `translated_rust/` workspace shape (rlib +
   cdylib + bin — see `forclift/producers/bindgen.py` and `docs/GENERALIZATION.md §5`).
   Reconcile `assemble_project` output to that shape, or add a thin adapter in the
   producer. **This is the most likely source of first-run friction — budget for it.**
3. **Swap-in wiring.** Add a `--translator {input,mtu}` (default `input`) to
   `forclift/cli.py:_context` (or a `standard_producers(translator=...)` param); when
   `mtu`, call `register_mtu_translate_producer(env)` AFTER `register_input_producers`
   so it overrides the `rust.src` kind.
4. **Vendoring / packaging.** Decide how diffusionMTUs code is importable from
   forclift: (a) `pip install -e` the diffusionMTUs repo, (b) vendor it under
   `forclift/vendor/diffusionmtus/` (there was a `sync_diffusionmtus.sh` rsync
   pattern on the other branch — reuse that idea), or (c) subprocess to
   `python run_project.py`. Prefer (a)/(b) for a library call; (c) is the quickest spike.
5. **Smoke test.** `python -m forclift run --project <proj> --translator mtu --set correctness`
   on one small project (e.g. an `array_list` / `binary_heap` from the diffusionMTUs
   `out/` corpus) → expect the crate to build and `value`/`bineq` to run (green or an
   honest DIVERGE, never a crash).

### Milestone 2 — decomposed via the module DAG
Replace the internal per-file loop with forclift's `dag`, so modules translate
leaf-first, in parallel, each in its own worktree against frozen sibling interfaces.

1. Derive a `ModulePlan` from the `defn_graph` artifact via `cluster_modules`
   (semantic prior = source file, matching diffusionMTUs' per-file granularity).
2. Express the translator as a `dag(body, derive=...)` where `body(module)` runs
   diffusionMTUs' Stage T/S/C for that module's symbols and records a merge signal
   (`forclift/core/decompose.py`). Sibling signatures come from the frozen
   skeleton/bindgen seam instead of diffusionMTUs' hand-rolled `registry`.
3. Retire `diffusionMTUs/project_index.py` and the `phase_rustgen` file loop; the
   `dag` owns order + parallelism, forclift owns isolation + merge.

## 6. Open decisions for the next session (resolve before coding M1)

1. **Packaging** — install-e vs vendor vs subprocess (§5.4). Pick one; it shapes imports.
2. **Where does `translated_rust/` get its Cargo layout** — teach `assemble_project`
   forclift's shape, or adapt in the producer? (§5.2)
3. **vLLM reachability from the forclift run env** — same host, or an endpoint the
   container can hit? (§4)
4. **Config surface** — expose diffusionMTUs' `Config` (model, concurrency,
   `rustgen_c_source_context`, the new `lock_round_trip_max_open`) through forclift's
   `--config`/`params`, or keep a diffusionMTUs config JSON path? Recommend the latter
   for M1 (least coupling).
5. **M1 vs M2 scope** — is a monolithic-crate producer acceptable as the first
   landed increment, or does the module DAG need to be in the first PR?

## 7. Key file map (read these first, in this order)

**forclift architecture**
- `docs/DECOMPOSE-DESIGN.md`, `docs/GENERALIZATION.md`, `docs/plans/09-translator-integration.md`,
  `docs/plans/10-decomposition-scale.md`, `docs/INTEGRATION.md` — design authority.
- `forclift/core/context.py` — `RunContext` / `Workspace` (paths the producer gets).
- `forclift/core/artifact.py` — `Producer` / `ArtifactRef` / `Resolved` / `ArtifactStore`.
- `forclift/producers/inputs.py` — the `rust.src` producer we're overriding (the template).
- `forclift/producers/agentic.py` — generation-producer pattern (fingerprint/caching discipline).
- `forclift/producers/__init__.py` — `standard_producers()` registration + swap comment.
- `forclift/core/decompose.py` + `forclift/core/graph.py` (`cluster_modules`, `ModulePlan`) — M2.
- `forclift/producers/defn_graph.py` — the C symbol graph to derive modules from.
- `forclift/cli.py` — how a run is assembled (`cmd_run`, `_context`).
- `forclift/checks/{value,bineq,abi,quality}.py` — the validators that gate the output.

**diffusionMTUs (the translator being integrated)**
- `run_project.py` — current orchestration (`phase_rustgen` is the code to relocate).
- `rustgen/{types_stage,spec_stage,code_stage,compile_loop,assemble,sibling_deps}.py` — the brains to keep.
- `chunker.py`, `lock_check.py`, `strategies/` — MTU discovery to keep.
- `project_index.py` — the part forclift REPLACES (`defn_graph` + `cluster_modules`).
- `config.py`, `llm.py` — the vLLM seam to keep.

## 8. Context already done this session (not part of the integration, just state)

- Six legacy files (`run_rust.py`, `deps_stage.py`, `ffi_stage.py`, `layout_stage.py`,
  `surface.py`, `tools/surface_report.py`) archived under `diffusionMTUs/legacy/`
  (see `legacy/README.md`). They were the OLD single-file rustgen + FFI-equivalence
  path; `sibling_deps.py` + `phase_rustgen` supersede them.
- Six now-dead config fields removed from `Config` (`rustgen_ffi_enabled`,
  `rustgen_ffi_max_tokens`, `rustgen_deps_mode`, `bindgen_bin`, `bindgen_clang_args`,
  `c_source_root`); `load_config` accepts-but-warns on them (`DEPRECATED_CONFIG_KEYS`).
- Added config knob `lock_round_trip_max_open` (default 3) — the round-trip probe's
  strictness, now A/B-able.
- `concurrency: 8` set in all `configs/*.json` presets.

Note: forclift's FFI/ABI story is its OWN (bindgen skeleton, `abi`/`value` lanes) —
do NOT try to revive diffusionMTUs' archived FFI cluster for it; use forclift's.
