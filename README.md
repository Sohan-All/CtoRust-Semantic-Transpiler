# CtoRust-Semantic-Transpiler

Translates C projects to Rust by **describing** the C first and generating Rust
from the description — not by transliterating it.

The premise is that most C-to-Rust failures come from porting control flow and
storage tricks that were never part of the program's behavior. So each file is
first decomposed into **MTUs** (Minimum Translation Units): chunks of code whose
behavior can be written in plain English, with a list of invariants, such that
the description alone is enough to write an idiomatic Rust implementation.
Codegen then works from the description. Control flow is re-derived, not copied.

`SPEC.md` specifies MTU discovery in full. This README covers the whole
pipeline: discovery, code generation, the correctness gates, and how a
translation gets scored.

---

## Pipeline

One pass per project:

```
chunker (tree-sitter)   ->  per-file MTU discovery      (run.py, parallel)
project_index (SCC)     ->  shared types                (stage T)
stage S (spec)          ->  stage C (code)              (one call per MTU, parallel)
assemble                ->  compile loop                (cargo check -> repair, 5 rounds)
                        ->  stage-T repair / stub repair
                        ->  semantic_check              (diagnostic only, no LLM)
```

**Stage T** writes the shared type definitions once, so units generated in
parallel agree on the data model. **Stage S** pins each unit's Rust signatures.
**Stage C** writes the bodies. Everything after that is repair.

Each unit's prompt carries context it could not derive on its own: the unit's
own C source, call-site expressions from across the project, the printf output
formats it emits, which C functions' return values reach the process exit
status, and which of its parameters are callbacks.

---

## Layout

| path | what |
|---|---|
| `run_project.py` | entry point — whole project, all phases |
| `run.py` | one C file through MTU discovery |
| `chunker.py` | tree-sitter seed blocks, call sites, callback parameters |
| `strategies/` | MTU discovery: `diffusion.py` (merge to fixpoint), `whole_file.py` |
| `project_index.py` | cross-file index, SCC grouping |
| `exit_status.py` | return-flow analysis — which C returns become the exit code |
| `rustgen/types_stage.py` | stage T |
| `rustgen/spec_stage.py` | stage S + the project-wide responsibility pass |
| `rustgen/code_stage.py` | stage C |
| `rustgen/compile_loop.py` | `cargo check` -> repair, and every write gate |
| `rustgen/types_repair.py` | repairs the shared types block when stage T's gates exhaust |
| `rustgen/stub_repair.py` | resolves leftover `todo!()` in the finished crate |
| `rustgen/semantic_check.py` | oracle-free omission checks, diagnostic only |
| `rustgen/common.py` | the checkers (`illegal_stubs`, `emptied_blocks`, …) |
| `config.py` | every threshold and knob |
| `configs/` | model presets |
| `test_*.py` | see [Tests](#tests) |

---

## Running it

Use the repo's venv.

```bash
venv/bin/python run_project.py <c_root>
```

`<c_root>` is the directory **containing** `src/` — for the corpus that is
`.../<project>/test_case`. Phases can be run separately with
`--phase {index,mtu,types,rustgen,all}`.

State (MTUs, specs, code, records) goes to `out/` by default.
**`DIFFUSIONMTUS_OUT` redirects all of it**, which is how parallel runs on the
same project stay isolated.

```bash
DIFFUSIONMTUS_OUT=/path/to/run1 venv/bin/python run_project.py <c_root> --config cfg.json
```

The run writes a `project.jsonl` record: config, per-stage outputs, repair
transitions, token usage, and every failure. That record is the audit trail —
the gates below read it, and so should you.

### Models

`config.py` `MODEL_SERVERS` maps a served model name to an endpoint. Two local
vLLM instances of the same weights are registered so two runs proceed at once
without sharing a batch scheduler (concurrent requests in one vLLM instance get
interleaved into the same batches, which couples the runs). An Anthropic
backend via Vertex is also wired.

Sampling is **temperature 1.0, top_p 0.95, top_k 64**, sent explicitly on every
request so the run's record describes what was actually sampled.

---

## Configuration

Everything tunable lives in `config.py` as a dataclass; a `--config foo.json`
overrides fields by name.

The `rustgen_*` booleans below are **ablation knobs, not tuning**. Each removes
exactly one prompt input or one repair pass so an A/B run can attribute its
share of a measured gain. All default `True` and nothing should ship with one
`False`.

| knob | removes |
|---|---|
| `rustgen_call_sites` | CALL SITES block |
| `rustgen_symbol_map` | symbol map in stage S |
| `rustgen_output_formats` | OUTPUT FORMATS block |
| `rustgen_exit_status` | EXIT STATUS block |
| `rustgen_callback_params` | CALLBACK PARAMETERS block |
| `rustgen_owner_routing` | routing missing-item errors to the owning section |
| `rustgen_scoped_prohibitions` | scoping `must_not_implement` to relevant concerns |
| `rustgen_types_repair` | stage-T repair (restores "record and continue") |
| `rustgen_stub_repair` | the stub repair pass |
| `rustgen_stub_context` | the stub loop's retrieval — its control arm |

`rustgen_c_source_context` controls how much of the unit's original C the
spec/code stages see: `off` (description and invariants only), `literals`
(string/numeric literals only), or `full` (raw C lines, labeled
reference-only). Default `full` — withholding the C's literal output text loses
data without buying anything, since the point is not transliterating *control
flow*.

---

## Correctness gates

A pipeline that survives failures is only an improvement if a damaged crate
cannot be mistaken for a whole one. So every fallback records what it lost, and
a run that lost anything scores nothing.

The generation-side checks all live at the **write point**
(`compile_loop.set_section`), not at each caller, so they cover every writer —
generation, each repair tier, and the sibling-stub resolver alike.

| check | refuses |
|---|---|
| `unbalanced_delimiters` | a section that will not parse |
| `parse_regression` | a repair that makes a *balanced* section unparseable |
| `emptied_blocks` | a repair that strips every item out of an `impl`/`trait` |
| `lost_impl_methods` | a repair that drops a method other sections still call |
| `illegal_stubs` | a bare `todo!()` outside a `*_deps` module |
| `illegal_type_bodies` | a real method body in the shared types block |
| `stubbed_callbacks` | a callback written as a global stub instead of an `Fn` bound |
| `remaining_stubs` | *scoring* a crate that still contains a runtime panic |

Two checkers deliberately answer different questions: `illegal_stubs` asks "is
this section illegitimately stubbed, mid-pipeline?" and permits a documented
`todo!("reason")`; `remaining_stubs` asks "does the crate about to be scored
contain a panic?" and permits nothing.

`todo!()` type-checks — it coerces to `!` — so a fully stubbed crate compiles
clean and would otherwise score its panics as ordinary divergence. That is the
failure these gates exist for.

Each candidate damage rule was scored against the recorded repair history
before being adopted; the obvious formulations each fail differently, and the
reasoning is kept inline in `common.py`.

---

## Repair

Three loops, in order:

1. **Compile loop** — `cargo check`, then repair, up to 5 rounds. Tiers from
   surgical edits through full-section rewrites; a rejected repair falls
   through to the next tier rather than costing a round.
2. **Stage-T repair** (`rustgen/types_repair.py`) — when the shared-types gates
   exhaust their retries. That branch used to be fatal in every recorded run,
   because a stubbed shared block hands every unit a phantom API to defer to.
   The repair's output must survive brace balance, `lost_type_definitions`, the
   original gates, and a real `cargo check` of the block. The right fix is
   usually deletion, not completion.
3. **Stub repair** (`rustgen/stub_repair.py`) — resolves leftover `todo!()` in
   the finished crate. Runs *after* the compile loop, because repairs are a net
   stub producer. Three verification tiers: compiler-checkable, evidence-
   checkable (a patch with no cited evidence is refused — a type-correct wrong
   callback compiles clean), and uncheckable, which declines.

Two rules learned the hard way and worth keeping if you extend these:

- **Never put generated Rust inside a JSON string.** Under a constrained
  grammar the model does not reliably escape quotes and newlines, and returns
  plausible-looking, completely corrupt code. Every prompt here returns a
  fenced ` ```rust ` block parsed by `extract_rust`. JSON is for short metadata
  only.
- **Splice, don't replace.** `splice_function` swaps one function's exact span
  and copies everything else byte-for-byte, which makes "the repair deleted a
  sibling" impossible rather than merely detectable.

---

## Evaluation

Translations are scored by **forclift**, a differential-testing harness in the
sibling `uwisc-docker/` repo. The C program is the oracle, so a translation can
under-perform but never forge a pass.

```bash
cd ../uwisc-docker
python3 -m forclift check bineq --project <proj> --results <dir> --translator input
```

`--translator input` scores a pre-built crate from `<proj>/translated_rust/`;
`--translator mtu` invokes this pipeline directly. It prints one verdict line,
e.g. `bineq DIVERGE 4/16 comparable cases diverge`.

Before scoring, a run must clear the gates above: `BUILD_FAILED`,
`STUB_CRATE`, `DEGRADED` and `CRASHED` are all verdicts that mean *no score*,
not a bad score.

**The test vectors are the metric, never an input.** No repair or prompt path
may read them.

### Reading a number

The pipeline is noisy and single runs mean nothing.

- One configuration, unchanged, produced divergence counts of 1, 4, 6 and 9 on
  the same project.
- *Which* cases fail is close to pure noise — two runs of one arm failed
  disjoint sets. *How many* is roughly stable. Trust the count, not the list.
- With 2 runs per arm the best two-sided p available is 0.333. n=4 is the floor
  for significance to be possible; ~7 for 80% power.
- Prefer **paired** comparisons, and pair by corpus project rather than by
  re-running one project.

Sampling is unseeded, so repeats are **trials**, not seeds — a trial cannot be
reproduced exactly. ("Seed" already means the chunker's tree-sitter seed
blocks.)

The experiment harness (`run_ablation.sh`, batch drivers, recorded runs and
verdicts) lives outside this repo under `mtu_runs/` and is not versioned.

---

## Tests

Run each with `PYTHONPATH=. venv/bin/python <file>`; all should print
`ALL PASS`.

| file | covers |
|---|---|
| `test_exit_status.py` | return-flow analysis, incl. the negative cases |
| `test_prompts.py` | every prompt template formats, and the retry paths execute |
| `test_degradation.py` | failure paths, per-unit persistence, the crash boundary |
| `test_responsibility.py` | the project-scoped responsibility pass |
| `test_types_repair.py` | stage-T repair, success *and* exhaustion |
| `test_stub_repair.py` | stub repair, splicing, give-up, an LLM that raises |
| `test_callback_params.py` | callback detection, incl. typedef'd and unnamed forms |

The tests drive the paths with a fake LLM — including one that raises — rather
than asserting on checkers in isolation. Testing a checker is not testing the
*response* to the checker: thorough `illegal_stubs` tests still let a
`KeyError` ship in the retry note they triggered, because nothing executed that
path.

---

## Known limitations

Open, with mechanisms understood to varying degrees:

- **Dual-container invariants.** A C type keeping two views of one collection
  (a heap array and a linear list) collapses into one Rust `Vec` at stage T,
  but MTU discovery has already recorded the two-container arrangement as a
  behavioral invariant — so the unit honors it and inserts twice. This is a
  storage arrangement promoted to behavior, the same failure as C's NUL
  terminators leaking into Rust strings. Largest single divergence source.
- **C function-pointer parameters.** Callbacks want an `Fn`-bound generic
  threaded from the call site. Half the corpus declares them. The misclassi-
  fication that turned them into global stubs is fixed; the caller side, struct-
  field callbacks (allocator vtables), and closure signatures wanting two
  aliasing borrows are not.
- **Signatures that cannot be composed.** Stage S pins each unit's signatures
  in parallel with no check that the *set* is satisfiable.
  `require_project(&self) -> &Project` plus `add_task(&mut Project)` is
  unsatisfiable under the borrow checker; neither unit is wrong, and the unit
  correctly stubs rather than fakes it.
- **Cross-unit error-variant disagreement.** Two units independently map the
  same C status onto different error variants, so a caller's match falls
  through. `exit_status.py` covers the flows that reach `main`; internal
  statuses are not covered.
- **Duplicate definitions across units.** The responsibility pass is now
  project-scoped, but compliance is stochastic at temperature 1.0. A static
  pre-check (two sections defining the same `impl T { fn n }`) would catch what
  remains without an LLM call.
- **`final: 0` is not the same test as "the crate builds."** The compile loop
  uses `cargo check`; forclift uses `cargo build --release`. Codegen-time
  failures pass the first and fail the second.

---

## Conventions

Carried over because each one cost something to learn:

- Assert a detector against known-**good** and known-**bad** input. A gate that
  rejects everything looks exactly like a gate that works.
- A gate must cover every **writer**, not the first one. Ask which other stages
  can write that text, and put the check at the write point.
- Test a relaxation against every fixture at once, not the one that motivated
  it.
- A stalled loop accuses the last thing that said no. Read what a check
  actually rejected before believing it is the blocker.
- Before adding a detector, find the writer. A guard that catches damage is
  worth less than the prompt line that stops it being produced.
- An exemption becomes a hole once the model learns its shape. A placeholder in
  a prompt is a template the model will copy literally — write examples as
  filled-in instances, never as `<slot>` syntax.
- Prefer the measurement that reproduces over the one that reaches p<0.05.
- A mechanism story that passes every check short of a measurement is still not
  a measurement.
