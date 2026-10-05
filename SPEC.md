# MTU Discovery Pipeline — Specification

This document specifies the pipeline that decomposes a C codebase into **Minimum
Translation Units (MTUs)**: chunks of code whose behavior can be described in English
with no reference to C-specific constructs, such that the description alone is
sufficient to write an idiomatic implementation in Rust (or any language).

This spec covers MTU **discovery only**. Converting MTUs into Rust specs and Rust code
is a later phase and deliberately out of scope here.

## Motivation

Translation is reframed as an engineering problem: instead of asking one large model to
translate C to Rust directly, we break the work into many small, easy subtasks
(describe a chunk, combine two descriptions, judge a description) that cheap models
(Claude Haiku) can do reliably. This is cheaper, parallelizable per file, and — because
each strategy/policy is a pluggable module — A/B testable end to end.

Eventually this pipeline slots into the main translator (`uwisc-docker` repo) under
`/translators/MTUs/`, switchable against the old translation method with a single
config flag. It is being developed standalone in this directory first.

## Core concepts

| Term | Definition |
|---|---|
| **MTU** | A set of source line ranges plus a 1–5 sentence English description that passes the lock check. The unit of translation for the Rust phase. |
| **Explanation** | An intermediate (range(s), description, invariants) record. Starts fine-grained; grows via merging. |
| **Locked** | An explanation that passed the lock-check stack. It is an MTU; no further merging. |
| **Open** | An explanation still eligible for merging and lock attempts. |
| **Irreducible** | An explanation that will not lock (too C-entangled) and must not be merged further. Flagged for escalation to a stronger model, which attempts the *same task* — a language-agnostic description — with more capability. Never a C-aware direct translation. |
| **Invariants** | Semantic details (overflow behavior, aliasing, index bounds, ownership, error paths) attached to an explanation. Must survive merging verbatim even as the prose generalizes. |

## Guarantees (both strategies)

1. **Coverage**: every source line of the input file belongs to exactly one final unit
   (locked or irreducible). No gaps, no overlaps. Enforced mechanically by a shared
   validator, never assumed from model output.
2. **Provenance**: every unit carries its line range(s) and, for merged units, the full
   merge ancestry, so the source can always be re-consulted at spec time.
3. **Termination**: bounded passes and per-unit staleness rules; the pipeline cannot
   loop forever (see Termination below).

## Architecture

```
diffusionMTUs/
├── SPEC.md                    # this document
├── config.py                  # strategy selection, model IDs, thresholds
├── chunker.py                 # tree-sitter syntax seeding + intra-file call graph
├── llm.py                     # thin Anthropic client wrappers (fresh-call helpers)
├── lock_check.py              # shared lock-check stack (regex → C-mention → round-trip)
├── validator.py               # coverage validator (no gaps/overlaps/full file)
├── state.py                   # JSONL state store + explanations.md renderer
├── strategies/
│   ├── diffusion.py           # Strategy A: fine-grained explain + iterative merge
│   └── whole_file.py          # Strategy B: single-call direct MTU extraction
├── run.py                     # CLI entry point: run.py <file.c> --strategy {diffusion,whole_file}
└── out/<file-stem>/
    ├── state.jsonl            # source of truth
    └── explanations.md        # human-readable render
```

Strategies are selected by config/CLI flag. `chunker`, `lock_check`, `validator`, and
`state` are shared between strategies so A/B comparisons isolate the strategy itself.

## Stage 0 — Syntax seeding (shared)

Deterministic, no LLM. Uses **tree-sitter** (`pip install tree-sitter tree-sitter-c`);
it parses non-compiling and preprocessor-heavy C without a build.

1. Parse the file; walk root children to get top-level units with exact line ranges:
   `function_definition`, `struct_specifier`, `typedef`, `preproc_def`/`preproc_include`,
   globals, stray comments.
2. Functions longer than ~40 lines are split one level down, at the top-level
   statements of the body (large `if`/`for`/`switch`/`while` blocks).
3. Build an **intra-file call graph**: collect `call_expression` identifiers in each
   function body, keep edges whose callee is a function defined in this file. Calls to
   external/library functions are recorded as *external dependencies* on the caller's
   seed block (they end up as invariant notes, e.g. "calls `malloc`").
4. Condense strongly connected components. Mutually recursive functions are collapsed
   into one seed unit (neither is describable without the other). The result is a DAG.
5. Emit: the ordered list of seed blocks (line ranges) + the call DAG + a topological
   order over it.

Deterministic chunking means reruns and cross-model comparisons chunk identically —
differences we measure are attributable to the models and policies, not chunking luck.

## Strategy A — Diffusion merge

Inspired by a reverse-diffusion process: start with maximally fine-grained, detailed
explanations; progressively "denoise" toward general, language-agnostic descriptions.

### Pass 0 — Describe

For each seed block (in **topological order** over the call DAG, leaves first), one
fresh Haiku call: "Describe what this code does in 1–5 sentences, plus a bullet list of
semantic invariants a reimplementer must preserve." When describing a caller, the
already-produced descriptions of its callees are substituted into the prompt in place
of re-deriving them ("sorts the entries" instead of restating the callee's body). This
is what keeps descriptions short as units grow.

Leaf-level calls within a pass are independent → issued in parallel (async client).

### Merge passes

Each pass:

1. **Lock attempts**: run the lock-check stack (below) on every open explanation that
   changed since its last attempt. Passing → status `locked`.
2. **Candidate generation** (pluggable policy): candidate pairs are the union of
   - *file-adjacency edges*: explanations covering adjacent line ranges, and
   - *call-DAG edges*: caller–callee pairs (a function and its helper belong together
     regardless of file position).
   Both members must be open.
3. **Merge**: for each selected candidate pair, one fresh Haiku call: "Combine these
   two descriptions into one 1–5 sentence description of the combined behavior. Carry
   over every invariant from both, verbatim." Output replaces the pair with a new open
   explanation whose range is the union of ranges (possibly non-contiguous, when merged
   along a call edge) and whose `parent_ids` record ancestry.
4. Re-check termination conditions.

Greedy pair selection within a pass: prefer call-DAG edges over adjacency edges;
process in topological order; a unit participates in at most one merge per pass.

### Termination

- **Fixpoint (primary)**: a full pass produces zero merges and zero new locks → stop.
- **Per-unit staleness**: an open explanation that fails the lock check twice without
  being changed by a merge in between → mark `irreducible` (re-checking an unchanged
  explanation just repeats the same "no").
- **Size guard**: an open explanation whose source range exceeds **300 lines** and
  still won't lock → mark `irreducible` rather than merging further. Past this point
  merging generalizes away exactly the details a spec needs.
- **Safety-net pass cap**: hard cap of **6 merge passes**. Hitting it regularly means
  the fixpoint logic has a bug; remaining open explanations are marked `irreducible`.

Terminal invariant: every unit is `locked` or `irreducible`, and the validator confirms
coverage.

## Strategy B — Whole-file direct

Tests whether a weak model can do the whole decomposition in one shot, trading the
diffusion machinery for ~10× fewer calls.

1. One fresh Haiku call with the entire file: "Partition this file into MTUs. For each,
   output the line range, a 1–5 sentence language-agnostic description, and invariants.
   Every line must be covered by exactly one MTU."
2. **Mechanical validation** (`validator.py`): parse the returned ranges; check no
   gaps, no overlaps, full coverage. On violation, retry with the specific violations
   quoted back ("lines 141–158 are uncovered; ranges 30–45 and 40–60 overlap").
   Max 3 retries, then the file is marked failed for this strategy (an A/B data point,
   not a crash).
3. Each proposed MTU then goes through the **same lock-check stack** as Strategy A.
   Failures are marked `irreducible` (no merging machinery in this strategy).

The characteristic failure mode this guards against is silent omission — the model
describing the interesting 80% and skipping error-handling and cleanup paths. The
validator converts that from a silent failure into a detectable, retryable one.

Files exceeding a size threshold (start: ~500 lines / conservative token estimate) are
out of scope for Strategy B in v1 and routed to Strategy A.

## Lock-check stack (shared)

Run stages cheapest-first; a unit locks only if **all** stages pass. Each stage only
runs if the previous passed.

1. **Regex blocklist** (free, deterministic): the description text must not match a
   curated list of C-isms — `pointer`, `malloc`/`free`, `#define`, `null-terminated`,
   `char *`, `struct` (as a C keyword reference), `errno`, `void *`, etc. The list
   lives in `lock_check.py` and grows as we see escapes.
2. **C-mention check** (fresh Haiku call): "Does this description rely on any
   C-specific constructs, idioms, or memory-model assumptions? List them or answer
   NONE." Any listing → fail.
3. **Round-trip probe** (fresh Haiku call): "You are to implement the following
   behavior in Rust. List the questions you would need answered before you could start.
   If none, answer NONE." If the questions are C-shaped ("who frees this?", "is the
   buffer null-terminated?") → fail. Classification of question C-shapedness is itself
   a cheap judgment; v1 uses the same blocklist regex over the returned questions plus
   a length heuristic, refined later.

**The lock check covers the whole unit — description AND invariants.** Both feed the
Rust spec later, so a C-ism hiding in an invariant fails the lock the same as one in
the prose: the blocklist runs over both, the C-mention prompt receives both, and the
round-trip probe sees the complete unit.

**Invariant rephrasing.** Invariants are copied verbatim through merges, so a C-phrased
invariant can never be fixed by further merging — left alone it would fail the lock
forever and incorrectly drive the unit to irreducible via staleness. When a lock
failure localizes to a specific invariant (blocklist and C-mention failures identify
the offending line), issue one fresh rephrase call: "Restate this fact as observable
behavior, without reference to C mechanisms" (e.g. "calls realloc, old pointer invalid"
→ "on growth, prior references to the contents are not preserved"). A rephrase counts
as a change for staleness purposes (the unit earns a fresh lock attempt). If rephrasing
cannot remove the C-dependence, that is a genuine irreducibility signal. The pass-0
describe prompt also instructs "state invariants as observable behavior, not
implementation mechanism" to prevent most of these up front. Max 2 rephrase attempts
per invariant.

The stack is a pluggable component — stages can be swapped/reordered per config for
A/B experiments (e.g., strong-model lock verdicts: Haiku does the O(n) describe/merge
work, a stronger model does only the O(#MTUs) lock decisions).

## Irreducible escalation

Irreducible units are **not** translated C-aware. They are flagged with status
`irreducible` and queued for a stronger model (e.g. Sonnet/Opus) to attempt the same
language-agnostic description task. Same task, bigger brain — the philosophy of the
pipeline is preserved, and the irreducible set doubles as a map of where the expensive
tokens go. Escalation is a stub in v1 (units are flagged and reported, not yet
escalated automatically).

## State format

`state.jsonl` is the source of truth — one JSON object per line, append-friendly.
`explanations.md` is rendered from it (line-range header + description + invariants,
separated by `====` lines) and is never hand-edited or parsed back.

Explanation record:

```json
{
  "id": "exp_0007",
  "ranges": [[120, 143], [201, 215]],
  "text": "Maintains a growable string buffer supporting append and clear...",
  "invariants": ["append doubles capacity when full", "clear retains allocation"],
  "status": "open | locked | irreducible",
  "pass": 2,
  "parent_ids": ["exp_0003", "exp_0004"],
  "external_deps": ["memcpy"],
  "model": "claude-haiku-4-5",
  "strategy": "diffusion",
  "lock_attempts": 1,
  "lock_failures": [{"stage": "round_trip", "detail": "asked about null-termination"}]
}
```

`parent_ids` gives the full merge tree for free — essential when debugging why an MTU
came out wrong. Seed-graph records (call edges, topo order) and run metadata (config
snapshot, timestamps) are also written as typed records in the same file.

## Models and API usage

- **Worker model**: `claude-haiku-4-5` ($1/$5 per MTok, 200K context) via the
  `anthropic` Python SDK, `ANTHROPIC_API_KEY` from env. Local open-weight models are
  explicitly out of scope for now; `llm.py` is the single seam where a different
  backend would plug in later.
- **Fresh calls everywhere**: every describe/merge/lock operation is an independent
  `messages.create` call with a short, self-contained prompt. There is no per-call
  instantiation cost — a "new instance" is just a new request — and short independent
  prompts are cheaper than threaded conversations.
- **Parallelism**: `AsyncAnthropic` + `asyncio.gather` over independent calls (all
  leaf describes in a pass; all merge calls in a pass; files are fully independent).
  Modest concurrency cap (start: 8) to stay under rate limits; the SDK retries
  429/5xx automatically.
- **`max_tokens`**: small — descriptions are 1–5 sentences; 1024 is ample.
- **Cost lever for later**: the Batches API runs the same requests at 50% price with
  up-to-hours latency. Not used in v1 (interactive iteration matters more than cost
  while developing), but the fresh-call design maps onto it directly.

## Evaluation (pre-Rust)

Until the Rust-generation phase exists, strategies are compared on:

1. **Coverage & validity**: validator pass rate; Strategy B retry/failure counts.
2. **Lock precision**: sample of locked MTUs judged (by a strong model, and by eye)
   on "could a competent Rust programmer implement this without seeing the C?"
3. **Irreducible rate & location**: how much of the file failed to lock, and whether
   it concentrates where expected (pointer-heavy code).
4. **Granularity profile**: MTU count and size distribution vs. the functions-as-MTUs
   baseline (Strategy B restricted to function boundaries — a nearly-free baseline
   both strategies must beat to justify their machinery).
5. **Cost**: total tokens / calls per file per strategy (from `usage` on responses,
   recorded into `state.jsonl`).

The eventual endpoint metric is quality of the resulting Rust through the existing
test infrastructure; intermediate-step nondeterminism is handled by running more
samples per configuration, with attribution helped by the deterministic chunking.

## Rust generation (v1 — validation-free)

`rustgen/` consumes a completed run's `out/<stem>/state.jsonl` and emits
`out/<stem>/rust_crate/` (Cargo.toml + src/lib.rs). No validation gates by
design: the deliverable is code to eyeball; `cargo check` may be run by hand
but is not part of the pipeline. All stages use fresh parallel Haiku calls.

```
state.jsonl ─► Stage T (types) ─► Stage S (specs) ─► Stage C (code) ─► crate
                 1 call/file        1 call/MTU         1 call/MTU
```

- **Stage T — type synthesis** (`rustgen/types_stage.py`): all units'
  descriptions+invariants → shared Rust data model (structs, one error enum),
  `pub mod deps` with `todo!()` stubs for non-libc externals, and a glossary
  mapping description concepts to type names.
- **Stage S — specs** (`rustgen/spec_stage.py`): per MTU, signatures only.
  The collected signatures become sibling context for codegen (same trick as
  the lock-check sibling context).
- **Stage C — code** (`rustgen/code_stage.py`): per MTU, implement against
  spec + shared types + sibling signatures. Invariants are the acceptance
  criteria — each must be honored and cited with an `// invariant:` comment
  where non-obvious. Unimplementable parts become targeted `todo!()`.
- **Assembly** (`rustgen/assemble.py`): deterministic; MTU blocks in source
  order; irreducible-sourced code marked `// LOW CONFIDENCE`.
- CLI: `python3 -m rustgen.run_rust out/<stem>`. Provenance records
  (`rust_types`/`rust_spec`/`rust_code`) append to state.jsonl.

- **Compile loop with per-error routed repairs** (`rustgen/compile_loop.py`):
  after assembly, `cargo check --message-format=json` diagnostics are
  attributed to sections (MTU markers; the shared type layer and FFI module
  are sections too) and ROUTED by what each error implicates: all spans
  (primary + secondary + child notes) in one section → cheap single-section
  repair; spans touching 2+ sections, or one-span name errors whose missing
  identifier another section defines → union-find CLUSTERS repaired in one
  joint call carrying the authority rule (each section's stage-S spec is its
  contract; fix the deviator; ties → change the caller; never rename
  #[no_mangle] symbols, shared types, or deps stubs). A zero-progress round
  escalates once to a bundled call over all dirty sections (cap 4); a second
  stall ends the loop. A **regression guard** snapshots the best state and
  reverts + stops if a round increases errors (observed 2 -> 61 from an
  unguarded shared-types repair). `--compile-rounds 0` disables; `--resume`
  reruns repairs on recorded stage outputs without regenerating.
  Result on the 5-file corpus: **all five crates compile with zero errors and
  export their full public C symbol surface** (70 symbols) — the routed
  ladder resolved the cross-section E0308 class that pure per-section repair
  provably could not (vector 22→0, commit 36→0).

  Repairs are also tiered by GRANULARITY (whole-section rewrites proved to be
  overkill: a one-char brace fix via section rewrite went 1→3 errors):
  tier 0 applies rustc's own machine-applicable `suggested_replacement` spans
  deterministically (byte-offset edits on lib.rs, split back into section
  state via `split_lib`, re-checked — zero LLM calls); tier 1 asks for exact
  find/replace edits for sections with <=`rustgen_surgical_max_errors` errors
  (all-or-nothing application; any non-matching find falls through); tier 2 is
  the full-section rewrite, now the fallback (cluster repairs stay full —
  they renegotiate interfaces). Measured on a fresh signature.c generation:
  10 errors -> 0 with 10 rustc-auto fixes, 2 surgical edits, and ZERO
  rewrites.

- **FFI shim stage** (`rustgen/ffi_stage.py`) — the equivalence contract.
  Equivalence testing (uwisc-docker's cando2 value lane) binds ONLY at the
  file's public C functions; `static` internals are decomposition, not
  contract — requiring internal equivalence would force mirroring C's
  decomposition, against the redesign philosophy (cando2's own spec catalog
  is bindgen-over-public-headers, so it already agrees). The chunker detects
  `static` and captures each function's C signature; one Haiku call emits a
  `pub mod ffi` of `#[no_mangle] pub unsafe extern "C" fn` shims with exact
  C symbol names, adapting raw ABI arguments onto the idiomatic core (unsafe
  and raw pointers are permitted in this module only). The crate builds as
  `cdylib` + rlib; the compile loop treats the shim module as its own
  repairable section. Verified: `nm -D` on the built .so shows exactly the
  public C symbols. Actually RUNNING cando2 requires the uwisc-docker docker
  environment (cando2 binary + a C cdylib to record against) and dependency
  linkage for the deps stubs — deferred to integration.

- **Deps linkage stage** (`rustgen/deps_stage.py`, `--deps-mode extern
  --c-root <C tree>`): replaces the `todo!()` deps stubs so shims can actually
  run. Each stub keeps its exact idiomatic signature; per dep, its real C
  declaration/macro is located deterministically in the C source tree
  (headers incl. `include/git2/` preferred; case-insensitive macro fallback)
  and Haiku CLASSIFIES the resolution: **NATIVE** when the stub operates on
  this crate's own idiomatic state (calling C on a Rust struct would corrupt
  memory — data-structure/utility deps become Vec/str ops) vs **LINK** when
  the dep passes opaque handles to C-side state (extern "C" decl in `mod c` +
  marshaling adapter; symbols stay undefined in the cdylib and resolve at
  load time next to the C library — the incremental-port pattern). C macros
  are implemented natively. Unresolved deps keep an honest `todo!()`.
  Measured: tree-cache links 9 extern + 4 native (1 unresolved), builds clean,
  `nm` shows exactly the 9 git_tree_* symbols as U; pqueue classifies
  all-native (fully self-contained crate, no C needed).
  Known limitation: C function-pointer callbacks can't marshal into Rust fn
  types without a trampoline (pqueue's comparator shim) — such symbols are
  also outside cando2's amenability scope.
  Reply-shape hardening (learned the hard way across vector/signature/commit;
  the rule: NEVER trust reply layout — re-parse and re-render mechanically):
  extern decls are re-parsed through `surface.fn_decls` and re-rendered
  sanitized via `surface.extern_decl_rust` (struct pointers → c_void, junk
  struct definitions/embedded braces dropped — models return whole
  `extern "C" {...}` blocks with arbitrary extras); item bodies are scrubbed
  of label lines (`NATIVE:`) and stray code fences; undeclared `c::` helper
  references are BACKFILLED deterministically from bindgen decls before
  falling back to demoting the adapter to an honest `todo!()` stub; the
  unresolved-stub reconstruction regex must not stop at `;` inside `[T; N]`.
  Operational lesson: re-linking deps AND regenerating the whole FFI module
  on an already-converged crate is a regression risk (commit: 1 error → 121);
  the effective recovery was deterministic state restore (slice state.jsonl
  before the bad round; append-only latest-wins makes this cheap) plus
  mechanical sanitation, then a short compile loop (69→0).
  Final corpus state: all 5 crates build at 0 errors, 0 ABI drift, 72 C
  symbols exported; undefined git_* awaiting load-time resolution: tree-cache
  9, vector 1, signature 3, commit 11, pqueue 0 (self-contained).

- **todo!() cleanup pass (July 10)** — the deps stubs that survived linkage
  fell into: (1) demoted on malformed adapter replies, (2) declarations
  unfindable (libc like `mktime`, variadics like `git_error_set`), (3) honest
  design gaps. Fixes for 1+2, all in `rustgen/deps_stage.py`:
  `/usr/include` fallback for declaration lookup with a call-site filter
  (a `...mktime(&tm)) / 60;` line is not a declaration — the prefix before
  the name must contain only type/qualifier tokens); variadic `...`
  passthrough in extern decls; deterministic bindgen backfill for adapters
  referencing undeclared `c::` helpers; and `relink_todo_stubs()` — a
  TARGETED pass that re-adapts only the todo stubs and splices surgically,
  because full-section regeneration on a converged crate is the regression
  risk documented above. Also fixed in `run_rust.py`: compile-loop repairs
  to types_rs/ffi_rs are now recorded to state.jsonl (previously each
  --resume silently reverted them, resurrecting ~40 errors on commit).
  pqueue's callback gap closed with a global-slot trampoline in the FFI
  shim (`ComparisonFn` is a plain fn pointer; the C comparator is stored in
  a static AtomicUsize and read by a trampoline fn — one live comparator at
  a time, matching libgit2 usage).
  Post-cleanup corpus: all 5 crates at 0 errors / 0 ABI drift; deps todos
  eliminated everywhere except commit's 4 honest gaps (git_commit_lookup —
  adapter replies failed twice; commitarray marshalling; parse_ext; the
  repo-backpointer design gap), left as labelled todo!("...") for the
  equivalence phase to prioritize.

- **ABI surface alignment** (`rustgen/surface.py`, wired via `--c-root` +
  `--bindgen-args`) — bindgen as the authority for exported shims. The
  equivalence harness calls our cdylib with types derived by bindgen from the
  C HEADERS (forclift's value.spec/skeleton machinery); shims generated from
  tree-sitter signatures off the `.c` file can drift (measured on vector:
  three shims silently DROPPED a callback parameter, one returned int where
  the ABI says void — arity drift = stack garbage at replay, not a clean
  failure). `abi_decls()` runs the same bindgen invocation (own header +
  public umbrella, discovered `-I` dirs, extra clang defines from config) and
  the shim prompt carries the decl as "ABI AUTHORITY"; because the decl is
  mechanically verifiable, `generate_shims` VERIFIES each shim against it
  (arity, scalar width classes, pointer-ness — pointee types are
  ABI-irrelevant, so idiomatic `*mut c_void` opaques pass) and retries once
  with the exact mismatch quoted. Honest fallback: bindgen missing/headers
  broken → `{}` → tree-sitter signature stands as before.
  `python3 -m rustgen.surface out/<stem> --c-root <tree>` prints a per-symbol
  drift report for an assembled crate.

- **cando2 preflight** (`tools/surface_report.py`): intersects forclift's
  amenable-catalog machinery (bindgen → `forclift.engine.value_spec
  .generate_specs`) with our exported surface. Measured on the 5-file corpus:
  **0/72 symbols amenable** to the single-call value lane — every function
  traffics in pointers to non-value structs / opaque handles / callee-alloc
  double pointers, which the conservative skeleton correctly excludes. The
  equivalence evidence for handle-based APIs therefore comes from the
  EXECUTABLE lanes, chiefly `lib_swap` (the trusted C driver loads our Rust
  cdylib in place of the C library and runs the shared input-case pool;
  missing/ABI-wrong symbols read NOT_LINKABLE = DIVERGE) — which is exactly
  the incremental-port shape the deps linkage stage builds, and why ABI
  surface alignment is the load-bearing contract.

Known limitations (observed on generated output): stage T sometimes models
generic element storage as `Vec<u8>` (C's void* leaking through the
descriptions — a generic `<T>` would be idiomatic) and uses raw-pointer-typed
opaque handles in deps stubs; swap-like operations can misuse insert-semantics
deps. Compiling ≠ correct: these are what the equivalence-testing phase is for.

## Out of scope (v1)

- MTU → Rust spec and Rust code generation.
- Cross-file analysis (external calls are recorded as opaque dependencies).
- **First lib_swap differential (July 10, `libswap_project/`, since removed
  from this repo — it was a vendored libgit2 tree)** — the local
  prototype of forclift's lib_swap lane, built without docker. Layout:
  `test_case` (symlink to the libgit2 clar tree), `c_build/` (cmake+ninja →
  baseline `driver` CLI + `libdriver.so`), `lanes/<crate>/` (per-crate hybrid:
  libdriver.so relinked with ONLY that crate's object removed and its Rust
  cdylib linked in — cross-boundary symbols resolve through the dynamic
  loader in both directions, the incremental-port pattern as an artifact),
  `run_lanes.py` (scripted CLI scenarios under fixed identities/dates,
  baseline vs swapped, per-step rc/stdout/stderr diff; loader diagnostics
  classify NOT_LINKABLE, anything else DIVERGE).
  Linkability gaps found and fixed en route (all generalizable classes):
  (1) MACRO-GENERATED public functions — GIT_COMMIT_GETTER expanded 9 getters
  invisible to the tree-sitter function scan; the FFI stage never enumerated
  them (fixed by generating shims from the macro block with bindgen ABI
  authority; TOOLING GAP: ffi_stage should diff bindgen's header catalog
  against tree-sitter's pubs). (2) GIT_INLINE header functions — git__free
  has NO dynamic symbol anywhere (static inline in alloc.h); extern-linking
  it can never resolve. Nativized to libc free. TOOLING GAP: deps lookup must
  detect inline definitions and force NATIVE. (3) vector's remove_matching
  needed the global-slot trampoline (same class as pqueue's comparator).
  RESULT: all five lanes LINKABLE, all five DIVERGE, and the diagnosis is
  uniform — THE BOUNDARY DATA CONTRACT. Function-signature ABI (what we
  aligned) is necessary but not sufficient: C code accesses the structs our
  Rust fills. vector: git_vector is EMBEDDED BY VALUE in C structs → Rust
  Vector's different size/layout corrupts the enclosing struct → SIGSEGV in
  `init`. pqueue: git_pqueue IS git_vector (typedef) → revwalk's embedded
  pqueue corrupts OIDs (log prints an "id" that is raw pointer bytes).
  signature: C reads sig->name/email directly from Rust-allocated Signature
  (String ≠ char*) → SIGSEGV at commit. commit: object machinery allocates
  sizeof(git_commit) and our __parse writes a Rust Commit into it → clean
  "Error creating commit". tree-cache: "Could not write tree" via index's
  cache. Meanwhile init/add matched on pqueue/commit/signature/tree-cache
  lanes and the hash-and-cat scenario matched broadly — paths that don't
  cross a struct boundary behave identically.
  NEXT DESIGN DECISION (needs Sohan): layout-faithful boundary types —
  #[repr(C)] struct definitions derived from bindgen (the data analog of the
  ABI authority) for every type that crosses the C/Rust boundary by value or
  with direct field access, with idiomatic behavior kept behind them. This
  bounds "redesign, not translate": interior code stays free; boundary
  DATA must be bit-faithful, exactly like boundary functions.

- **Layout-faithful boundary types (July 12) — first lib_swap MATCH.**
  Design decision (Sohan): the idiomatic Rust core stays untouched; C layout
  is confined to a conversion layer at the FFI boundary (preserving Rust
  quality is the point of the whole pipeline). Implementation:
  `rustgen/layout_stage.py` (deterministic: parses bindgen output, walks the
  type closure from the file's exported signatures, emits a `mod c_abi` of
  verbatim repr(C) mirror structs + exact fn signatures; a header-scoped
  second bindgen pass also catches macro-generated publics tree-sitter can't
  see) + `ffi_stage.py` additions (CONVERSION_PROMPT generating per-struct
  `sync_in`/`sync_out`/`sync_init` — init writes every field fresh because C
  passes UNINITIALIZED memory to initializers; BOUNDARY_BLOCK shim
  discipline: bindgen signature verbatim, sync-in → call core → sync-out,
  `c_abi_malloc/realloc/free` (#[link_name] to libc) for anything C retains)
  wired via `--regen-ffi` on run_rust (regenerates ONLY the FFI layer).
  Proven on vector: **lane MATCH** on all scenarios (init/add/commit/log/
  cat-file byte-identical to C, same commit OIDs).
  What the differential caught on the way (each a generalizable class):
  (1) ALLOCATOR MIXING — MTU bodies called git__malloc/reallocarray/free on
  Rust Vec buffers and rebuilt them with from_raw_parts (vector's specs pin
  these C mechanisms as invariants, and the code stage obeyed). Fixed by a
  standing ALLOCATOR OWNERSHIP rule in the code-stage prompt plus targeted
  unit regeneration with an override retry that quotes the violation (the
  rule alone loses to the spec — enforcement needs the retry). A manual
  `impl Drop` freeing the Vec buffer with libc free was the first crash.
  (2) LENGTH DESYNC — units tracked `length` as a field while sync_out
  derives the C length from `contents.len()`; any unit that only bumped the
  field silently no-opped through the boundary (remove → index clear loop
  double-free; pop → revwalk repeating the same commit forever; clear,
  remove_matching). Rule: the collection IS the state; derived fields must
  be recomputed from it.
  (3) BOUNDARY MACHINERY BELONGS IN THE SHIM — for functions whose contract
  is inherently about C memory (the search family's ENOTFOUND=-3 +
  insertion-point out-param + pointer-identity fallback; insert_sorted's
  on_dup receiving a pointer INTO the live array slot; dispose's exact
  free-and-null sequence), the shim mirrors the C directly over the mirror
  struct — a synced Rust copy cannot honor live-slot semantics. These are
  deterministic and small; the idiomatic core keeps the behavioral units.
  Debugging kit that made this tractable without gdb: an LD_PRELOAD
  SIGSEGV/SIGABRT backtrace handler (backtrace_symbols_fd), minimal C
  probes per function family diffed C-vs-Rust, and `timeout -s ABRT` to
  backtrace infinite loops.
  Remaining rollout: --regen-ffi + differential triage for pqueue,
  signature, tree-cache, commit (their lanes still DIVERGE with pre-boundary
  FFI layers).

- Automatic escalation of irreducibles (flag + report only).
- Local open-weight model backends.
- Non-adjacent merge candidates beyond call-DAG edges (log the demand first).
- Integration into `uwisc-docker` `/translators/MTUs/` behind the config flag —
  happens after both strategies work standalone here.
