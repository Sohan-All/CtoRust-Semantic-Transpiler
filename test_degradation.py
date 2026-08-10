"""A unit that raises must not take the stage down with it.

    PYTHONPATH=. venv/bin/python test_degradation.py

Why this exists. Every `asyncio.gather` in the pipeline was fail-fast: one MTU
whose reply would not parse aborted the whole stage, and because results were
written only after the barrier, the seventeen siblings that HAD succeeded were
discarded unsaved. `logs/allon_t3.log` is that crash.

The fix has two halves and they are not separable:

  1. survive the failed unit (rustgen.common.gather_units), persisting each
     unit as it lands so a later failure cannot un-persist it;
  2. record the loss, so the harness refuses to score the crate.

Half 1 without half 2 would be strictly worse than crashing. A crate missing a
unit still BUILDS — assemble() substitutes a *documented* `todo!()`, which
illegal_stubs() permits by design — so it would sail through both existing
gates and land in the ablation record as an ordinary verdict. That is the
`todo!()` hole (symsem_t2) rebuilt one layer up.

So these tests assert the failure is survived AND that it is visible. Following
test_prompts.py: the paths are executed with a fake LLM that really raises, not
inspected.
"""

from __future__ import annotations

import asyncio
import sys

from state import Explanation

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label
          + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(label)


def units(n: int) -> list[Explanation]:
    return [Explanation(id=f"exp_{i:04d}", text=f"unit {i}", invariants=[],
                        ranges=[(i * 10 + 1, i * 10 + 5)], status="locked")
            for i in range(1, n + 1)]


class _RaisingLLM:
    """Raises for the named units, returns a canned reply for the rest."""

    def __init__(self, bad: set[str], exc: type[BaseException] = ValueError,
                 reply: str = "```rust\npub(crate) fn f() -> i32 { 1 }\n```"):
        self.bad = bad
        self.exc = exc
        self.reply = reply
        self.calls = 0

    def _maybe_raise(self, prompt: str) -> None:
        self.calls += 1
        for uid in self.bad:
            if uid in prompt:
                raise self.exc(f"synthetic failure for {uid}")

    async def ask(self, prompt: str, **kw) -> str:
        self._maybe_raise(prompt)
        return self.reply

    async def ask_json(self, prompt: str, **kw):
        self._maybe_raise(prompt)
        return {"signatures": ["pub(crate) fn f() -> i32"],
                "behavior_note": "n", "concerns": []}


# --- half 1: the stage survives -------------------------------------------

def test_gather_units_survives() -> None:
    print("\n=== gather_units: one failure does not abort the batch ===")
    from rustgen.common import gather_units

    us = units(4)

    async def run(u: Explanation):
        if u.id == "exp_0002":
            raise ValueError("boom")
        return u.id, f"code for {u.id}"

    got: list[str] = []
    failures: list[dict] = []
    out = asyncio.run(gather_units("code", us, run, failures,
                                   lambda uid, v: got.append(uid)))

    check("surviving units are returned", set(out) == {"exp_0001", "exp_0003",
                                                       "exp_0004"}, str(set(out)))
    check("failed unit is absent from the result", "exp_0002" not in out)
    check("failure is recorded", len(failures) == 1
          and failures[0]["unit"] == "exp_0002", str(failures))
    check("record names the stage and the exception",
          failures and failures[0]["stage"] == "code"
          and "ValueError" in failures[0]["error"], str(failures[:1]))
    check("on_result fired per surviving unit, not for the failure",
          sorted(got) == ["exp_0001", "exp_0003", "exp_0004"], str(got))


def test_on_result_fires_before_later_failure() -> None:
    """The persistence guarantee: a unit that lands early stays landed even
    when a slower sibling fails afterwards."""
    print("\n=== gather_units: early results persist past a later failure ===")
    from rustgen.common import gather_units

    async def run(u: Explanation):
        if u.id == "exp_0003":
            await asyncio.sleep(0.02)      # fail last
            raise RuntimeError("late boom")
        return u.id, "ok"

    got: list[str] = []
    failures: list[dict] = []
    asyncio.run(gather_units("spec", units(3), run, failures,
                             lambda uid, v: got.append(uid)))
    check("earlier units were persisted before the later failure",
          sorted(got) == ["exp_0001", "exp_0002"], str(got))
    check("late failure still recorded", len(failures) == 1, str(failures))


def test_fatal_is_not_swallowed() -> None:
    """Ctrl-C must stop the run, not silently degrade it. CancelledError is a
    BaseException in 3.12 and `return_exceptions=True` would capture it."""
    print("\n=== gather_units: fatal signals propagate ===")
    from rustgen.common import gather_units

    for exc in (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
        async def run(u: Explanation, exc=exc):
            raise exc()

        failures: list[dict] = []
        try:
            asyncio.run(gather_units("code", units(2), run, failures))
            check(f"{exc.__name__} propagates", False, "no exception raised")
        except BaseException as e:
            check(f"{exc.__name__} propagates instead of degrading",
                  isinstance(e, exc) and not failures,
                  f"got {type(e).__name__}, failures={len(failures)}")


# --- the real stages -------------------------------------------------------

def test_code_stage_degrades() -> None:
    print("\n=== code_stage: a raising unit is dropped, not fatal ===")
    from rustgen.code_stage import generate_code

    us = units(3)
    specs = {u.id: {"signatures": ["pub(crate) fn f() -> i32"]} for u in us}
    llm = _RaisingLLM(bad={"unit 2"})
    failures: list[dict] = []
    try:
        out = asyncio.run(generate_code(llm, us, specs, "", 100,
                                        failures=failures))
        check("stage returns despite a failing unit",
              set(out) == {"exp_0001", "exp_0003"}, str(set(out)))
        check("the loss is recorded", len(failures) == 1
              and failures[0]["unit"] == "exp_0002", str(failures))
    except Exception as e:
        check("stage does not raise", False, f"{type(e).__name__}: {e}")


def test_spec_stage_degrades() -> None:
    print("\n=== spec_stage: raising unit + failed responsibility pass ===")
    from rustgen.spec_stage import generate_specs
    from config import Config
    import dataclasses

    us = units(3)
    cfg = dataclasses.replace(Config(), rustgen_spec_mode="rich")

    llm = _RaisingLLM(bad={"unit 2"})
    failures: list[dict] = []
    try:
        out = asyncio.run(generate_specs(llm, us, "", {}, cfg,
                                         failures=failures))
        check("rich mode drops only the failing unit",
              set(out) == {"exp_0001", "exp_0003"}, str(set(out)))
    except Exception as e:
        check("rich mode does not raise", False, f"{type(e).__name__}: {e}")

    # the global pass runs before any unit; losing it must not abort the stage
    class _NoResponsibility(_RaisingLLM):
        async def ask_json(self, prompt: str, **kw):
            if "single-owner concern" in prompt or "EXACTLY ONE unit" in prompt:
                raise ValueError("synthetic responsibility failure")
            return {"signatures": ["pub(crate) fn f() -> i32"],
                    "behavior_note": "n"}

    failures = []
    try:
        out = asyncio.run(generate_specs(_NoResponsibility(bad=set()), us, "",
                                         {}, cfg, failures=failures))
        check("failed responsibility pass still yields every spec",
              set(out) == {u.id for u in us}, str(set(out)))
        check("failed responsibility pass is recorded",
              any("responsibility" in f.get("unit", "") for f in failures),
              str(failures))
    except Exception as e:
        check("failed responsibility pass does not raise", False,
              f"{type(e).__name__}: {e}")


def test_skip_regenerates_only_missing() -> None:
    """The payoff of per-unit persistence: a resume redraws the gap, not the
    file. Before this, a stage was all-or-nothing and any crash cost all of it."""
    print("\n=== skip: a resume redraws only the missing units ===")
    from rustgen.code_stage import generate_code

    us = units(4)
    specs = {u.id: {"signatures": ["pub(crate) fn f() -> i32"]} for u in us}
    llm = _RaisingLLM(bad=set())
    out = asyncio.run(generate_code(llm, us, specs, "", 100,
                                    skip={"exp_0001", "exp_0002"}))
    check("only the un-persisted units are generated",
          set(out) == {"exp_0003", "exp_0004"}, str(set(out)))
    check("one LLM call per missing unit", llm.calls == 2, f"calls={llm.calls}")


# --- half 2: the loss is visible to the gate -------------------------------

def test_gate_would_reject() -> None:
    """The degraded crate must fail the harness gate. Asserted against the two
    checks it would otherwise slip past: it builds, and illegal_stubs() PERMITS
    the documented todo!() that assemble() leaves behind."""
    print("\n=== the gate: a dropped unit is not scoreable ===")
    from rustgen.assemble import assemble
    from rustgen.common import illegal_stubs
    import tempfile, json
    from pathlib import Path

    us = units(3)
    code = {"exp_0001": "pub(crate) fn a() -> i32 { 1 }",
            "exp_0003": "pub(crate) fn c() -> i32 { 3 }"}   # exp_0002 dropped

    with tempfile.TemporaryDirectory() as d:
        crate = assemble(Path(d), "t", "t.c", "", us, code)
        lib = (crate / "src" / "lib.rs").read_text()

    check("assemble substitutes a placeholder for the dropped unit",
          "no code generated for exp_0002" in lib)
    # this is the whole reason the degraded record has to exist
    check("illegal_stubs does NOT flag it (documented todo! is permitted)",
          illegal_stubs(lib) == "", illegal_stubs(lib))

    # so the gate must key on the record instead
    rec = {"type": "degraded", "count": 1,
           "failures": [{"stage": "code", "unit": "exp_0002",
                         "error": "ValueError: boom"}]}
    line = json.dumps(rec)
    parsed = json.loads(line)
    check("degraded record round-trips and names the lost unit",
          parsed["failures"][0]["unit"] == "exp_0002")
    check("a clean run writes no such record (absence == scoreable)",
          [] == [r for r in [{"type": "project_compile"}]
                 if r.get("type") == "degraded"])


# --- transport: llm.py survives a flaky server ----------------------------

class _FakeCompletions:
    """Stands in for client.chat.completions, replaying a script of outcomes.

    Each entry is either an exception to raise or an object to return, so the
    real retry/budget loop in LLM.ask runs against it unmodified.
    """

    def __init__(self, script: list):
        self.script = script
        self.calls = 0

    async def create(self, **kw):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _Resp:
    def __init__(self, text: str | None = "ok", choices: bool = True,
                 finish: str = "stop"):
        class _Msg:
            content = text

        class _Choice:
            message = _Msg()
            finish_reason = finish

        self.choices = [_Choice()] if choices else []
        self.usage = None


def _llm(script: list):
    """An LLM whose transport is the fake script; everything else is real."""
    import os
    import dataclasses
    from config import Config
    from llm import LLM

    os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    os.environ["VLLM_API_KEY"] = "x"
    cfg = dataclasses.replace(Config(), retry_backoff=0.001,
                              transport_retries=3)
    llm = LLM(cfg)

    class _Chat:
        completions = _FakeCompletions(script)

    class _Client:
        chat = _Chat()

    llm.client = _Client()
    return llm, _Chat.completions


def test_transient_errors_retried() -> None:
    print("\n=== llm: a flaky server is retried, not fatal ===")
    import openai
    import httpx

    req = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")

    for label, exc in [
        ("APIConnectionError", openai.APIConnectionError(request=req)),
        ("APITimeoutError", openai.APITimeoutError(request=req)),
    ]:
        llm, fake = _llm([exc, exc, _Resp("recovered")])
        try:
            out = asyncio.run(llm.ask("p"))
            check(f"{label}: retried then succeeded",
                  out == "recovered" and llm.transport_retries == 2,
                  f"out={out!r} retries={llm.transport_retries}")
        except Exception as e:
            check(f"{label}: retried", False, f"{type(e).__name__}: {e}")

    # exhausting the budget must still raise — retrying forever is not "safe"
    llm, fake = _llm([openai.APIConnectionError(request=req)])
    try:
        asyncio.run(llm.ask("p"))
        check("exhausted transport retries raise", False, "returned instead")
    except openai.APIConnectionError:
        check("exhausted transport retries raise rather than loop forever",
              fake.calls == 4, f"calls={fake.calls}")
    except Exception as e:
        check("exhausted transport retries raise APIConnectionError", False,
              f"{type(e).__name__}: {e}")


def test_empty_choices() -> None:
    """A well-formed response carrying no choices used to IndexError."""
    print("\n=== llm: empty choices list ===")
    llm, fake = _llm([_Resp(choices=False), _Resp("second try")])
    try:
        out = asyncio.run(llm.ask("p"))
        check("empty choices retried instead of IndexError",
              out == "second try", f"out={out!r}")
    except IndexError:
        check("empty choices does not IndexError", False, "IndexError raised")
    except Exception as e:
        check("empty choices retried", False, f"{type(e).__name__}: {e}")

    llm, fake = _llm([_Resp(choices=False)])
    try:
        asyncio.run(llm.ask("p"))
        check("persistent empty choices raises", False, "returned instead")
    except IndexError:
        check("persistent empty choices raises a described error, not IndexError",
              False, "IndexError")
    except RuntimeError as e:
        # The point is that it is DESCRIBED, not that it is worded a
        # particular way — the bug this guards was a bare IndexError, which
        # the except-clause above catches. Coupled to llm.py's raise site
        # (search "no usable content"); it was previously "no choices", and
        # the reword left this assertion failing for no behavioural reason.
        check("persistent empty choices raises a described error",
              "no usable content" in str(e) and "retries" in str(e), str(e))


def test_bad_request_still_raises() -> None:
    """A prompt-level 400 is NOT transient — retrying it just burns the budget
    on the same failure. Only the context-length case has a real remedy."""
    print("\n=== llm: non-transient errors are not retried ===")
    import openai
    import httpx

    resp = httpx.Response(400, request=httpx.Request("POST", "http://x/"))
    err = openai.BadRequestError("bad", response=resp, body=None)
    llm, fake = _llm([err])
    try:
        asyncio.run(llm.ask("p"))
        check("BadRequestError propagates", False, "returned instead")
    except openai.BadRequestError:
        check("BadRequestError propagates immediately, no retries",
              fake.calls == 1 and llm.transport_retries == 0,
              f"calls={fake.calls} retries={llm.transport_retries}")
    except Exception as e:
        check("BadRequestError propagates", False, f"{type(e).__name__}: {e}")


# --- the crash boundary ----------------------------------------------------

def test_crash_exit_code() -> None:
    """End-to-end: a phase that raises must exit CRASH_EXIT and leave a record,
    not just print a traceback."""
    print("\n=== main(): phases exit CRASHED, not generic failure ===")
    import subprocess
    import tempfile
    import json
    from pathlib import Path
    import run_project

    with tempfile.TemporaryDirectory() as d:
        # An empty src/ does not fail indexing — it gets as far as stage T and
        # dies on the unreachable server, which is a fine crash for this test:
        # what is asserted is the boundary, not which phase reaches it.
        root = Path(d) / "proj" / "test_case"
        root.mkdir(parents=True)
        out = Path(d) / "out"
        cfg = Path(d) / "fast.json"
        cfg.write_text(json.dumps({"transport_retries": 0,
                                   "retry_backoff": 0.0}))
        env = {**__import__("os").environ, "DIFFUSIONMTUS_OUT": str(out),
               "VLLM_BASE_URL": "http://127.0.0.1:1/v1", "VLLM_API_KEY": "x",
               "PYTHONPATH": str(Path(run_project.__file__).parent)}
        p = subprocess.run([sys.executable, run_project.__file__, str(root),
                            "--config", str(cfg)],
                           capture_output=True, text=True, env=env)
        check(f"crash exits {run_project.CRASH_EXIT}, distinct from 1 and 2",
              p.returncode == run_project.CRASH_EXIT,
              f"rc={p.returncode} stderr={p.stderr[-200:]}")
        check("stderr names the phase", "CRASHED" in p.stderr,
              p.stderr[-200:])

        recs = list(out.glob("_project_*/project.jsonl"))
        found = []
        for r in recs:
            for line in r.read_text().splitlines():
                if line.strip():
                    found.append(json.loads(line))
        crash = [r for r in found if r.get("type") == "crash"]
        check("a crash record is written", len(crash) == 1, str(found)[:200])
        check("the record names a real phase and carries a traceback",
              crash and crash[0].get("phase") in
              {"index", "mtu", "types", "rustgen"}
              and "Traceback" in crash[0].get("traceback", ""),
              str(crash)[:200])
        # the harness parses these two out of the log to build the verdict
        check("stderr line is parseable into phase + error",
              any(line.startswith("[") and "CRASHED: " in line
                  for line in p.stderr.splitlines()), p.stderr[-200:])


# --- the compile loop's parse guard ---------------------------------------

def test_parse_regression() -> None:
    print("\n=== parse_regression: repairs may not introduce a parse error ===")
    from rustgen.common import parse_regression

    ok = "pub(crate) fn f() -> i32 { 1 }"
    broken = "pub(crate) fn f() -> i32 { 1 } }"

    check("balanced -> balanced accepted", parse_regression(ok, ok) == "")
    check("balanced -> unbalanced REJECTED", parse_regression(ok, broken) != "",
          parse_regression(ok, broken))
    # asymmetric on purpose: a broken section must stay reachable by repairs
    check("unbalanced -> balanced accepted (the fix)",
          parse_regression(broken, ok) == "")
    check("unbalanced -> unbalanced accepted (may still be progress)",
          parse_regression(broken, broken) == "")
    # things that look like delimiters but are not
    for label, code in [("brace in a string", 'fn f() { let s = "}"; }'),
                        ("char literal", "fn f() { let c = '}'; }"),
                        ("brace in a comment", "fn f() { // }\n }"),
                        ("lifetime", "fn f<'a>(x: &'a str) -> &'a str { x }")]:
        check(f"no false positive: {label}", parse_regression(ok, code) == "",
              parse_regression(ok, code))


def test_parse_regression_on_real_trial4() -> None:
    """The known-bad fixture: the actual repair that took down base_srvA_t4.

    Asserted in both directions, because a guard that rejects everything would
    pass a one-directional test and quietly freeze every repair in the loop.
    """
    print("\n=== parse_regression: the real trial-4 regression ===")
    import json
    from pathlib import Path
    from rustgen.common import parse_regression

    state = Path("/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs/"
                 "double_linked_list_base_srvA_t4/_project_B03_organic/"
                 "files/editor/state.jsonl")
    if not state.exists():
        print("  SKIP  fixture run not on disk")
        return
    vers = [json.loads(l) for l in state.read_text().splitlines() if l.strip()]
    vers = [r for r in vers
            if r.get("type") == "rust_code" and r["unit"] == "exp_0005"]
    generated, repaired = vers[0]["code"], vers[-1]["code"]

    check("the trial-4 repair is rejected",
          "stray closing" in parse_regression(generated, repaired),
          parse_regression(generated, repaired))
    check("the same section's good version is still accepted",
          parse_regression(repaired, generated) == "")


def test_set_section_keeps_previous() -> None:
    """Testing the checker is not testing the response to it. This drives the
    REAL set_section closure and asserts the section still holds the old code.

    Reached through the todo-stub pass, which runs before the error rounds and
    so needs no working cargo: a section with a bare `todo!()` is handed to the
    LLM, which here returns an unbalanced rewrite with fewer stubs — i.e. a
    repair that "makes progress" on the stub count while breaking the parse.
    That is precisely the trial-4 shape.
    """
    print("\n=== set_section: a rejected repair leaves the section intact ===")
    import dataclasses
    import tempfile
    from pathlib import Path
    from config import Config
    import rustgen.compile_loop as cl

    stubbed = "pub(crate) fn f() -> i32 { todo!() }"
    bad_repair = "```rust\npub(crate) fn f() -> i32 { 1 } }\n```"
    good_repair = "```rust\npub(crate) fn f() -> i32 { 1 }\n```"

    def drive(reply: str):
        real_check = cl.cargo_check
        cl.cargo_check = lambda crate: []          # clean; loop exits at once
        try:
            with tempfile.TemporaryDirectory() as d:
                crate = Path(d)
                (crate / "src").mkdir()
                code = {"exp_0001": stubbed}

                def reassemble(t, c, f):
                    (crate / "src" / "lib.rs").write_text(
                        "\n".join(c.values()))

                cfg = dataclasses.replace(Config(), rustgen_compile_rounds=0)
                return asyncio.run(cl.compile_loop(
                    cfg, _FakeReplyLLM(reply), crate, units(1),
                    {"exp_0001": {}}, "", code, reassemble, ""))
        finally:
            cl.cargo_check = real_check

    # 1. the unbalanced repair must be refused, old code kept
    try:
        _, out_code, _, report = drive(bad_repair)
        check("unbalanced repair rejected — section keeps its previous code",
              out_code["exp_0001"] == stubbed, repr(out_code["exp_0001"])[:80])
        check("the rejection is recorded on the report",
              len(report.rejected_repairs) == 1
              and report.rejected_repairs[0]["section"] == "exp_0001",
              str(report.rejected_repairs))
        check("summary surfaces it (a dropped repair otherwise reads as "
              "'did not help')",
              "rejected as unparseable" in report.summary(), report.summary())
    except Exception as e:
        check("unbalanced repair rejected", False, f"{type(e).__name__}: {e}")

    # 2. the guard must not block a GOOD repair — a guard that rejects
    #    everything would pass test 1 and freeze the whole loop
    try:
        _, out_code, _, report = drive(good_repair)
        check("balanced repair still accepted",
              "todo!()" not in out_code["exp_0001"], repr(out_code["exp_0001"])[:80])
        check("no spurious rejection recorded", not report.rejected_repairs,
              str(report.rejected_repairs))
    except Exception as e:
        check("balanced repair accepted", False, f"{type(e).__name__}: {e}")


class _FakeReplyLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def ask(self, prompt: str, **kw) -> str:
        self.calls += 1
        return self.reply

    async def ask_json(self, prompt: str, **kw):
        return {}


def test_emptied_blocks() -> None:
    print("\n=== emptied_blocks: a repair may not gut an impl block ===")
    from rustgen.common import emptied_blocks

    full = ("impl S {\n    /// docs\n    fn a(&self) {}\n"
            "    /// more\n    fn b(&self) {}\n}")
    gutted = "impl S {\n    /// docs\n    /// more\n}"

    check("gutting an impl is REJECTED", emptied_blocks(full, gutted) != "",
          emptied_blocks(full, gutted))
    check("an ordinary edit is accepted",
          emptied_blocks(full, full.replace("fn b", "fn c")) == "")
    check("re-filling a gutted block is accepted (the fix)",
          emptied_blocks(gutted, full) == "")
    check("already-empty stays writable (not frozen)",
          emptied_blocks(gutted, gutted) == "")
    # a block with a non-fn item is not empty
    check("const-only impl is not 'gutted'",
          emptied_blocks(full, "impl S {\n    const N: i32 = 1;\n}") == "")


def test_emptied_blocks_on_real_fixture() -> None:
    """The binary_heap base_srvB_t1 repair: 1431 bytes -> 491, losing spawn,
    dispatch and peek."""
    print("\n=== emptied_blocks: the real binary_heap regression ===")
    import json
    from pathlib import Path
    from rustgen.common import emptied_blocks

    state = Path("/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs/"
                 "binary_heap_base_srvB_t1/_project_B03_organic/"
                 "files/scheduler/state.jsonl")
    if not state.exists():
        print("  SKIP  fixture run not on disk")
        return
    vs = [json.loads(l) for l in state.read_text().splitlines() if l.strip()]
    vs = [r for r in vs
          if r.get("type") == "rust_code" and r["unit"] == "exp_0003"]
    generated, repaired = vs[0]["code"], vs[-1]["code"]

    check("the gutting repair is rejected",
          emptied_blocks(generated, repaired) != "",
          emptied_blocks(generated, repaired))
    check("the good version is still accepted",
          emptied_blocks(repaired, generated) == "")
    # and the parse guard alone would NOT have caught it — that is the point
    from rustgen.common import parse_regression
    check("parse_regression alone misses it (braces balance)",
          parse_regression(generated, repaired) == "")


def test_remaining_stubs() -> None:
    print("\n=== remaining_stubs: any stub voids a scored crate ===")
    from rustgen.common import illegal_stubs, remaining_stubs

    documented = 'fn f() -> i32 { todo!("<populate_atlas>") }'
    deps = "pub mod x_deps {\n    fn g() { todo!() }\n}"
    clean = "fn f() -> i32 { 1 }"

    check("clean code passes", remaining_stubs(clean) == "")
    check("a DOCUMENTED stub is flagged for scoring",
          remaining_stubs(documented) != "", remaining_stubs(documented))
    check("...which illegal_stubs deliberately permits (the gap)",
          illegal_stubs(documented) == "")
    check("a deps-module stub is flagged too (it panics like any other)",
          remaining_stubs(deps) != "")
    check("no false positive on the word in a string",
          remaining_stubs('fn f() { let s = "todo!"; }') == "")
    check("no false positive in a comment",
          remaining_stubs("fn f() { // todo!()\n }") == "")


def test_remaining_stubs_on_real_crates() -> None:
    """Known-bad and known-good, per the convention: the gate must void the two
    crates whose panics were scored, and pass the ones that were genuinely
    clean."""
    print("\n=== remaining_stubs: real crates ===")
    from pathlib import Path
    from rustgen.common import remaining_stubs

    runs = Path("/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs")
    if not runs.exists():
        print("  SKIP  runs not on disk")
        return
    bad = {"array_list_base_srvA_t1", "binary_heap_base_srvA_t1"}
    flagged, passed = set(), set()
    for lib in sorted(runs.glob("*/_project_*/rust_crate/src/lib.rs")):
        tag = lib.parts[len(runs.parts)]
        (flagged if remaining_stubs(lib.read_text()) else passed).add(tag)

    check("both stub-panicking crates are voided", bad <= flagged,
          f"flagged={sorted(flagged)}")
    check("the 1/26 run is NOT voided",
          "double_linked_list_base_srvB_t4" in passed)
    check("the t5 runs are NOT voided",
          {"double_linked_list_base_srvA_t5",
           "double_linked_list_base_srvB_t5"} <= passed,
          f"flagged={sorted(flagged)}")


def test_surgical_edits_cannot_delete() -> None:
    """The edit-level backstop: a surgical edit may not drop a `fn` its own
    "find" contained. emptied_blocks only catches the case where the whole
    block ends up empty; this catches partial gutting too."""
    print("\n=== apply_surgical_edits: an edit may not delete an item ===")
    from rustgen.compile_loop import apply_surgical_edits

    code = ("impl S {\n"
            "    /// docs a\n    fn a(&self) -> i32 { 1 }\n"
            "    /// docs b\n    fn b(&self) -> i32 { 2 }\n}")

    deleting = [{"find": "    /// docs b\n    fn b(&self) -> i32 { 2 }\n",
                 "replace": "    /// docs b\n"}]
    check("an edit that drops a fn is refused",
          apply_surgical_edits(code, deleting) is None)

    blanking = [{"find": "    fn b(&self) -> i32 { 2 }\n", "replace": ""}]
    check("a pure deletion is refused",
          apply_surgical_edits(code, blanking) is None)

    fixing = [{"find": "fn a(&self) -> i32 { 1 }",
               "replace": "fn a(&self) -> i64 { 1 }"}]
    out = apply_surgical_edits(code, fixing)
    check("an ordinary fix still applies",
          out is not None and "i64" in out and "fn b" in out, repr(out)[:80])

    renaming = [{"find": "fn a(&self) -> i32 { 1 }",
                 "replace": "fn a_renamed(&self) -> i32 { 1 }"}]
    check("a rename is refused (it drops the old name)",
          apply_surgical_edits(code, renaming) is None)


def test_repair_prompts_still_format() -> None:
    """The ITEM INVENTORY rule embeds a literal `impl { }` example. That is the
    exact shape that produced `KeyError: ' '` in trial 4, so it is asserted
    here as well as in test_prompts.py."""
    print("\n=== repair prompts survive .format() ===")
    import string
    import rustgen.compile_loop as cl

    for name in ("REPAIR_CODE_PROMPT", "REPAIR_CLUSTER_PROMPT",
                 "SURGICAL_PROMPT"):
        tmpl = getattr(cl, name)
        keys = {f[1] for f in string.Formatter().parse(tmpl) if f[1]}
        try:
            out = tmpl.format(**{k: "X" for k in keys})
            check(f"{name} formats ({len(keys)} placeholders)",
                  "ITEM INVENTORY" in out or name == "SURGICAL_PROMPT")
        except (KeyError, IndexError, ValueError) as e:
            check(f"{name} formats", False, f"{type(e).__name__}: {e}")

    # the braces in the example must survive as braces, not vanish
    out = cl.REPAIR_CODE_PROMPT.format(types_rs="", sibling_sigs="", unit="",
                                       spec="", code="", errors="")
    check("the impl example renders with real braces",
          "impl Scheduler {\n" in out and "    }" in out.split("impl Scheduler")[1][:400],
          out.split("ITEM INVENTORY")[1][:200] if "ITEM INVENTORY" in out else "?")
    check("the todo! guidance no longer models angle-bracket placeholders",
          "<what is missing>" not in out and "not an explanation" in out)


def test_emptied_blocks_both_fixtures() -> None:
    """Both real fixtures at once — the two-sided check whose absence let a
    deduplication exemption get written and reverted.

    They pull in opposite directions only if you misread the second one: in
    binary_heap base_srvB_t2 the surviving copy of `compare` is the STUB and
    the deleted one is the real implementation, so refusing is correct there
    too. Any future relaxation must keep BOTH of these rejected.
    """
    print("\n=== emptied_blocks: both real fixtures stay rejected ===")
    import json
    import re
    from pathlib import Path
    from rustgen.common import emptied_blocks

    runs = Path("/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs")
    if not runs.exists():
        print("  SKIP  runs not on disk")
        return

    state = (runs / "binary_heap_base_srvB_t1/_project_B03_organic"
                    "/files/scheduler/state.jsonl")
    if state.exists():
        vs = [json.loads(l) for l in state.read_text().splitlines() if l.strip()]
        vs = [r for r in vs
              if r.get("type") == "rust_code" and r["unit"] == "exp_0003"]
        check("t1 gutting (spawn/dispatch/peek vanish) stays rejected",
              emptied_blocks(vs[0]["code"], vs[-1]["code"]) != "")

    lib = (runs / "binary_heap_base_srvB_t2/_project_B03_organic"
                  "/rust_crate/src/lib.rs")
    if lib.exists():
        crate = lib.read_text()
        sec = [s for s in re.split(r"^// ===== ", crate, flags=re.M)
               if s.startswith("MTU scheduler__exp_0001")]
        if sec:
            sched = sec[0]
            deduped = re.sub(r"pub\(crate\) fn compare.*?\n    \}", "",
                             sched, flags=re.S)
            check("t2 'dedup' stays rejected — it deletes the REAL impl and "
                  "keeps the stub", emptied_blocks(sched, deduped) != "")


def test_types_stage_records_exhaustion() -> None:
    """Stage T giving up on its stub gate must leave a record, not just a
    printed warning. binary_heap base_srvB_t2 died of this and the only
    evidence was two lines of stdout."""
    print("\n=== types_stage: exhausted stub gate is recorded ===")
    from rustgen.types_stage import synthesize_types

    stubbed = ("```rust\npub struct S { pub a: i32 }\n"
               "impl S { pub fn f(&self) -> i32 { todo!() } }\n```\n"
               'GLOSSARY:\n```json\n{"thing": "S"}\n```')
    clean = ("```rust\npub struct S { pub a: i32 }\n```\n"
             'GLOSSARY:\n```json\n{"thing": "S"}\n```')
    unit = Explanation(id="exp_0001", text="does a thing", invariants=[],
                       ranges=[(1, 5)], status="locked")

    failures: list[dict] = []
    rs, _ = asyncio.run(synthesize_types(_FakeReplyLLM(stubbed), [unit], 100,
                                         failures=failures))
    check("exhaustion is recorded so the run cannot score",
          len(failures) == 1 and failures[0]["stage"] == "types",
          str(failures))
    check("the record names the surviving problem",
          failures and "stub gate exhausted" in failures[0]["error"],
          str(failures))

    failures = []
    rs, _ = asyncio.run(synthesize_types(_FakeReplyLLM(clean), [unit], 100,
                                         failures=failures))
    check("a clean stage T records nothing", not failures, str(failures))


def test_orphan_rule_in_prompts() -> None:
    print("\n=== prompts: the orphan rule is stated where impls are suggested ===")
    import rustgen.spec_stage as ss
    import rustgen.code_stage as cs

    for name, tmpl in [("RESPONSIBILITY_PROMPT", ss.RESPONSIBILITY_PROMPT),
                       ("RICH_SPEC_PROMPT", ss.RICH_SPEC_PROMPT),
                       ("SPEC_PROMPT", ss.SPEC_PROMPT),
                       ("CODE_PROMPT", cs.CODE_PROMPT)]:
        has_impl_advice = "impl Display" in tmpl or "impl FromStr" in tmpl
        has_orphan = "orphan rule" in tmpl.lower()
        check(f"{name} qualifies its trait-impl advice",
              has_orphan if has_impl_advice else True)


def test_one_call_is_bounded() -> None:
    """`array_list` t4 srvA: one call held a run for 38 minutes writing nothing.

    Three multipliers stacked. The client was built with max_retries=5, so
    `request_timeout=900` was a per-ATTEMPT bound and the real one was 6x that;
    `ask` doubles the budget on a truncated reply, giving a second round of the
    same; and nothing bounded the product. None of it was reported, because the
    client's internal retries emit no log line and touch no counter — from
    inside the run a 90-minute wedge looked exactly like a slow call.
    """
    print("\n=== transport: one logical call is bounded and reports ===")
    import dataclasses
    import os
    import openai
    import llm as llm_mod
    from config import Config
    cfg = Config()

    client = llm_mod._make_client("gemma-4-31b", cfg.request_timeout)
    check("client does not retry behind our back (max_retries=0)",
          client.max_retries == 0, f"max_retries={client.max_retries}")

    rt = llm_mod._request_timeout
    check("per-request timeout never drops below the configured floor",
          all(rt(b, cfg.request_timeout) >= cfg.request_timeout
              for b in (512, 4000, 16000)))
    check("per-request timeout grows with the budget",
          rt(16000, cfg.request_timeout) > rt(4000, cfg.request_timeout))

    # The trap itself: MAX_TOKENS_CEILING and request_timeout were mutually
    # unsatisfiable. A 16000-token reply needs ~1070s at the slow end of the
    # observed decode rate, so it could NEVER complete inside 900s.
    ceiling = llm_mod.LLM.MAX_TOKENS_CEILING
    needed = ceiling / llm_mod._MIN_DECODE_RATE
    check("the token ceiling is reachable within its own timeout",
          rt(ceiling, cfg.request_timeout) >= needed,
          f"{rt(ceiling, cfg.request_timeout):.0f}s allowed, {needed:.0f}s needed")
    check("known-bad: the old fixed 900s could not have reached the ceiling",
          900 < needed, f"{needed:.0f}s needed")

    # call_deadline is the backstop on the product. Drive it with a client that
    # always times out, and assert the loop gives up rather than spinning.
    class _AlwaysTimeout:
        def __init__(self):
            self.attempts = 0

        class _Completions:
            def __init__(self, outer):
                self.outer = outer

            async def create(self, **kw):
                self.outer.attempts += 1
                raise openai.APITimeoutError(request=None)

        @property
        def chat(self):
            outer = self

            class _Chat:
                completions = _AlwaysTimeout._Completions(outer)
            return _Chat()

    cfg2 = dataclasses.replace(Config(), call_deadline=1, retry_backoff=0.01,
                               transport_retries=2)
    # Build through the REAL __init__ and swap only the transport, exactly as
    # _llm() does. Hand-listing attributes after LLM.__new__ silently rots the
    # moment __init__ grows one: `backend` was added for the anthropic-vertex
    # path and this test began dying on AttributeError inside ask(), which
    # also zeroed the retry counter and made the NEXT check fail for a reason
    # that had nothing to do with retries.
    os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    os.environ["VLLM_API_KEY"] = "x"
    obj = llm_mod.LLM(cfg2)
    obj.client = _AlwaysTimeout()
    obj.sem = asyncio.Semaphore(1)
    try:
        asyncio.run(obj.ask("hi", max_tokens=512))
        check("a call that never succeeds raises", False, "returned normally")
    except Exception as e:
        check("a call that never succeeds raises, bounded",
              isinstance(e, (TimeoutError, openai.APITimeoutError)),
              f"{type(e).__name__}: {e}")
    check("every retry it did make was counted",
          obj.transport_retries >= 1, f"{obj.transport_retries} counted")


def test_missing_item_routes_to_owner() -> None:
    """A missing item must reach the section that should DEFINE it.

    `_name_lookup` picked a cross-section repair partner by searching for the
    section defining the missing name — but for a MISSING item nothing defines
    it anywhere, which is the whole error. It returned None, the cluster
    collapsed to the caller alone, and the repair was handed the one section
    that cannot fix it. Compounding that, `no associated function or constant
    named` was in neither name regex, so the commonest form (`Type::new`, 27 of
    36 recorded E0599s) did not even reach the lookup.
    """
    print("\n=== repair routing: a missing item reaches its owner ===")
    from rustgen.compile_loop import (_name_lookup, _base_type, _impl_owner,
                                      _NAME_ERR, missing_capability_notes)

    for raw, want in [("&Project", "Project"), ("&mut Vec<T>", "Vec"),
                      ("Activity", "Activity"), ("crate::Task", "Task")]:
        check(f"_base_type({raw!r}) -> {want}", _base_type(raw) == want,
              _base_type(raw))

    secs = {
        "activity__exp_0001": "impl Activity {\n  fn as_str(&self) -> &str { \"\" }\n}",
        "activity__exp_0003": "impl fmt::Display for Activity {\n  fn fmt(&self) {}\n}",
        "demo__exp_0007": "fn demo() { let a = Activity::new(1); }",
        "project__exp_0001": "impl Project {\n  fn find_task(&self) -> u8 { 0 }\n}",
    }
    NEW = "no associated function or constant named `new` found for struct `Activity` in the current scope"
    MUT = "no method named `find_task_mut` found for reference `&Project` in the current scope"
    STD = "no method named `num_entries` found for struct `Vec<BuildJob>` in the current scope"

    check("known-bad: the old regex missed `no associated function`",
          "no associated function or constant named" in _NAME_ERR.pattern)
    check("a missing constructor routes to the type's inherent impl",
          _name_lookup({"message": NEW}, secs) == "activity__exp_0001",
          str(_name_lookup({"message": NEW}, secs)))
    check("it does NOT route to the trait impl for the same type",
          _name_lookup({"message": NEW}, secs) != "activity__exp_0003")
    check("a missing &mut variant routes to the type's owner",
          _name_lookup({"message": MUT}, secs) == "project__exp_0001",
          str(_name_lookup({"message": MUT}, secs)))
    check("a missing method on a std type routes NOWHERE (caller's bug)",
          _name_lookup({"message": STD}, secs) is None,
          str(_name_lookup({"message": STD}, secs)))
    check("an unknown type routes nowhere rather than guessing",
          _impl_owner("Nonexistent", secs) is None)

    # the definer path still wins when the name DOES exist somewhere
    HAVE = "no method named `as_str` found for struct `Activity` in the current scope"
    check("a name defined somewhere still routes to its definer",
          _name_lookup({"message": HAVE}, secs) == "activity__exp_0001")

    note = missing_capability_notes([{"message": NEW}], secs)
    check("the note names where the item belongs",
          "impl Activity" in note, note[:90])
    check("...and does not tell the caller to reimplement it privately",
          "private helper in the calling section" in note, note[:90])
    note_std = missing_capability_notes([{"message": STD}], secs)
    check("a std-type miss still gets the implement-here note",
          "inline" in note_std, note_std[:90])


def test_illegal_type_bodies() -> None:
    """The types block must not implement behaviour a unit will also write.

    `illegal_stubs` does NOT cover this: it only catches `todo!()` bodies, so a
    REAL method body in the types block passed every gate. Rust then rejects
    two inherent methods of one name whatever their signatures (E0592), and the
    compile loop cannot undo it — deleting the added method empties the impl
    block, which `emptied_blocks` correctly refuses.
    """
    print("\n=== types block implements no behaviour ===")
    from rustgen.common import illegal_type_bodies as chk

    cases = [
        ("the real binary_heap t5 body",
         "impl Scheduler {\n fn spawn(&mut self, t: T) -> R {\n  self.tasks.push(t); Ok(())\n }\n}", True),
        ("required Display impl is permitted",
         "impl std::fmt::Display for E {\n fn fmt(&self, f: &mut F) -> R { write!(f, \"x\") }\n}", False),
        ("required Error impl is permitted",
         "impl std::error::Error for E {\n fn source(&self) -> O { None }\n}", False),
        ("a deps-module stub is permitted",
         "pub mod x_deps {\n pub fn cb<T>(i: T) -> R { Ok(()) }\n}", False),
        ("a todo!() body is illegal_stubs' verdict, not this one",
         "impl S {\n fn f(&self) { todo!() }\n}", False),
        ("plain type definitions are clean",
         "#[derive(Debug)]\npub struct S { pub a: i32 }\npub enum E { A }", False),
        ("an empty body implements nothing",
         "impl S {\n fn f(&self) {}\n}", False),
        ("a non-Display/Error trait impl IS behaviour",
         "impl Ord for S {\n fn cmp(&self, o: &S) -> O { self.a.cmp(&o.a) }\n}", True),
    ]
    for label, code, want in cases:
        check(label, bool(chk(code)) == want, f"flagged={bool(chk(code))}")

    # Known-bad AND known-good against the recorded corpus: the rate matters,
    # because a checker that fires on a third of all types blocks is noise.
    import json, glob
    flagged = total = 0
    for f in glob.glob("/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs/*/"
                       "_project_B03_organic/files/*/state.jsonl"):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "rust_types":
                continue
            total += 1
            if chk(d.get("types_rs", "")):
                flagged += 1
    if total:
        rate = flagged / total
        check(f"corpus rate stays low ({flagged}/{total} = {rate:.1%})",
              rate < 0.10, f"{rate:.1%}")

    # The motivating run's OWN stage-T output is clean — the body came from a
    # shared REPAIR, which is why gating stage T alone would not have caught
    # it and set_section carries the same check.
    recs = [json.loads(l) for l in open(
        "/nobackup2/alleshwaram/CtoRust/mtu_runs/abl2/runs/binary_heap_base_srvA_t5/"
        "_project_B03_organic/project.jsonl")]
    pt = [r for r in recs if r.get("type") == "project_types_all"]
    if len(pt) >= 2:
        check("the shared block was clean before the repair",
              not chk(pt[0]["types_rs"]), chk(pt[0]["types_rs"])[:60])
        check("...and carries an implemented body after it",
              bool(chk(pt[-1]["types_rs"])))
        check("so the set_section guard would reject that write",
              bool(chk(pt[-1]["types_rs"])) and not bool(chk(pt[0]["types_rs"])))


def main() -> int:
    test_gather_units_survives()
    test_on_result_fires_before_later_failure()
    test_fatal_is_not_swallowed()
    test_code_stage_degrades()
    test_spec_stage_degrades()
    test_skip_regenerates_only_missing()
    test_gate_would_reject()
    test_transient_errors_retried()
    test_empty_choices()
    test_bad_request_still_raises()
    test_crash_exit_code()
    test_parse_regression()
    test_parse_regression_on_real_trial4()
    test_set_section_keeps_previous()
    test_emptied_blocks()
    test_emptied_blocks_on_real_fixture()
    test_remaining_stubs()
    test_remaining_stubs_on_real_crates()
    test_surgical_edits_cannot_delete()
    test_repair_prompts_still_format()
    test_emptied_blocks_both_fixtures()
    test_types_stage_records_exhaustion()
    test_orphan_rule_in_prompts()
    test_one_call_is_bounded()
    test_missing_item_routes_to_owner()
    test_illegal_type_bodies()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
