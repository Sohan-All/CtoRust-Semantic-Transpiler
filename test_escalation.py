"""Escalation metering: the record shape, deltas, and the enable logic.

Run: cd diffusionMTUs && PYTHONPATH=. venv/bin/python test_escalation.py

Tests the SUCCESS path as well as the failure path — the branch nobody
exercises is where this bites. In particular a rejected firing and a raising
firing must both still produce a record, because a firing that vanishes is
indistinguishable from one that never happened, except that it was paid for.
"""
import dataclasses
import os

os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"   # never contacted
os.environ["VLLM_API_KEY"] = "test"

from config import Config
from llm import LLM
from rustgen.escalation import (Escalation, SITES, enabled_sites,
                                should_escalate_compile)

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


class FakeLLM:
    """Only the counters Escalation reads. Constructed rather than mocked so a
    field rename in LLM shows up here as a failure instead of a silent zero."""

    def __init__(self, model="claude-opus-5"):
        self.cfg = dataclasses.replace(Config(), worker_model=model)
        self.calls = 10
        self.input_tokens = 1000
        self.output_tokens = 2000

    def spend(self, calls=1, tin=100, tout=300):
        self.calls += calls
        self.input_tokens += tin
        self.output_tokens += tout


def main():
    # ---- accepted firing: cost is the DELTA, not the running total
    llm = FakeLLM()
    with Escalation("compile", llm, trigger="final", value=2) as e:
        llm.spend(calls=2, tin=500, tout=900)
        e.done(accepted=True)
    r = e.record
    check(r["type"] == "escalation", "record type")
    check(r["site"] == "compile" and r["trigger"] == "final"
          and r["trigger_value"] == 2, "site and trigger recorded")
    check(r["calls"] == 2, "calls is a delta, not the instance total")
    check(r["input_tokens"] == 500 and r["output_tokens"] == 900,
          "tokens are deltas, not instance totals")
    check(r["accepted"] is True and r["rejected_by"] == "",
          "accepted firing records no rejection")
    check(r["model"] == "claude-opus-5", "model comes off the llm's cfg")
    check(r["error"] == "", "no error on the happy path")
    check(isinstance(r["seconds"], float), "wall time recorded")

    # ---- rejected firing is recorded just as fully. This is the point.
    llm = FakeLLM()
    with Escalation("stubs", llm, trigger="stubs_remaining", value=4) as e:
        llm.spend(tout=750)
        e.done(accepted=False, rejected_by="lost_impl_methods")
    r = e.record
    check(r["accepted"] is False, "rejected firing records accepted=False")
    check(r["rejected_by"] == "lost_impl_methods", "the gate's complaint is kept")
    check(r["output_tokens"] == 750,
          "a REJECTED firing still records what it cost")

    # ---- a raising firing: record survives, exception propagates
    llm = FakeLLM()
    try:
        with Escalation("types", llm, trigger="gate_exhausted", value=True) as e:
            llm.spend(tout=120)
            raise RuntimeError("vertex went away")
        check(False, "exception must propagate")
    except RuntimeError:
        check(True, "exception propagates (site decides its own recovery)")
    r = e.record
    check(r["accepted"] is False, "a raising firing is not accepted")
    check("RuntimeError" in r["error"] and "vertex" in r["error"],
          "the error is recorded")
    check(r["output_tokens"] == 120, "a crashed firing still records its cost")

    # ---- an unfinished firing defaults to not-accepted rather than silently ok
    llm = FakeLLM()
    with Escalation("types", llm, trigger="gate_exhausted", value=True) as e:
        llm.spend()
    check(e.record["accepted"] is False,
          "a firing that never called done() is not counted as accepted")

    # ---- unknown site fails loudly rather than creating a 4th category
    try:
        Escalation("typos", FakeLLM(), trigger="x", value=1)
        check(False, "unknown site must raise")
    except ValueError:
        check(True, "unknown site raises")

    # ---- enabled_sites: the master switch and the per-site knobs
    off = Config()
    check(off.escalation_model == "", "escalation ships disabled")
    check(enabled_sites(off) == [], "no model -> no sites, whatever the knobs")

    on = dataclasses.replace(off, escalation_model="claude-opus-5")
    check(enabled_sites(on) == list(SITES), "a model enables all three sites")

    partial = dataclasses.replace(on, escalate_compile=False)
    check("compile" not in enabled_sites(partial), "a per-site knob switches off")
    check(len(enabled_sites(partial)) == 2, "the others stay on")

    # off-with-knobs-on must still be off: the master switch dominates
    check(enabled_sites(dataclasses.replace(off, escalate_types=True)) == [],
          "per-site knobs cannot enable escalation without a model")

    # ---- site B's gate. Known-good AND known-bad: a gate that never fires and
    # a gate that always fires both look like "no effect" from the outside.
    sites = list(SITES)
    on3 = dataclasses.replace(on, escalation_max_errors=3)
    for n in (1, 2, 3):
        check(should_escalate_compile(on3, sites, n) is True,
              f"final: {n} is a repair -> escalate")
    for n in (4, 10, 19, 34):
        check(should_escalate_compile(on3, sites, n) is False,
              f"final: {n} is a rewrite -> do not escalate")
    check(should_escalate_compile(on3, sites, 0) is False,
          "final: 0 built — nothing to escalate")
    check(should_escalate_compile(on3, sites, -1) is False,
          "final: -1 (never recorded) is not a firing")
    check(should_escalate_compile(on3, ["types", "stubs"], 1) is False,
          "the per-site knob gates it even at 1 error")
    check(should_escalate_compile(on3, [], 1) is False,
          "escalation off -> never")
    # the threshold is a config value, not a constant baked into the gate
    check(should_escalate_compile(
        dataclasses.replace(on, escalation_max_errors=1), sites, 2) is False,
        "a tighter escalation_max_errors is respected")
    check(should_escalate_compile(
        dataclasses.replace(on, escalation_max_errors=10), sites, 8) is True,
        "a looser escalation_max_errors is respected")

    # ---- site B's OUTCOME rule. Found live: the first multi-site run went
    # 3 -> 1 and was still BUILD_FAILED, so "fewer errors" is not "a result".
    from rustgen.escalation import compile_outcome
    check(compile_outcome(3, 0) == (True, True),
          "3 -> 0 is both an improvement and a resolution")
    check(compile_outcome(3, 1) == (True, False),
          "3 -> 1 is kept but is NOT accepted — both score BUILD_FAILED")
    check(compile_outcome(1, 0) == (True, True), "1 -> 0 resolves")
    check(compile_outcome(3, 3) == (False, False), "no change is not improvement")
    check(compile_outcome(3, 5) == (False, False), "worse is not improvement")
    check(compile_outcome(3, -1) == (False, False),
          "an unrecorded result is neither")

    # ---- the per-RUN token budget. A field that reads like a bound but does
    # nothing is worse than no field: it creates a false belief that spend is
    # capped. This was declared for a whole session before being enforced.
    from rustgen.escalation import over_budget, spent_tokens
    recs = [{"input_tokens": 1000, "output_tokens": 500},
            {"input_tokens": 2000, "output_tokens": 500}]
    check(spent_tokens(recs) == 4000, "spend counts input AND output")
    check(spent_tokens([]) == 0, "no firings, no spend")
    small = dataclasses.replace(on, escalation_run_token_budget=4000)
    check(over_budget(recs, small) is True, "at the budget -> stop")
    check(over_budget(recs[:1], small) is False, "under the budget -> continue")
    check(over_budget([], small) is False, "a fresh run is never over budget")
    big = dataclasses.replace(on, escalation_run_token_budget=100000)
    check(over_budget(recs, big) is False, "a large budget does not trip")
    check(over_budget(recs, dataclasses.replace(
        on, escalation_run_token_budget=0)) is False,
        "budget 0 disables the cap rather than blocking everything")
    check(over_budget(recs, dataclasses.replace(
        on, escalation_run_token_budget=-1)) is False,
        "a negative budget also disables rather than inverting")
    check(Config().escalation_run_token_budget > 0,
          "the shipped default IS a real cap, not 0")

    # ---- the escalation config reaches the run record, or an arm is unauditable
    rec = dataclasses.replace(on, escalate_stubs=False).to_record()
    body = rec.get("config", rec)
    for k in ("escalation_model", "escalate_types", "escalate_compile",
              "escalate_stubs", "escalation_max_errors"):
        check(k in body, f"{k} is in the config record")
    check(body["escalation_model"] == "claude-opus-5"
          and body["escalate_stubs"] is False,
          "the config record carries the ACTUAL values")

    # ---- usage_record carries model and role (what makes a run auditable)
    real = LLM(dataclasses.replace(Config(), worker_model="gemma-4-31b"),
               role="rustgen")
    u = real.usage_record()
    check(u["role"] == "rustgen", "role reaches the usage record")
    check(u["model"] == "gemma-4-31b", "model reaches the usage record")
    check(LLM(Config()).usage_record()["role"] == "worker",
          "role defaults to worker")

    print(f"\n{PASS} passed, {FAIL} failed")
    print("ALL PASS" if FAIL == 0 else "FAILURES")
    return 1 if FAIL else 0


raise SystemExit(main())
