"""plan.md site C: the stub-residue gate.

Run: cd diffusionMTUs && PYTHONPATH=. venv/bin/python test_site_c.py

The gate is all-or-nothing, and that is the property worth proving. A crate
with one stub left scores exactly as badly as one with six — `remaining_stubs`
voids on one — so a gate that fires when the residue is unclearable spends
money for a verdict that cannot change. `cc_array base` cleared 5 stubs and
stayed STUB_CRATE in all 6 runs; that is the failure this gate exists to avoid.

Known-good AND known-bad throughout: a gate that never fires and a gate that
always fires are indistinguishable from "no effect" downstream.
"""
import os

os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"
os.environ["VLLM_API_KEY"] = "test"

from rustgen.stub_repair import (SHARED_ID, StubContext, find_open_stubs,
                                 should_escalate_stubs, unanswerable_stubs)
from state import Explanation

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


UNITS = [Explanation(id="u1", ranges=[[1, 5]], text="does a thing",
                     invariants=[], status="locked")]


def ctx_for(sections, call_texts=None):
    crate = "\n".join(sections.values())
    return StubContext(crate, UNITS, c_sources={}, call_texts=call_texts or {})


def main():
    # ---- clean crate: nothing to do, and the caller must not announce a skip
    clean = {"u1": "pub fn run() -> i32 { 42 }\n"}
    fire, why = should_escalate_stubs(clean, ctx_for(clean), SHARED_ID)
    check(fire is False, "a clean crate does not fire")
    check("no patchable stubs" in why, "  and says so as the quiet case")

    # ---- one stub, and it IS called from the crate -> clearable -> fire
    called = {
        "u1": ("pub fn helper(x: i32) -> i32 { todo!() }\n"
               "pub fn run() -> i32 { helper(1) }\n"),
    }
    c = ctx_for(called)
    check(c.has_callers("helper") is True, "a crate call site is evidence")
    fire, why = should_escalate_stubs(called, c, SHARED_ID)
    check(fire is True, "a clearable residue fires")
    check(why == "", "  with no reason-not-to")

    # ---- one stub, called only from the C -> still evidence -> fire
    only_c = {"u1": "pub fn cb(x: i32) -> i32 { todo!() }\n"}
    c = ctx_for(only_c, call_texts={"cb": ["cc_array_reduce(a, cb)"]})
    check(c.has_callers("cb") is True, "a C call site is evidence too")
    fire, _ = should_escalate_stubs(only_c, c, SHARED_ID)
    check(fire is True, "evidence from the C alone is enough to fire")

    # ---- one stub, called from NOWHERE -> unanswerable -> do not spend
    orphan = {"u1": "pub fn never_used(x: i32) -> i32 { todo!() }\n"}
    c = ctx_for(orphan)
    check(c.has_callers("never_used") is False,
          "no call site anywhere -> no evidence")
    stubs = [s for s in find_open_stubs(orphan, SHARED_ID) if s.patchable]
    check(len(unanswerable_stubs(stubs, c)) == 1, "the orphan is unanswerable")
    fire, why = should_escalate_stubs(orphan, c, SHARED_ID)
    check(fire is False, "an unanswerable residue does NOT fire")
    check("never_used" in why, "  and names the stub responsible")

    # ---- THE ONE THAT MATTERS: four clearable + one orphan -> spend NOTHING.
    # Clearing four of five leaves the crate STUB_CRATE, so the verdict is
    # unchanged and the money is gone.
    mixed = {
        "u1": ("pub fn a() -> i32 { todo!() }\n"
               "pub fn b() -> i32 { todo!() }\n"
               "pub fn c_() -> i32 { todo!() }\n"
               "pub fn d() -> i32 { todo!() }\n"
               "pub fn orphan() -> i32 { todo!() }\n"
               "pub fn run() -> i32 { a() + b() + c_() + d() }\n"),
    }
    c = ctx_for(mixed)
    stubs = [s for s in find_open_stubs(mixed, SHARED_ID) if s.patchable]
    check(len(stubs) == 5, f"five patchable stubs found (got {len(stubs)})")
    dead = unanswerable_stubs(stubs, c)
    check([s.function for s in dead] == ["orphan"],
          "exactly the orphan is unanswerable")
    fire, why = should_escalate_stubs(mixed, c, SHARED_ID)
    check(fire is False,
          "ONE unanswerable stub blocks the whole set — partial clearance "
          "does not un-void a crate")
    check("1 of 5" in why, "  and the reason quantifies it")

    # ---- shared types are not patchable and must not drag the gate either way
    shared_only = {SHARED_ID: "pub struct S;\nimpl S { fn f(&self) { todo!() } }\n"}
    fire, why = should_escalate_stubs(shared_only, ctx_for(shared_only),
                                      SHARED_ID)
    check(fire is False, "a stub only in shared types does not fire")
    check("no patchable stubs" in why, "  because it is not patchable at all")

    # ---- blind must NOT change the gate: it starves the MODEL of context,
    # it does not make the pipeline forget what it knows about spending money
    blind = StubContext("\n".join(called.values()), UNITS, blind=True)
    check(blind.has_callers("helper") is True,
          "blind does not blind the SPEND decision")
    check(should_escalate_stubs(called, blind, SHARED_ID)[0] is True,
          "  so the blind arm makes the same escalation decision")
    check("no information available" in blind.answer(
        {"q": "callers_of", "arg": "helper"}),
        "  while the model is still answered blind")

    print(f"\n{PASS} passed, {FAIL} failed")
    print("ALL PASS" if FAIL == 0 else "FAILURES")
    return 1 if FAIL else 0


raise SystemExit(main())
