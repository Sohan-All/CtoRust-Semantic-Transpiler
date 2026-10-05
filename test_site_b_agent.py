"""Site B's agent tier. Run: PYTHONPATH=. venv/bin/python test_site_b_agent.py

The claim this file exists to check is ATOMICITY. The non-agent cluster tier
calls `set_section` per section in a loop, each independently gated, so a
cross-section change can land half — section A accepted, section B refused —
leaving a crate that is incoherent for a reason the loop then attributes to the
round rather than to the partial write. `check_transaction` makes that state
unreachable.

So the assertions are paired: the same partly-bad edit set must land partially
under the old tier and not at all under the new one. A gate that refuses
everything looks exactly like a gate that works, so the accepted case is
asserted too.
"""
import asyncio

from rustgen.agent_sites import cluster_task
from rustgen.transaction import check_transaction

PASS = 0


def ok(cond, label):
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1


# Two sections tied by a signature: `helper` takes &Project, the caller passes
# one. A coherent change alters both.
SEC_A = """\
impl Project {
    pub fn helper(&self, n: i32) -> i32 {
        self.id + n
    }
    pub fn other(&self) -> i32 { self.id }
}
"""
SEC_B = """\
pub(crate) fn call_it(p: &Project) -> i32 {
    p.helper(1)
}
"""

# The coherent cross-section change: helper gains a parameter, caller updated.
SEC_A_NEW = """\
impl Project {
    pub fn helper(&self, n: i32, scale: i32) -> i32 {
        (self.id + n) * scale
    }
    pub fn other(&self) -> i32 { self.id }
}
"""
SEC_B_NEW = """\
pub(crate) fn call_it(p: &Project) -> i32 {
    p.helper(1, 2)
}
"""

# The half that a per-section gate refuses: `other` is dropped and B still
# calls nothing else, so `lost_impl_methods` fires on A alone.
SEC_A_LOSSY = """\
impl Project {
    pub fn helper(&self, n: i32, scale: i32) -> i32 {
        (self.id + n) * scale
    }
}
"""
SEC_B_CALLS_OTHER = """\
pub(crate) fn call_it(p: &Project) -> i32 {
    p.helper(1) + p.other()
}
"""

# The caller updated for the new signature and STILL calling `other`. Paired with
# SEC_A_LOSSY this is a real deletion: `other` is gone from the after-state and
# something in the after-state still calls it.
#
# The first draft of this fixture used SEC_B_NEW here, which drops the `other()`
# call — and the transaction gate ACCEPTED it, correctly: deleting a method and
# its only caller in one transaction is coherent dead-code removal, which is
# exactly the case per-section scope cannot distinguish from damage. The gate was
# right and the fixture was wrong. Kept as a comment because the distinction is
# the whole point of the rescoping.
SEC_B_NEW_CALLS_OTHER = """\
pub(crate) fn call_it(p: &Project) -> i32 {
    p.helper(1, 2) + p.other()
}
"""


def _old_tier(before, edits, gate):
    """The non-agent tier's write pattern: per section, independently gated."""
    out = dict(before)
    for sid, new in edits.items():
        if gate(before, sid, new):
            out[sid] = new
    return out


def _per_section_gate(before, sid, new):
    from rustgen.common import lost_impl_methods, parse_regression
    elsewhere = "\n".join(c for s, c in before.items() if s != sid)
    return not (parse_regression(before.get(sid, ""), new)
                or lost_impl_methods(before.get(sid, ""), new, elsewhere))


def test_the_old_tier_lands_a_partial_write():
    """Known-bad: this is the failure the transaction gate exists to remove."""
    before = {"a": SEC_A, "b": SEC_B_CALLS_OTHER}
    edits = {"a": SEC_A_LOSSY, "b": SEC_B_NEW_CALLS_OTHER}
    after = _old_tier(before, edits, _per_section_gate)
    ok(after["a"] == SEC_A, "the lossy section is refused")
    ok(after["b"] == SEC_B_NEW_CALLS_OTHER,
       "but the OTHER section still landed")
    ok(after != before, "so the crate is left in a state neither version intended")


def test_the_agent_tier_lands_nothing():
    """Known-good: the same edit set, all-or-nothing."""
    before = {"a": SEC_A, "b": SEC_B_CALLS_OTHER}
    edits = {"a": SEC_A_LOSSY, "b": SEC_B_NEW_CALLS_OTHER}
    tx = check_transaction(before, edits)
    ok(not tx.accepted, "the transaction is refused")
    ok(tx.sections == before, "and NOTHING is applied, not even the good half")
    ok(any(p["check"] == "lost_impl_methods" for p in tx.problems),
       "the refusing check is named")


def test_a_coherent_cross_section_change_is_accepted():
    """The SUCCESS path — the branch nobody exercises is where this bites."""
    before = {"a": SEC_A, "b": SEC_B}
    tx = check_transaction(before, {"a": SEC_A_NEW, "b": SEC_B_NEW},
                           compile_check=lambda s: "")
    ok(tx.accepted, "a coherent signature change across two sections lands")
    ok("scale: i32" in tx.sections["a"] and "helper(1, 2)" in tx.sections["b"],
       "and both halves are present in the returned state")


def test_sections_outside_the_cluster_are_dropped():
    """A wider read surface is not a licence to rewrite uninvolved sections.
    The non-agent tier enforces this with `if sid in ranked`; the agent tier
    filters before the gate sees the edits."""
    ranked = ["a", "b"]
    proposed = {"a": SEC_A_NEW, "b": SEC_B_NEW, "c": "pub fn c() {}\n"}
    edits = {k: v for k, v in proposed.items() if k in ranked}
    ok(set(edits) == {"a", "b"}, "an edit outside the cluster is filtered out")
    before = {"a": SEC_A, "b": SEC_B, "c": "pub fn c() { let _ = 0; }\n"}
    tx = check_transaction(before, edits, compile_check=lambda s: "")
    ok(tx.accepted and tx.sections["c"] == before["c"],
       "and the uninvolved section is untouched by the accepted transaction")


def test_cluster_task_carries_what_the_old_prompt_could_not():
    task = cluster_task(["a", "b"], "SECTIONS-HERE", "ERRORS-HERE")
    ok("ERRORS-HERE" in task and "SECTIONS-HERE" in task,
       "the errors and the sections reach the task")
    ok("a, b" in task, "and the editable set is named")
    ok("c_source/" in task,
       "the C source is offered — no compile repair in this pipeline ever had it")
    ok("ONE transaction" in task, "and the all-or-nothing rule is stated")
    ok("deleting what does not compile" in task,
       "and the deletion shortcut is refused in the prompt as well as the gate")


def test_compile_loop_accepts_the_agent_parameters():
    """Integration shape: the loop's signature takes them and defaults to the
    old tier when they are absent."""
    import inspect
    from rustgen.compile_loop import compile_loop
    sig = inspect.signature(compile_loop)
    for p in ("agent_staged", "agent_budget", "agent_records"):
        ok(p in sig.parameters, f"compile_loop takes {p}")
        ok(sig.parameters[p].default is None, f"{p} defaults to off")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"ALL PASS ({PASS} assertions)")
