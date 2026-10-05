"""Transaction-scoped gates. Run: PYTHONPATH=. venv/bin/python test_transaction.py

The assertion that matters is `test_cross_section_move_*`: a method moving
between sections must be ACCEPTED under transaction scope and REFUSED under the
per-section scope it replaces. Both directions, because a gate that accepts
everything looks exactly like a gate that works — and this one is a relaxation,
which is the direction that needs the known-bad half most.
"""
from rustgen.transaction import (TxResult, behaviour_lost, check_transaction,
                                 validate_edits)
from rustgen.common import lost_impl_methods

PASS = 0


def ok(cond, label):
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1


# ---------------------------------------------------------------- fixtures

PROJ_A = """\
impl Project {
    pub fn new(id: i32) -> Self {
        Project { id, tasks: Vec::new() }
    }
    pub fn total(&self) -> usize {
        self.tasks.len()
    }
}
"""

# `new` has moved out; `total` stays.
PROJ_A_MOVED = """\
impl Project {
    pub fn total(&self) -> usize {
        self.tasks.len()
    }
}
"""

# The section that receives it, and which also calls it.
CALLER_B = """\
pub(crate) fn build() -> Project {
    Project::new(100)
}
"""

CALLER_B_WITH_NEW = """\
impl Project {
    pub fn new(id: i32) -> Self {
        Project { id, tasks: Vec::new() }
    }
}

pub(crate) fn build() -> Project {
    Project::new(100)
}
"""


def test_accepts_a_clean_single_section_edit():
    before = {"a": PROJ_A, "b": CALLER_B}
    new_a = PROJ_A.replace("self.tasks.len()", "self.tasks.len() + 0")
    r = check_transaction(before, {"a": new_a})
    ok(r.accepted, "a clean edit is accepted")
    ok("len() + 0" in r.sections["a"] and "len() + 0" not in before["a"],
       "an accepted edit is the state returned, and it differs from the input")
    ok(r.sections["b"] == CALLER_B, "an untouched section is unchanged")
    ok("parse_regression" in r.checks_run, "the structural checks ran")
    ok(r.record["accepted"] is True and r.record["touched"] == ["a"],
       "an accepted transaction records what it touched")


def test_cross_section_move_is_refused_by_the_old_per_section_scope():
    """Known-bad for the relaxation: the scope being replaced refuses this."""
    # `elsewhere` drawn from the BEFORE state, which is what a per-section
    # write point has available when it evaluates section `a`.
    problem = lost_impl_methods(PROJ_A, PROJ_A_MOVED, CALLER_B)
    ok(problem != "", "per-section scope refuses a legitimate move")
    ok("new" in problem, "and it names the method it thinks was lost")


def test_cross_section_move_is_accepted_by_transaction_scope():
    """Known-good: the same edit, evaluated as one transaction."""
    before = {"a": PROJ_A, "b": CALLER_B}
    r = check_transaction(before, {"a": PROJ_A_MOVED, "b": CALLER_B_WITH_NEW})
    ok(r.accepted, "transaction scope accepts a move whose destination is in "
                   "the same transaction")
    ok(set(r.touched) == {"a", "b"}, "both sections are recorded as touched")
    ok("fn new" in r.sections["b"], "the destination text landed")


def test_a_real_deletion_is_still_refused():
    """The relaxation must not swallow the case the rule exists for: `new` is
    dropped and NOTHING in the after-state defines it, while a caller remains."""
    before = {"a": PROJ_A, "b": CALLER_B}
    r = check_transaction(before, {"a": PROJ_A_MOVED})
    ok(not r.accepted, "deleting a called method is still refused")
    ok(any(p["check"] == "lost_impl_methods" for p in r.problems),
       "and refused by the check that owns that question")
    ok(r.sections == before, "a refused transaction leaves the caller where it started")


def test_all_or_nothing():
    before = {"a": PROJ_A, "b": CALLER_B, "c": "pub fn c() {}\n"}
    good = "pub fn c() { let _ = 1; }\n"
    broken = "impl Project { pub fn total(&self) -> usize { self.tasks.len()\n"
    r = check_transaction(before, {"c": good, "a": broken})
    ok(not r.accepted, "one bad section rejects the whole transaction")
    ok(r.sections["c"] == "pub fn c() {}\n",
       "the GOOD section in a rejected transaction is not applied either")


def test_parse_regression_is_asymmetric():
    broken = "pub fn x() { \n"
    before = {"a": broken}
    still_broken = "pub fn x() { let _ = 1;\n"
    r = check_transaction(before, {"a": still_broken})
    ok(r.accepted, "an already-unbalanced section stays writable")

    before2 = {"a": "pub fn x() {}\n"}
    r2 = check_transaction(before2, {"a": broken})
    ok(not r2.accepted, "a balanced section may not be made unbalanced")
    ok(any(p["check"] == "parse_regression" for p in r2.problems),
       "and the parse check is the one that says so")


def test_emptied_blocks_is_not_relaxed():
    """Deliberately NOT transaction-scoped — see the module docstring and the
    reverted exemption at common.py:220."""
    gutted = """\
impl Project {
    /// Creates a project.
    /// Totals the tasks.
}
"""
    before = {"a": PROJ_A, "b": CALLER_B_WITH_NEW}
    r = check_transaction(before, {"a": gutted, "b": CALLER_B_WITH_NEW})
    ok(not r.accepted,
       "gutting an impl block is refused even when the names survive elsewhere")
    ok(any(p["check"] == "emptied_blocks" for p in r.problems),
       "and refused by emptied_blocks specifically")


def test_validate_edits():
    before = {"a": "pub fn a() {}\n"}
    clean, problems = validate_edits(before, {"zz": "pub fn z() {}\n"})
    ok(not clean and problems, "an edit to an unknown section is refused")
    ok(problems[0]["check"] == "unknown_section", "and named as such")

    clean, problems = validate_edits(before, {"a": "   \n  "})
    ok(not clean and problems, "an empty edit is refused")
    ok(problems[0]["check"] == "empty_edit", "and named as such")

    clean, problems = validate_edits(before, {"a": "pub fn a() { }\n"})
    ok(clean and not problems, "a well-formed edit passes validation")


def test_empty_transaction():
    r = check_transaction({"a": "pub fn a() {}\n"}, {})
    ok(not r.accepted, "a transaction proposing nothing is not accepted")
    ok(r.problems, "and it records why rather than failing silently")


def test_compile_check_gates_and_a_broken_cargo_does_not():
    before = {"a": PROJ_A, "b": CALLER_B}
    good = PROJ_A.replace("self.tasks.len()", "self.tasks.len() + 0")

    r = check_transaction(before, {"a": good},
                          compile_check=lambda s: "error[E0308]: mismatched types")
    ok(not r.accepted, "a transaction that fails cargo is refused")
    ok(r.compile_problem, "and the compiler output is recorded, not just a flag")
    ok(r.sections == before, "and nothing is applied")

    def raiser(_):
        raise FileNotFoundError("cargo")
    r2 = check_transaction(before, {"a": good}, compile_check=raiser)
    ok(r2.accepted,
       "an unusable cargo is not a verdict on the edit — it must not reject")

    r3 = check_transaction(before, {"a": good}, compile_check=lambda s: "")
    ok(r3.accepted, "a clean cargo accepts (the SUCCESS path, not just failure)")
    ok("compile_check" in r3.checks_run, "and the check is recorded as having run")


def test_compile_check_sees_the_whole_after_state():
    """One check per transaction, over the merged state — not per section."""
    seen = {}
    before = {"a": PROJ_A, "b": CALLER_B}

    def spy(sections):
        seen.update(sections)
        seen["__calls__"] = seen.get("__calls__", 0) + 1
        return ""
    check_transaction(before, {"a": PROJ_A_MOVED, "b": CALLER_B_WITH_NEW},
                      compile_check=spy)
    ok(seen["__calls__"] == 1, "cargo runs exactly once per transaction")
    ok("fn new" in seen["b"] and "fn new" not in seen["a"],
       "and it sees the merged after-state of every edited section")


def test_rejected_transaction_records_as_fully_as_an_accepted_one():
    before = {"a": PROJ_A, "b": CALLER_B}
    rej = check_transaction(before, {"a": PROJ_A_MOVED})
    acc = check_transaction(before, {"a": PROJ_A_MOVED, "b": CALLER_B_WITH_NEW})
    ok(set(rej.record) == set(acc.record),
       "a refused transaction carries the same record keys as an accepted one")
    ok(rej.record["problems"] and not acc.record["problems"],
       "and the problems list is what distinguishes them")
    ok(rej.record["touched"] and rej.record["checks_run"],
       "a refusal still records what it touched and what it consulted")
    ok(rej.summary().startswith("[tx] REJECTED")
       and acc.summary().startswith("[tx] ACCEPTED"), "both summarise")


def test_behaviour_lost_is_a_whole_crate_view():
    before = {"a": PROJ_A, "b": CALLER_B}
    after_moved = {"a": PROJ_A_MOVED, "b": CALLER_B_WITH_NEW}
    ok(behaviour_lost(before, after_moved) == [],
       "a move loses no behaviour crate-wide")
    after_deleted = {"a": PROJ_A_MOVED, "b": CALLER_B}
    ok(behaviour_lost(before, after_deleted) == ["new"],
       "a deletion is visible crate-wide even when no per-section gate fired")


def test_shared_types_asymmetry():
    shared_clean = "pub struct S { pub x: i32 }\n"
    shared_bodied = """\
pub struct S { pub x: i32 }
impl S {
    pub fn compute(&self) -> i32 { self.x * 2 }
}
"""
    r = check_transaction({"__shared__": shared_clean, "a": "pub fn a() {}\n"},
                          {"__shared__": shared_bodied}, shared_id="__shared__")
    ok(not r.accepted, "a transaction may not implement behaviour in the types block")

    r2 = check_transaction({"__shared__": shared_bodied, "a": "pub fn a() {}\n"},
                           {"__shared__": shared_bodied.replace("* 2", "* 3")},
                           shared_id="__shared__")
    ok(r2.accepted, "a types block that ALREADY had bodies stays writable")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"ALL PASS ({PASS} assertions)")
