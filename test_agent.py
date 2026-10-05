"""Read agent + containment. Run: PYTHONPATH=. venv/bin/python test_agent.py

The containment assertions are the ones that matter. The DARPA vectors are the
metric and never an input; an earlier repair loop in this project was scored on
vectors it had consumed. `plan.md`'s Containment section warns that an agent with
filesystem search WILL find them, because for the task it has been set they are
genuinely the most useful file in the tree — so the guarantee has to be that they
are absent from the reachable root, not that the prompt asked nicely.

Asserted against known-good and known-bad both: a root that reaches nothing looks
identical to one that is correctly scoped.

Fake LLMs here are keyed on call ORDER, never on prompt text — a retry prompt
quotes the problem back, so a text-keyed fake lets the stage fix itself and the
test measures nothing.
"""
import asyncio
import shutil
import tempfile
from pathlib import Path

from rustgen.agent import (AgentBudget, StagedRoot, StagedRootError,
                           assert_contained, parse_transaction, run_agent,
                           stage, tool_grep, tool_list, tool_read,
                           _first_json_object)

PASS = 0


def ok(cond, label):
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1


class OrderedLLM:
    """Replies in sequence. Keyed on call order, per the convention."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ask(self, prompt, max_tokens=None, **kw):
        self.prompts.append(prompt)
        if not self.replies:
            return '{"action": "give_up", "why": "out of scripted replies"}'
        return self.replies.pop(0)


class RaisingLLM:
    async def ask(self, prompt, max_tokens=None, **kw):
        raise RuntimeError("transport died")


def _corpus_shaped_tree(base: Path) -> tuple[Path, Path]:
    """A `test_case/` with `test_vectors/` as its SIBLING — the real corpus
    layout, which is what makes staging `c_root` wholesale a live hazard."""
    proj = base / "binary_heap"
    (proj / "test_case/src").mkdir(parents=True)
    (proj / "test_case/src/main.c").write_text("int main(void){return 0;}\n")
    (proj / "test_vectors").mkdir(parents=True)
    (proj / "test_vectors/case_001.txt").write_text("EXPECTED OUTPUT\n")
    return proj / "test_case", proj / "test_vectors"


# --------------------------------------------------------------- containment

def test_staging_excludes_the_vectors():
    tmp = Path(tempfile.mkdtemp())
    try:
        test_case, vectors = _corpus_shaped_tree(tmp)
        crate = tmp / "crate/src"
        crate.mkdir(parents=True)
        (crate / "lib.rs").write_text("pub fn f() {}\n")

        sr = stage(tmp / "staged", crate_src=tmp / "crate",
                   c_src=test_case / "src")
        staged_files = {p.name for p in sr.root.rglob("*") if p.is_file()}
        ok("main.c" in staged_files, "the C source IS staged (known-good)")
        ok("lib.rs" in staged_files, "the crate IS staged (known-good)")
        ok("case_001.txt" not in staged_files,
           "the vectors are NOT staged (known-bad)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_staging_drops_venv_and_build_dirs():
    """Not a containment rule — a cost and signal one, and it was invisible to
    every unit test here until a real project was staged and counted: the naive
    copy took 16,634 files, 15,619 of them `venv/`."""
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "pipeline"
        (src / "rustgen").mkdir(parents=True)
        (src / "rustgen/common.py").write_text("_DEPS_MOD = 1\n")
        (src / "venv/lib/site-packages/numpy").mkdir(parents=True)
        (src / "venv/lib/site-packages/numpy/core.py").write_text("x = 1\n")
        (src / "out").mkdir()
        (src / "out/state.jsonl").write_text("{}\n")
        (src / "__pycache__").mkdir()
        (src / "__pycache__/x.pyc").write_text("junk\n")
        (src / "notes.bin").write_bytes(b"\x00\x01")

        sr = stage(tmp / "staged", pipeline_src=src)
        rel = {str(p.relative_to(sr.root)) for p in sr.root.rglob("*") if p.is_file()}
        ok("pipeline/rustgen/common.py" in rel, "pipeline source IS staged")
        ok(not any("venv" in r for r in rel), "venv is not staged")
        ok(not any(r.startswith("pipeline/out") for r in rel), "out/ is not staged")
        ok(not any("__pycache__" in r for r in rel), "caches are not staged")
        ok("pipeline/notes.bin" not in rel,
           "non-source files are not staged from the pipeline tree")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_assert_contained_catches_a_vectors_directory():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "root/src").mkdir(parents=True)
        (tmp / "root/src/lib.rs").write_text("pub fn f() {}\n")
        assert_contained(tmp / "root")          # known-good: does not raise
        ok(True, "a clean root passes containment")

        (tmp / "root/test_vectors").mkdir()
        (tmp / "root/test_vectors/v.txt").write_text("x\n")
        raised = False
        try:
            assert_contained(tmp / "root")
        except StagedRootError:
            raised = True
        ok(raised, "a root containing test_vectors is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_assert_contained_catches_an_escaping_symlink():
    """A component check cannot see this, which is why the walk resolves."""
    tmp = Path(tempfile.mkdtemp())
    try:
        _, vectors = _corpus_shaped_tree(tmp)
        (tmp / "root").mkdir()
        (tmp / "root/lib.rs").write_text("pub fn f() {}\n")
        (tmp / "root/sneaky").symlink_to(vectors)
        raised = False
        try:
            assert_contained(tmp / "root")
        except StagedRootError:
            raised = True
        ok(raised, "a symlink escaping the staged root is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_refuses_traversal_and_absolute_paths():
    tmp = Path(tempfile.mkdtemp())
    try:
        _, vectors = _corpus_shaped_tree(tmp)
        (tmp / "root/src").mkdir(parents=True)
        (tmp / "root/src/lib.rs").write_text("pub fn f() {}\n")
        sr = StagedRoot(root=(tmp / "root").resolve())

        ok(sr.resolve("src/lib.rs").is_file(),
           "an in-bounds path resolves (known-good)")

        for bad in ("../binary_heap/test_vectors/case_001.txt",
                    "../../etc/passwd",
                    "src/../../binary_heap/test_vectors"):
            raised = False
            try:
                sr.resolve(bad)
            except StagedRootError:
                raised = True
            ok(raised, f"traversal refused: {bad}")

        # An absolute path is stripped to a relative one rather than honoured,
        # so it lands INSIDE the root. When what remains still names a
        # forbidden component it is refused outright rather than confined —
        # both halves matter, so both are asserted.
        raised = False
        try:
            sr.resolve(str(vectors))
        except StagedRootError:
            raised = True
        ok(raised, "an absolute path at the vectors is refused, not followed")

        p = sr.resolve("/etc/passwd")
        ok(p.is_relative_to(sr.root),
           "a benign absolute path is confined under the root, not followed out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tools_refuse_out_of_bounds_legibly():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "root").mkdir()
        (tmp / "root/lib.rs").write_text("pub fn alpha() {}\n")
        sr = StagedRoot(root=(tmp / "root").resolve())
        from rustgen.agent import _dispatch
        answer = _dispatch(sr, {"tool": "read", "path": "../../etc/passwd"})
        ok("refused" in answer.lower(),
           "an out-of-bounds read is answered with a refusal, not a traceback")
        ok(_dispatch(sr, {"tool": "nope"}).startswith("("),
           "an unknown tool is answered, not raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------- tools

def test_read_grep_list():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "root/src").mkdir(parents=True)
        (tmp / "root/src/lib.rs").write_text(
            "pub fn alpha() {}\nimpl Display for Scheduler {\n    fn fmt() {}\n}\n")
        sr = StagedRoot(root=(tmp / "root").resolve())

        ok("src/lib.rs" in tool_list(sr, ""), "list finds a nested file")
        ok("alpha" in tool_read(sr, "src/lib.rs"), "read returns content")
        ok("lines 2-3" in tool_read(sr, "src/lib.rs", 2, 3),
           "read honours a line range")
        ok("no such path" in tool_list(sr, "nope"), "list of a missing path is answered")
        ok("not a file" in tool_read(sr, "src"), "read of a directory is answered")

        hits = tool_grep(sr, r"impl\s+Display", "")
        ok("lib.rs:2" in hits,
           "grep finds a trait impl by shape — the question the fixed menu could "
           "not express")
        ok("no matches" in tool_grep(sr, "zzz_nothing", ""),
           "grep reports an honest miss")
        ok("bad regex" in tool_grep(sr, "([", ""), "a bad regex is answered, not raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_is_byte_bounded():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "root").mkdir()
        (tmp / "root/big.c").write_text("x" * 200_000 + "\n")
        sr = StagedRoot(root=(tmp / "root").resolve())
        out = tool_read(sr, "big.c")
        ok(len(out) < 60_000,
           "one unbounded read cannot eat the run's token budget")
        ok("truncated" in out, "and the truncation is stated rather than silent")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------- parsing

def test_parse_transaction():
    reply = '''{"action": "propose", "why": "wire it"}

```rust section=cli__exp_0002
pub(crate) fn a() { println!("{}", s); }
```

```rust section=demo__exp_0001
pub(crate) fn b() {}
```
'''
    edits = parse_transaction(reply)
    ok(set(edits) == {"cli__exp_0002", "demo__exp_0001"},
       "both fenced sections are parsed")
    ok('println!("{}", s);' in edits["cli__exp_0002"],
       "and string literals survive verbatim — the reason code never rides "
       "inside a JSON string")
    ok(parse_transaction("no fences here") == {}, "a reply with no fence yields none")


def test_last_fence_wins_per_section():
    reply = ('```rust section=a\nfn draft() {}\n```\n'
             '```rust section=a\nfn revised() {}\n```\n')
    edits = parse_transaction(reply)
    ok("revised" in edits["a"] and "draft" not in edits["a"],
       "a redrafted section takes the last fence, matching extract_rust")


def test_first_json_object_ignores_braces_in_code():
    reply = ('{"action": "propose", "why": "x"}\n'
             '```rust section=a\nfn f() { if x { y } }\n```')
    obj = _first_json_object(reply)
    ok(obj and obj["action"] == "propose",
       "the leading JSON object is read past Rust braces that follow")
    ok(_first_json_object('{"a": "brace } in a string"}')["a"].endswith("string"),
       "a brace inside a JSON string does not terminate the object")
    ok(_first_json_object("no json at all") is None, "a reply with no object is None")


# ---------------------------------------------------------------------- loop

def _sr(tmp):
    (tmp / "root/src").mkdir(parents=True)
    (tmp / "root/src/lib.rs").write_text("pub fn alpha() {}\n")
    return StagedRoot(root=(tmp / "root").resolve())


def test_agent_reads_then_proposes():
    tmp = Path(tempfile.mkdtemp())
    try:
        sr = _sr(tmp)
        llm = OrderedLLM([
            '{"action": "read", "calls": [{"tool": "grep", "pattern": "alpha", "path": ""}]}',
            '{"action": "propose", "why": "found it"}\n```rust section=a\nfn a() {}\n```',
        ])
        res = asyncio.run(run_agent(llm, sr, "task"))
        ok(res.action == "propose", "the loop reaches a proposal")
        ok(res.edits == {"a": "fn a() {}"}, "and carries the parsed edits")
        ok(res.turns == 2 and res.tool_calls == 1, "turns and tool calls are counted")
        ok("alpha" in llm.prompts[1],
           "the tool ANSWER is fed back into the next prompt (the SUCCESS path)")
        ok(res.record["action"] == "propose" and res.record["sections"] == ["a"],
           "the record names the sections proposed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_agent_gives_up():
    tmp = Path(tempfile.mkdtemp())
    try:
        res = asyncio.run(run_agent(
            OrderedLLM(['{"action": "give_up", "why": "nothing says what it does"}']),
            _sr(tmp), "task"))
        ok(res.action == "give_up" and "nothing says" in res.why,
           "give_up is carried with its reason")
        ok(res.edits == {}, "and proposes nothing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_propose_without_code_is_re_asked_once_then_stops():
    """The flat-schema failure in another costume: claiming the work and
    sending none. Nine rejections were spent on that shape before."""
    tmp = Path(tempfile.mkdtemp())
    try:
        llm = OrderedLLM(['{"action": "propose", "why": "done"}',
                          '{"action": "propose", "why": "done"}'])
        res = asyncio.run(run_agent(llm, _sr(tmp), "task"))
        ok(res.action == "give_up", "a propose with no code twice becomes a give_up")
        ok(res.turns == 2, "after exactly one re-ask, not a loop")
        ok("Send the code" in llm.prompts[1], "and the re-ask says what was missing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_propose_without_code_then_with_it_succeeds():
    tmp = Path(tempfile.mkdtemp())
    try:
        llm = OrderedLLM(['{"action": "propose", "why": "done"}',
                          '{"action": "propose", "why": "done"}\n'
                          '```rust section=a\nfn a() {}\n```'])
        res = asyncio.run(run_agent(llm, _sr(tmp), "task"))
        ok(res.action == "propose" and res.edits,
           "the recovery path works, not just the giving-up path")
        ok(res.stopped_by == "",
           "and a recovered firing does not report a stale stop reason")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_budget_caps():
    tmp = Path(tempfile.mkdtemp())
    try:
        sr = _sr(tmp)
        read = '{"action": "read", "calls": [{"tool": "list", "path": ""}]}'
        res = asyncio.run(run_agent(llm := OrderedLLM([read] * 20), sr, "task",
                                    budget=AgentBudget(max_turns=3)))
        ok(res.action == "give_up" and "turn cap" in res.stopped_by,
           "the turn cap stops the loop and says so")
        ok(res.turns <= 3, "and is actually enforced")

        many = ('{"action": "read", "calls": ['
                + ",".join(['{"tool": "list", "path": ""}'] * 10) + ']}')
        res2 = asyncio.run(run_agent(OrderedLLM([many] * 20), sr, "task",
                                     budget=AgentBudget(max_turns=50,
                                                        max_tool_calls=12)))
        ok("tool-call cap" in res2.stopped_by,
           "the tool-call cap is separate from the turn cap and fires")
        ok(res2.tool_calls <= 20, "and bounds the calls actually made")

        res3 = asyncio.run(run_agent(OrderedLLM([read] * 5), sr, "task",
                                     budget=AgentBudget(wall_seconds=-1)))
        ok("wall-clock" in res3.stopped_by, "the wall-clock cap fires")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transport_failure_is_survived_and_recorded():
    tmp = Path(tempfile.mkdtemp())
    try:
        res = asyncio.run(run_agent(RaisingLLM(), _sr(tmp), "task"))
        ok(res.action == "give_up", "a raising transport does not kill the run")
        ok("RuntimeError" in res.stopped_by,
           "and the failure is recorded rather than made invisible")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_garbage_reply_is_survived():
    tmp = Path(tempfile.mkdtemp())
    try:
        llm = OrderedLLM(["not json at all", "still not json",
                          '{"action": "give_up", "why": "ok"}'])
        res = asyncio.run(run_agent(llm, _sr(tmp), "task"))
        ok(res.action == "give_up", "unparseable replies are re-prompted, not fatal")
        ok(res.turns == 3, "and the loop recovers when a valid reply arrives")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"ALL PASS ({PASS} assertions)")
