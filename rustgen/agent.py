"""A read-capable escalation agent. See ../../plan.md item 6.

WHY THIS REPLACES A QUESTION MENU. The first paired escalation batch fired three
times and accepted nothing, and all three refusals were harness faults:

  - stage T emitted `pub mod deps`, `_DEPS_MOD` requires the literal `_deps`, and
    a legal stub was reported as illegal. The escalated model was handed a
    CORRECT block and asked to fix a defect that was not there;
  - a stub cited `scheduler_print_report`, whose behaviour was present in the
    crate as `impl Display for Scheduler`, and the name-keyed `defines()` handler
    answered "Nothing in this crate defines it. If the stub is waiting for it,
    nobody is going to provide it." Confidently wrong;
  - `unanswerable_stubs` gates on `has_callers(s.function)` where `Stub.function`
    is the ENCLOSING function, which is always called — so the gate that exists
    to refuse hopeless spends cannot detect one.

Those are three instances of one thing: a hand-written mechanical approximation
standing in for a semantic question. The fixed six-question menu assumes the
pipeline knows what information is missing, at exactly the point where it has
proven it does not. An agent that can read answers its own questions.

CONTAINMENT — READ THE STAGED ROOT SECTION BELOW BEFORE CHANGING ANYTHING HERE.
`plan.md`'s original Containment rule was "do not give the escalation agent
filesystem tools", and this module reverses the METHOD while keeping the
PROPERTY. The property is that the DARPA test vectors are structurally out of
reach. They are the metric, never an input, and an earlier repair loop in this
project was scored on vectors it had consumed. The guarantee here is that the
vectors are ABSENT FROM THE REACHABLE ROOT, never that the prompt asked nicely:
Containment's own warning is the acceptance criterion — *an agent with filesystem
search will find the test vectors, because for the task it has been set they are
genuinely the most useful file in the tree.*

WHAT THE AGENT MAY DO. Read, inside the staged root. Propose a transaction.
That is all. It does not write files; `rustgen.transaction.check_transaction`
decides, and every damage class this project has paid for was fixed by
constraining the output channel rather than by a better model (plan.md D4, D11).

NEVER PUT RUST INSIDE A JSON STRING. Tool calls are short metadata and use JSON.
The transaction comes back as fenced blocks, one per section, because under a
constrained grammar a model does not reliably escape quotes and newlines and
returns structurally plausible, completely corrupt code — measured live, see
TRIALS.md. Every other prompt in this pipeline returns a fence for the same
reason.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Directory names that must never appear in a staged root. Matched on the path
# COMPONENT, not as a substring, so a legitimately named source file is not
# caught and a directory cannot smuggle itself past by nesting.
FORBIDDEN_COMPONENTS = frozenset({
    "test_vectors", "test_vector", "vectors",
    "results", "_results", "score",
})

MAX_READ_BYTES = 40_000
MAX_GREP_HITS = 60
MAX_LIST_ENTRIES = 200

# Never staged. Not a containment rule — none of these holds the vectors — but a
# correctness and cost one, found by staging a real project and counting: the
# naive copy took 16,634 files, of which `venv/` was 15,619 and `out/` was
# recorded run state. Grepping a tree that is 94% third-party site-packages
# returns matches from libraries the agent cannot change, and copying it per
# firing fills a scratch disk over a batch.
IGNORE_DIRS = ("venv", ".venv", "site-packages", "__pycache__", ".git",
               "out", "target", "node_modules", ".pytest_cache", ".mypy_cache")


class StagedRootError(RuntimeError):
    """The staged root is not safe to hand an agent. Raised, never warned:
    a containment failure produces no visible bug, it produces a headline
    result that quietly is not publishable."""


@dataclass
class StagedRoot:
    """A directory the agent may read, with the vectors structurally absent.

    Construct with `stage()` rather than directly — the invariant is checked
    there, and a `StagedRoot` that was never checked looks exactly like one
    that was.
    """
    root: Path
    label: str = ""

    def resolve(self, rel: str) -> Path:
        """Map an agent-supplied path into the root, or raise.

        Refuses absolute paths and any `..` that escapes, by resolving and then
        asking whether the result is still under the root — string prefix checks
        on unresolved paths are the classic way this is got wrong, because
        `root/../root_evil` has the right prefix.
        """
        rel = (rel or "").strip().lstrip("/")
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            raise StagedRootError(f"path escapes the staged root: {rel!r}")
        if FORBIDDEN_COMPONENTS & set(p.parts):
            raise StagedRootError(f"path is out of bounds: {rel!r}")
        return p


def assert_contained(root: Path) -> None:
    """The containment invariant, as a check that runs on every stage().

    Walks the tree rather than trusting how it was built. Staging is a copy or
    a symlink farm assembled by a caller, and the caller is exactly the thing
    that can be wrong — this project's convention is that a gate belongs at the
    write point, not at the site that promises to behave.
    """
    root = Path(root)
    if not root.is_dir():
        raise StagedRootError(f"staged root does not exist: {root}")
    for p in root.rglob("*"):
        bad = FORBIDDEN_COMPONENTS & set(p.relative_to(root).parts)
        if bad:
            raise StagedRootError(
                f"staged root contains {sorted(bad)} at {p.relative_to(root)} — "
                f"the DARPA vectors are the metric, never an input")
        # A symlink pointing out of the root defeats the whole arrangement and
        # is invisible to a component check.
        if p.is_symlink():
            try:
                p.resolve().relative_to(root.resolve())
            except ValueError:
                raise StagedRootError(
                    f"staged root has a symlink escaping it: "
                    f"{p.relative_to(root)} -> {p.resolve()}")


def stage(dest: Path, *, crate_src: Path | None = None,
          c_src: Path | None = None, pipeline_src: Path | None = None,
          label: str = "") -> StagedRoot:
    """Build a staged root from the three things the agent legitimately needs.

    `pipeline_src` is included deliberately and is the reason site A is fixable
    at all: the `mod deps` refusal can only be diagnosed by reading
    `common.py:350` and discovering the gate wants `*_deps`. No amount of
    crate-and-C context reaches it.

    Copies rather than symlinks the C source, because the corpus layout puts
    `test_vectors/` as a SIBLING of `test_case/` and a symlinked tree invites a
    `..` that `resolve()` would then have to catch at request time. Cheaper to
    make it absent.
    """
    import shutil
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path | None, name: str, only_suffixes=None) -> None:
        if src is None:
            return
        src = Path(src)
        if not src.exists():
            return
        target = dest / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

        def _ignore(directory, entries):
            drop = set()
            for e in entries:
                if e in FORBIDDEN_COMPONENTS or e in IGNORE_DIRS:
                    drop.add(e)
                    continue
                p = Path(directory) / e
                if only_suffixes and p.is_file() and p.suffix not in only_suffixes:
                    drop.add(e)
            return drop
        shutil.copytree(src, target, symlinks=False, ignore=_ignore)

    _copy(crate_src, "crate")
    _copy(c_src, "c_source")
    # The pipeline is staged as SOURCE ONLY. The agent reads it to find out what
    # a gate wants — `_DEPS_MOD` wanting `*_deps` is the whole of site A — and
    # everything else in that tree is noise it would have to grep past.
    _copy(pipeline_src, "pipeline", only_suffixes={".py", ".md"})
    assert_contained(dest)
    return StagedRoot(root=dest.resolve(), label=label)


# ---------------------------------------------------------------- read tools

def tool_list(sr: StagedRoot, path: str = "") -> str:
    p = sr.resolve(path)
    if not p.exists():
        return f"(no such path: {path})"
    if p.is_file():
        return f"{path} is a file ({p.stat().st_size} bytes)"
    rows = []
    for child in sorted(p.rglob("*"))[:MAX_LIST_ENTRIES]:
        rel = child.relative_to(sr.root)
        rows.append(f"  {rel}{'/' if child.is_dir() else ''}")
    if not rows:
        return f"({path or '.'} is empty)"
    return f"under {path or '.'}:\n" + "\n".join(rows)


def tool_read(sr: StagedRoot, path: str, start: int = 1, end: int = 0) -> str:
    """Read a file, optionally a line range.

    Bounded in BYTES as well as lines. An agent's reads are individually cheap
    and unbounded in aggregate, and one unbounded read of a large C file can eat
    a whole run's token budget in a single call — which is the gap the existing
    `escalation_max_turns` does not cover, since it counts model turns.
    """
    p = sr.resolve(path)
    if not p.is_file():
        return f"(not a file: {path})"
    try:
        text = p.read_text(errors="ignore")
    except OSError as e:
        return f"(unreadable: {e})"
    lines = text.split("\n")
    start = max(1, int(start or 1))
    end = int(end) if end else len(lines)
    end = min(max(end, start), len(lines))
    body = "\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1))
    if len(body) > MAX_READ_BYTES:
        body = body[:MAX_READ_BYTES] + (
            f"\n... truncated at {MAX_READ_BYTES} bytes; "
            f"re-read with a narrower line range")
    return f"{path} lines {start}-{end} of {len(lines)}:\n{body}"


def tool_grep(sr: StagedRoot, pattern: str, path: str = "") -> str:
    """Regex search. This is the handler the fixed menu could not express — the
    `scheduler_print_report` refusal needed "what implements this behaviour",
    and a name lookup cannot answer it while `defines()` answers it wrongly."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"(bad regex: {e})"
    base = sr.resolve(path)
    files = [base] if base.is_file() else sorted(
        f for f in base.rglob("*") if f.is_file())
    hits = []
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.split("\n"), 1):
            if rx.search(line):
                hits.append(f"  {f.relative_to(sr.root)}:{n}: {line.strip()[:200]}")
                if len(hits) >= MAX_GREP_HITS:
                    return (f"matches for {pattern!r} (capped at "
                            f"{MAX_GREP_HITS}):\n" + "\n".join(hits))
    if not hits:
        return f"no matches for {pattern!r}"
    return f"matches for {pattern!r}:\n" + "\n".join(hits)


TOOLS = {"list": tool_list, "read": tool_read, "grep": tool_grep}

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["read", "propose", "give_up"]},
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["list", "read", "grep"]},
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["tool"],
            },
        },
        "why": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass
class AgentBudget:
    """Bounds. An agent loop is unbounded by construction and this project's
    first open gap is that nothing bounds a call or a run — a single call once
    held a run for 38 minutes with nothing in the log.

    `escalation_max_turns` counts MODEL turns and does not bound tool calls;
    `call_deadline` bounds one call and does not bound a loop. Both gaps are
    covered here.
    """
    max_turns: int = 8
    max_tool_calls: int = 40
    wall_seconds: float = 600.0
    turns: int = 0
    tool_calls: int = 0
    _t0: float = field(default_factory=time.monotonic)

    def exhausted(self) -> str:
        if self.turns >= self.max_turns:
            return f"turn cap reached ({self.max_turns})"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool-call cap reached ({self.max_tool_calls})"
        elapsed = time.monotonic() - self._t0
        if elapsed >= self.wall_seconds:
            return f"wall-clock cap reached ({self.wall_seconds:.0f}s)"
        return ""


# ------------------------------------------------------------------- parsing

_SECTION_FENCE = re.compile(
    r"```(?:rust)?\s+section=([A-Za-z_][\w:.$-]*)\s*\n(.*?)```", re.S)


def parse_transaction(reply: str) -> dict[str, str]:
    """Pull `​```rust section=<id>` blocks out of a reply.

    The section id rides on the FENCE INFO STRING, not in a JSON field beside
    the code, because the code must never be inside a JSON string. Short
    metadata in JSON, code in a fence — the rule the rest of this pipeline
    already follows and the one place it was broken produced Rust with every
    newline stripped and every string literal mangled.

    A later block for the same section wins, matching `extract_rust`'s
    last-fence-wins rule: a model that drafts and then revises emits both, and
    concatenating them duplicates every item.
    """
    out: dict[str, str] = {}
    for sid, body in _SECTION_FENCE.findall(reply or ""):
        out[sid] = body.strip()
    return out


AGENT_PROMPT = """\
You are repairing a Rust crate that was machine-translated from C. A previous,
cheaper model already tried and gave up, so re-reading the same context will not
help — the reason you are here is that you can go and LOOK.

{task}

You may read anything under the staged root:

  crate/     the generated Rust crate, as it stands
  c_source/  the original C this was translated from
  pipeline/  this translator's own source, including its lint rules

`pipeline/` is this project's code and you are one of its stages, so its lint
rules are yours to read. They are ordinary style and structure checks written as
regexes and name lookups — naming conventions, where a construct is allowed to
appear — and reading the rule is usually the quickest way to see which project
convention a rejected block missed.

Reply with a JSON object:

  {{"action": "read", "calls": [{{"tool": "grep", "pattern": "fn build_.*",
    "path": "crate"}}, {{"tool": "read", "path": "crate/src/lib.rs",
    "start": 1100, "end": 1200}}]}}

  {{"action": "propose", "why": "Display for Scheduler already renders the
    report; wire the call to it"}}

  {{"action": "give_up", "why": "nothing in the crate or the C says what this
    callback should do"}}

Tools: `list` (path), `read` (path, start, end), `grep` (pattern, path).

When you propose, put the JSON object FIRST and then one fenced block per
section you are changing, each tagged with its section id:

```rust section=cli__exp_0002
pub(crate) fn dispatch_command(...) -> bool {{
    ...the COMPLETE new text of this section...
}}
```

Rules that will get your edit refused if you break them:

- Send the COMPLETE text of every section you touch, not a diff and not an
  abridgement. Count the items in the section before you reply and make sure
  every one reappears with its real body. If you cannot fix a body, reproduce it
  unchanged.
- Your edits land as ONE transaction: all of them or none. That is why you may
  change a signature in one section and its call sites in another — do both in
  the same reply, or neither will land.
- Do not delete behaviour to make it compile. Removing the call that does not
  type-check makes the crate build and the translation wrong, and a wrong
  translation scores worse than an honest `todo!()`. If the honest answer is
  that this cannot be fixed without inventing behaviour, say `give_up`.
- Never emit a trait impl for a type this crate does not define.

Give up freely. It costs nothing and it is a normal outcome.
"""


@dataclass
class AgentResult:
    action: str = "give_up"
    why: str = ""
    edits: dict[str, str] = field(default_factory=dict)
    turns: int = 0
    tool_calls: int = 0
    stopped_by: str = ""
    transcript: list[dict] = field(default_factory=list)

    @property
    def record(self) -> dict:
        return {"type": "agent", "action": self.action, "why": self.why[:500],
                "sections": sorted(self.edits), "turns": self.turns,
                "tool_calls": self.tool_calls, "stopped_by": self.stopped_by,
                "transcript": self.transcript[-40:]}


async def run_agent(llm, sr: StagedRoot, task: str, *,
                    budget: AgentBudget | None = None,
                    max_tokens: int = 8000) -> AgentResult:
    """Read-and-propose loop. Returns what the agent wants to do; deciding
    whether it happens is `transaction.check_transaction`'s job.

    Never raises for an agent-side failure — a malformed reply, an unreadable
    path and an exhausted budget all resolve to a recorded `give_up`. A
    surviving failure must still be visible, so every one of them sets
    `stopped_by`.
    """
    budget = budget or AgentBudget()
    res = AgentResult()
    prompt = AGENT_PROMPT.format(task=task)
    empty_propose = False

    while True:
        stop = budget.exhausted()
        if stop:
            res.stopped_by = stop
            res.why = res.why or f"stopped: {stop}"
            return res
        budget.turns += 1
        res.turns = budget.turns
        try:
            reply = await llm.ask(prompt, max_tokens=max_tokens)
        except Exception as e:
            res.stopped_by = f"{type(e).__name__}: {e}"
            return res

        obj = _first_json_object(reply)
        action = str((obj or {}).get("action", "")).strip()

        if action == "propose":
            edits = parse_transaction(reply)
            res.why = str((obj or {}).get("why", ""))[:1000]
            res.tool_calls = budget.tool_calls
            if edits:
                res.action = "propose"
                res.edits = edits
                res.stopped_by = ""
                return res
            # A `propose` carrying no fenced block is the flat-schema failure in
            # another costume: the model claims the work and sends none, which
            # is what `{"action":"patch"}` did 9 times before DECIDE_SCHEMA lost
            # its `code` property. One re-ask, then stop rather than loop.
            if empty_propose:
                res.action = "give_up"
                res.stopped_by = "proposed without code twice"
                return res
            empty_propose = True
            prompt += ("\n\nYour last reply said `propose` and contained no "
                       "```rust section=<id> block. Send the code.")
            continue

        if action == "give_up":
            res.action = "give_up"
            res.why = str((obj or {}).get("why", ""))[:1000]
            res.tool_calls = budget.tool_calls
            return res

        if action == "read":
            calls = (obj or {}).get("calls") or []
            if not isinstance(calls, list) or not calls:
                prompt += "\n\nYou asked to read and named no calls."
                continue
            answers = []
            for call in calls:
                if budget.exhausted():
                    break
                if not isinstance(call, dict):
                    continue
                budget.tool_calls += 1
                answers.append(_dispatch(sr, call))
            res.tool_calls = budget.tool_calls
            res.transcript.append({"turn": budget.turns,
                                   "calls": [str(c)[:200] for c in calls[:10]]})
            prompt += ("\n\n--- you asked, and the answers are ---\n"
                       + "\n\n".join(answers)
                       + "\n\nNow read, propose or give_up.")
            continue

        prompt += ('\n\nReply with a JSON object whose "action" is one of '
                   '"read", "propose", "give_up".')


def _dispatch(sr: StagedRoot, call: dict) -> str:
    name = str(call.get("tool", "")).strip()
    fn = TOOLS.get(name)
    if fn is None:
        return f"({name}: unknown tool; use list, read or grep)"
    try:
        if name == "grep":
            return fn(sr, str(call.get("pattern", "")), str(call.get("path", "")))
        if name == "read":
            return fn(sr, str(call.get("path", "")),
                      int(call.get("start", 1) or 1), int(call.get("end", 0) or 0))
        return fn(sr, str(call.get("path", "")))
    except StagedRootError as e:
        # Deliberately legible rather than silent. An agent that wandered
        # towards the vectors should be visible in the transcript.
        return f"({name}: refused — {e})"
    except Exception as e:
        return f"({name}: failed — {type(e).__name__}: {e})"


def _first_json_object(text: str):
    """The leading JSON object of a reply that also carries fenced code.

    `extract_json` in `llm.py` takes the first JSON value in the whole reply,
    which is wrong here: a proposal's Rust can contain `{` and a greedy or
    last-wins scan finds the wrong thing. Scans for a balanced object from the
    first `{`, ignoring braces inside strings.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
