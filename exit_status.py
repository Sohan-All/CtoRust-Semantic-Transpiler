"""Return-flow analysis (no LLM) — which C functions' return values become the
process exit status, and which distinct codes each can produce.

Motivation. C's `run_script` returns a distinct 2 (`cli.c:231`); the generated
Rust gave it `Result<(), EditorError>` and the entry point collapsed that to
`Err(_) => 1`, so exit code 2 was reported as 1. Nothing catches this: it type
checks, it builds, and only a differential test on the process exit status sees
it. The prompt rule telling units to preserve distinct exit codes reached
`cli_run` — whose C `main` calls directly — and stopped there, because
`run_script`'s unit has no way to know its own return value ends up as the
program's status. This module computes that fact so the unit can be told.

NOT the call graph. The edge here is "A returns B's value", which is a strict
subgraph of "A calls B". `run_script` calls `handle_script_command` inside an
`if`, so that value dies at the condition and never becomes an exit code; on
call edges it would be tagged, and so would the four `handle_*` functions it
tail-calls. Return-flow edges stop at the condition and the whole subtree drops
out on its own.

The relation is walked BACKWARDS from `main`: only functions feeding `main`'s
return can influence the status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter_c
from tree_sitter import Language, Parser

C_LANGUAGE = Language(tree_sitter_c.language())

# Key for a function definition: (project-relative file, C name). Keyed by file
# as well as name so a `static` helper cannot be confused with a same-named
# function in a sibling file — C's own shadowing rule.
FnKey = tuple[str, str]


@dataclass
class _Local:
    """What one function's body does with its return values, before any
    cross-function resolution."""
    edges: set[str] = field(default_factory=set)      # callee NAMES it returns
    literals: set[int] = field(default_factory=set)   # `return 2` / `rc = 2`
    symbolic: set[str] = field(default_factory=set)   # `return EXIT_FAILURE`
    computed: bool = False                            # ternary/arithmetic/etc.


@dataclass
class ExitStatus:
    """Emitted per function whose return value reaches the exit status."""
    codes: list[int]          # distinct literal codes, transitively
    path: list[str]           # this fn -> ... -> main
    computed: bool            # some contributing return was not a literal
    symbolic: list[str]       # macro names seen instead of literals

    @property
    def exhaustive(self) -> bool:
        """False when `codes` may be missing values — a computed return
        expression or an unexpanded macro somewhere in the chain."""
        return not self.computed and not self.symbolic


def _fn_name(node) -> str | None:
    """Name of a function_definition, unwrapping pointer/paren declarators."""
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type == "function_declarator":
            inner = decl.child_by_field_name("declarator")
            if inner is not None and inner.type == "identifier":
                return inner.text.decode()
            decl = inner
        elif decl.type in ("pointer_declarator", "parenthesized_declarator"):
            decl = decl.child_by_field_name("declarator") or next(
                (c for c in decl.children if c.is_named), None)
        else:
            return decl.text.decode() if decl.type == "identifier" else None
    return None


def _int_literal(node) -> int | None:
    """`node` as an int if it is a plain integer literal, else None. Handles the
    unary minus C uses for `return -1`."""
    if node is None:
        return None
    if node.type == "number_literal":
        text = node.text.decode().rstrip("uUlL")
        try:
            return int(text, 0)
        except ValueError:
            return None
    if node.type == "unary_expression":
        op = node.child_by_field_name("operator")
        arg = node.child_by_field_name("argument")
        if op is not None and op.text == b"-":
            inner = _int_literal(arg)
            return None if inner is None else -inner
    if node.type == "parenthesized_expression":
        return _int_literal(next((c for c in node.children if c.is_named), None))
    return None


def _callee(node) -> str | None:
    """Callee name if `node` is a direct call of a named function."""
    if node is not None and node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None and fn.type == "identifier":
            return fn.text.decode()
    return None


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _sources_of(body, var: str) -> tuple[set[str], set[int], set[str], bool]:
    """What flows into local `var` inside `body`: callee names, literal values,
    macro identifiers, and whether anything else did.

    Covers both `rc = f(...)` (assignment_expression) and `int rc = 0`
    (init_declarator) — the C uses both for the same variable.
    """
    edges: set[str] = set()
    lits: set[int] = set()
    syms: set[str] = set()
    other = False
    for n in _walk(body):
        rhs = None
        if n.type == "assignment_expression":
            lhs = n.child_by_field_name("left")
            if lhs is not None and lhs.type == "identifier" and lhs.text.decode() == var:
                rhs = n.child_by_field_name("right")
        elif n.type == "init_declarator":
            lhs = n.child_by_field_name("declarator")
            if lhs is not None and lhs.type == "identifier" and lhs.text.decode() == var:
                rhs = n.child_by_field_name("value")
        if rhs is None:
            continue
        name = _callee(rhs)
        lit = _int_literal(rhs)
        if name is not None:
            edges.add(name)
        elif lit is not None:
            lits.add(lit)
        elif rhs.type == "identifier":
            syms.add(rhs.text.decode())
        else:
            other = True
    return edges, lits, syms, other


def _analyze(body) -> _Local:
    """Classify every `return` in one function body."""
    loc = _Local()
    for n in _walk(body):
        if n.type != "return_statement":
            continue
        expr = next((c for c in n.children if c.is_named), None)
        if expr is None:                       # bare `return;` — void, no status
            continue
        name = _callee(expr)
        lit = _int_literal(expr)
        if name is not None:
            loc.edges.add(name)
        elif lit is not None:
            loc.literals.add(lit)
        elif expr.type == "identifier":
            # `return rc;` — trace what was written into rc. A name that is
            # never assigned locally is treated as a macro/global constant.
            e, l, s, other = _sources_of(body, expr.text.decode())
            if not (e or l or s or other):
                loc.symbolic.add(expr.text.decode())
            loc.edges |= e
            loc.literals |= l
            loc.symbolic |= s
            loc.computed = loc.computed or other
        else:
            loc.computed = True                # ternary, arithmetic, comparison
    return loc


def analyze_project(sources: dict[str, str],
                    root: str = "main") -> dict[FnKey, ExitStatus]:
    """{(file, fn): ExitStatus} for every function whose return value reaches
    the process exit status.

    `sources` is {project-relative file: C text}. `root` is the C entry point
    (`main`; run_project finds its file via `idx.defines`).
    """
    parser = Parser(C_LANGUAGE)

    locals_by_key: dict[FnKey, _Local] = {}
    defined_in: dict[str, str] = {}          # fn name -> file (last wins)
    per_file_names: dict[str, set[str]] = {}
    for fname, text in sources.items():
        tree = parser.parse(text.encode())
        names: set[str] = set()
        for node in _walk(tree.root_node):
            if node.type != "function_definition":
                continue
            name = _fn_name(node)
            body = node.child_by_field_name("body")
            if name is None or body is None:
                continue
            locals_by_key[(fname, name)] = _analyze(body)
            defined_in.setdefault(name, fname)
            names.add(name)
        per_file_names[fname] = names

    def resolve(from_file: str, name: str) -> FnKey | None:
        """C shadowing: a same-file definition wins over a sibling's."""
        if name in per_file_names.get(from_file, ()):
            return (from_file, name)
        owner = defined_in.get(name)
        return (owner, name) if owner else None

    root_key = resolve(defined_in.get(root, ""), root) if root in defined_in else None
    if root_key is None:
        return {}

    # resolved return-flow edges, both directions
    out_edges: dict[FnKey, set[FnKey]] = {}
    for key, loc in locals_by_key.items():
        targets = {t for t in (resolve(key[0], n) for n in loc.edges) if t}
        out_edges[key] = targets

    # (1) membership — closure from the root over return-flow edges. An edge
    # A -> B means "A returns B's value", so the root's own edges lead to the
    # functions feeding it and the traversal runs FORWARD from `main`. (The
    # data flows the other way, toward main; the edges point at its sources.)
    members: set[FnKey] = {root_key}
    frontier = [root_key]
    while frontier:
        key = frontier.pop()
        for t in out_edges.get(key, ()):
            if t not in members:
                members.add(t)
                frontier.append(t)

    # (2) code sets — a function's codes are its own literals plus those of
    # every function it returns through. Union to a fixpoint; monotone, so
    # cycles converge instead of recursing forever.
    codes: dict[FnKey, set[int]] = {k: set(locals_by_key[k].literals) for k in members}
    syms: dict[FnKey, set[str]] = {k: set(locals_by_key[k].symbolic) for k in members}
    comp: dict[FnKey, bool] = {k: locals_by_key[k].computed for k in members}
    changed = True
    while changed:
        changed = False
        for key in members:
            for t in out_edges.get(key, ()):
                if t not in members:
                    continue
                if not codes[t] <= codes[key]:
                    codes[key] |= codes[t]
                    changed = True
                if not syms[t] <= syms[key]:
                    syms[key] |= syms[t]
                    changed = True
                if comp[t] and not comp[key]:
                    comp[key] = True
                    changed = True

    # Shortest path along the direction the VALUE travels, for the prompt block:
    # "run_script -> cli_run -> main". An edge A -> B means A returns B's value,
    # so B's value travels to A — the display path follows edges reversed.
    paths: dict[FnKey, list[str]] = {root_key: [root_key[1]]}
    frontier = [root_key]
    while frontier:
        nxt: list[FnKey] = []
        for key in frontier:
            for t in out_edges.get(key, ()):
                if t in members and t not in paths:
                    paths[t] = [t[1]] + paths[key]
                    nxt.append(t)
        frontier = nxt

    return {k: ExitStatus(codes=sorted(codes[k]),
                          path=paths.get(k, [k[1]]),
                          computed=comp[k],
                          symbolic=sorted(syms[k]))
            for k in members}


def interesting(info: ExitStatus) -> bool:
    """Whether this function's codes are worth telling its unit about.

    A function returning only 0 and/or 1 is already served correctly by the
    idiomatic `Result` -> `Err(_) => 1` mapping, so a block there would be pure
    prompt noise. Any code outside {0, 1} is precisely what that mapping cannot
    express.

    `symbolic` and `computed` also qualify, because both mean the code set is
    NOT known to be a subset of {0, 1} — a `return n > 3 ? 2 : 0` yields no
    literals at all, and staying silent there is the exact failure this module
    exists to prevent. Measured across B03_organic these two add nothing to the
    21-of-1162 emission count, so the honesty is free.
    """
    return (any(c not in (0, 1) for c in info.codes)
            or bool(info.symbolic) or info.computed)
