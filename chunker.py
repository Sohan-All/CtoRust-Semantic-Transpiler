"""Stage 0 — deterministic syntax seeding with tree-sitter (no LLM).

Produces:
  - seed blocks: ordered, non-overlapping line ranges covering the whole file
  - intra-file call graph over function-defining seed blocks
  - SCC condensation of that graph + a topological order (leaves first)

Line numbers are 1-indexed and inclusive throughout, matching editor conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter_c
from tree_sitter import Language, Parser

C_LANGUAGE = Language(tree_sitter_c.language())

TOP_LEVEL_KINDS = {
    "function_definition",
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
    "type_definition",
    "declaration",
    "preproc_def",
    "preproc_function_def",
    "preproc_include",
    "preproc_ifdef",
    "preproc_if",
    "comment",
}


@dataclass
class SeedBlock:
    id: str
    start: int                 # 1-indexed inclusive
    end: int                   # 1-indexed inclusive
    kind: str                  # tree-sitter node kind (or "gap" / "sub_statement")
    function: str | None = None  # enclosing/defined function name, if any
    calls_internal: list[str] = field(default_factory=list)   # names of same-file functions called
    calls_external: list[str] = field(default_factory=list)   # everything else
    # callee name -> the DISTINCT call expressions this block makes to it.
    # A callee's name alone does not say how it is called, and some parameters
    # only mean anything at the call site: C's parse_options(argc, argv, 2, &o)
    # vs (argc, argv, 3, &o) is the whole difference between "skip the
    # subcommand" and "skip the subcommand AND its target". A unit specifying
    # that signature in isolation cannot recover the 2-vs-3 from the body.
    call_sites: dict[str, list[str]] = field(default_factory=dict)
    is_public: bool = False      # function without `static` (part of the C ABI surface)
    c_signature: str = ""        # declaration text up to the body, for FFI shims
    # parameter name -> its C declaration, for parameters that ARE functions
    # (`int (*cmp)(const void *, const void *)`). Held separately because these
    # are the one kind of callee that is neither same-file nor external: the
    # CALLER supplies the behaviour, so the answer is a generic bounded by `Fn`,
    # never a `_deps` stub. See _callback_params for what that mistake cost.
    callback_params: dict[str, str] = field(default_factory=dict)

    def text(self, source_lines: list[str]) -> str:
        return "\n".join(source_lines[self.start - 1 : self.end])


@dataclass
class SeedGraph:
    blocks: list[SeedBlock]                     # file order, full coverage
    call_edges: list[tuple[str, str]]           # (caller block id, callee block id)
    scc_groups: list[list[str]]                 # block-id groups that must merge (|group| > 1 only)
    topo_order: list[str]                       # block ids, callees before callers

    def to_record(self) -> dict:
        return {
            "type": "seed_graph",
            "blocks": [
                {
                    "id": b.id, "start": b.start, "end": b.end, "kind": b.kind,
                    "function": b.function,
                    "calls_internal": b.calls_internal,
                    "calls_external": b.calls_external,
                    "call_sites": b.call_sites,
                    "is_public": b.is_public,
                    "c_signature": b.c_signature,
                    "callback_params": b.callback_params,
                }
                for b in self.blocks
            ],
            "call_edges": self.call_edges,
            "scc_groups": self.scc_groups,
            "topo_order": self.topo_order,
        }


def _node_lines(node) -> tuple[int, int]:
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    # nodes ending in a newline (preproc directives) report end at col 0 of the
    # NEXT line; pull the end back so ranges don't overlap the following node
    if node.end_point[1] == 0 and end > start:
        end -= 1
    return start, end


def _function_name(node) -> str | None:
    """Extract the defined name from a function_definition node."""
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type == "function_declarator":
            inner = decl.child_by_field_name("declarator")
            if inner is not None and inner.type == "identifier":
                return inner.text.decode()
            decl = inner
        elif decl.type in ("pointer_declarator", "parenthesized_declarator"):
            decl = decl.child_by_field_name("declarator") or next(
                (c for c in decl.children if c.is_named), None
            )
        else:
            return decl.text.decode() if decl.type == "identifier" else None
    return None


def _is_public(node) -> bool:
    """A function_definition is public unless it carries `static`."""
    for c in node.children:
        if c.type == "storage_class_specifier" and c.text == b"static":
            return False
    return True


def _c_signature(node, source: str) -> str:
    """Declaration text of a function_definition up to (not including) the
    body — the C signature an extern \"C\" shim must reproduce."""
    body = node.child_by_field_name("body")
    end_byte = body.start_byte if body is not None else node.end_byte
    sig = source.encode()[node.start_byte:end_byte].decode(errors="replace")
    return " ".join(sig.split())


def _split_top_level(params: str) -> list[str]:
    """Split a C parameter list on commas at paren depth 0.

    A function-pointer parameter carries its own commas inside its own parens —
    `int (*cmp)(const void *, const void *)` is ONE parameter containing two —
    so a plain `.split(",")` shreds exactly the parameters this module exists to
    find."""
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [p.strip() for p in out if p.strip()]


# `(*name)(` — a parameter that IS a function. The name is optional in a
# declaration (`void (*)(void)`); an unnamed one cannot be called from the body
# and so cannot be misread as an external call, which is what this detects for.
_FN_PTR_PARAM = re.compile(r"\(\s*\*\s*(\w+)\s*\)\s*\(")

# `typedef int (*avl_compare_t)(...)` — the SAME parameter written through a
# name. Every corpus project that does this keeps the typedef in a HEADER, so a
# single .c file's text can never resolve it; `fn_pointer_typedefs` is run over
# the project's headers and the names handed to `chunk`.
_FN_PTR_TYPEDEF = re.compile(r"typedef\s+[\w\s\*]+\(\s*\*\s*(\w+)\s*\)\s*\(")


def fn_pointer_typedefs(text: str) -> set[str]:
    """Names typedef'd to a function-pointer type, e.g. `ArrayListCompareFunc`.

    Callers pass the union over a project's headers into `chunk`. Without it the
    detection below is blind to `void arraylist_sort(ArrayListCompareFunc cmp)`,
    which is how 5 of the 28 corpus projects declare their callbacks — including
    `array_list`, `binary_heap` and `binomial_heap`. Missing them leaks
    `compare_func` and `callback` into `calls_external` exactly as the raw
    `(*fn)(...)` form did."""
    return set(_FN_PTR_TYPEDEF.findall(text))


def _callback_params(c_signature: str,
                     fn_ptr_types: frozenset[str] = frozenset()) -> dict[str, str]:
    """{parameter name: its full C declaration text} for every function-pointer
    parameter of a function.

    WHY THIS EXISTS. `calls_external` is "callees not defined in this file",
    which is the right test for a sibling or a libc call and the WRONG one for a
    callback parameter: the body calls `pred(item)`, `pred` is a parameter, and
    it lands in external deps. Stage T then stubs it in `<stem>_deps` as a
    single global function — unfillable by construction, because the callers
    pass a different predicate at each call site — and `sibling_deps` has been
    observed INVENTING a body for one (`cp` -> `Ok(item.clone())`), which
    compiles and is silently wrong. Voided every `cc_array` run ever recorded.

    Operates on `c_signature` (already whitespace-collapsed, and populated for
    every block of a function including split ones), so no tree-sitter walk is
    needed and split `sub_statement` blocks are covered for free."""
    open_paren = c_signature.find("(")
    if open_paren < 0:
        return {}
    # the parameter list is the balanced span after the FIRST `(` — for
    # `char *(*factory(int n))(void)` that is `int n`, the parameters of the
    # function being defined, which is what we want
    depth, close = 0, -1
    for i in range(open_paren, len(c_signature)):
        if c_signature[i] == "(":
            depth += 1
        elif c_signature[i] == ")":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close < 0:
        return {}
    out: dict[str, str] = {}
    for p in _split_top_level(c_signature[open_paren + 1:close]):
        m = _FN_PTR_PARAM.search(p)
        if m:
            out[m.group(1)] = p
            continue
        # `ArrayListCompareFunc compare_func` — same thing through a typedef.
        # Take the LAST identifier as the parameter name; a bare `Fn_t` with no
        # name cannot be called from the body, so it is correctly skipped.
        words = re.findall(r"\w+", p)
        if len(words) >= 2 and words[-2] in fn_ptr_types:
            out[words[-1]] = p
    return out


def _collect_calls(node) -> list[str]:
    """All call_expression callee identifiers under node (document order, deduped)."""
    seen: list[str] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                name = fn.text.decode()
                if name not in seen:
                    seen.append(name)
        stack.extend(reversed(n.children))
    return seen


_MAX_CALL_TEXT = 200
_MAX_SITES_PER_CALLEE = 4


def _collect_call_sites(node) -> dict[str, list[str]]:
    """callee identifier -> its DISTINCT call expressions under `node`.

    Distinct rather than all: three identical `f(a, b, 2)` calls carry the same
    information as one, while a fourth `f(a, b, 3)` is exactly the fact worth
    surfacing. Long calls are truncated — the leading arguments are the ones
    that disambiguate."""
    sites: dict[str, list[str]] = {}
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                name = fn.text.decode()
                text = " ".join(n.text.decode(errors="replace").split())
                if len(text) > _MAX_CALL_TEXT:
                    text = text[:_MAX_CALL_TEXT] + " ...)"
                seen = sites.setdefault(name, [])
                if text not in seen and len(seen) < _MAX_SITES_PER_CALLEE:
                    seen.append(text)
        stack.extend(reversed(n.children))
    return sites


def _split_large_function(node, fid_base: str, name: str | None, threshold: int) -> list[SeedBlock]:
    """Split a long function one level down: header + top-level body statements.

    Consecutive small statements are coalesced so we don't produce dozens of
    one-line blocks; each emitted block covers >= 1 statement and the set of
    blocks exactly covers the function's line span.
    """
    start, end = _node_lines(node)
    body = node.child_by_field_name("body")
    if body is None:
        return [SeedBlock(fid_base, start, end, "function_definition", name)]

    stmts = [c for c in body.children if c.is_named and c.type != "comment"] or []
    # cut points: statement start lines; coalesce so each piece is <= threshold lines
    pieces: list[tuple[int, int]] = []
    cur_start = start
    for stmt in stmts:
        s_start, s_end = _node_lines(stmt)
        if s_end - cur_start + 1 > threshold and s_start > cur_start:
            pieces.append((cur_start, s_start - 1))
            cur_start = s_start
    pieces.append((cur_start, end))

    if len(pieces) == 1:
        return [SeedBlock(fid_base, start, end, "function_definition", name)]
    return [
        SeedBlock(f"{fid_base}_p{i}", s, e, "sub_statement", name)
        for i, (s, e) in enumerate(pieces)
    ]


def chunk(source: str, split_over: int = 40,
          fn_ptr_types: frozenset[str] | None = None) -> SeedGraph:
    """`fn_ptr_types`: names typedef'd to function-pointer types, from
    `fn_pointer_typedefs` over the project's HEADERS. Optional and defaulted so
    every existing caller keeps working — omitting it only costs the typedef'd
    form of a callback parameter, never the literal `(*fn)(...)` one. The
    typedefs are unioned from the whole project because a .c file's callbacks
    are declared in a .h it includes."""
    fn_ptr_types = fn_ptr_types if fn_ptr_types is not None else frozenset()
    # the file's own typedefs count too, without the caller having to know
    fn_ptr_types = fn_ptr_types | fn_pointer_typedefs(source)
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(source.encode())
    source_lines = source.split("\n")
    n_lines = len(source_lines)

    blocks: list[SeedBlock] = []
    fn_calls: dict[str, list[str]] = {}          # function name -> called identifiers
    fn_call_sites: dict[str, dict[str, list[str]]] = {}  # caller -> callee -> call texts
    fn_block_ids: dict[str, list[str]] = {}      # function name -> its block ids

    idx = 0
    for node in tree.root_node.children:
        if not node.is_named:
            continue
        start, end = _node_lines(node)
        bid = f"seed_{idx:03d}"
        idx += 1
        if node.type == "function_definition":
            name = _function_name(node)
            calls = _collect_calls(node)
            if name:
                fn_calls[name] = calls
                fn_call_sites[name] = _collect_call_sites(node)
            if end - start + 1 > split_over:
                fblocks = _split_large_function(node, bid, name, split_over)
            else:
                fblocks = [SeedBlock(bid, start, end, node.type, name)]
            public = _is_public(node)
            sig = _c_signature(node, source)
            for b in fblocks:
                b.is_public = public
                b.c_signature = sig
            if name:
                fn_block_ids[name] = [b.id for b in fblocks]
            blocks.extend(fblocks)
        else:
            blocks.append(SeedBlock(bid, start, end, node.type))

    # fill gaps (blank lines / stray content between top-level nodes) so coverage
    # is total; merge each gap into the following block when possible
    blocks.sort(key=lambda b: b.start)
    covered: list[SeedBlock] = []
    cursor = 1
    for b in blocks:
        if b.start > cursor:
            gap_text = "\n".join(source_lines[cursor - 1 : b.start - 1]).strip()
            if gap_text:
                covered.append(SeedBlock(f"gap_{cursor}", cursor, b.start - 1, "gap"))
            else:
                b.start = cursor  # absorb pure whitespace into the next block
        covered.append(b)
        cursor = max(cursor, b.end + 1)
    if cursor <= n_lines:
        tail = "\n".join(source_lines[cursor - 1 :]).strip()
        if tail:
            covered.append(SeedBlock(f"gap_{cursor}", cursor, n_lines, "gap"))
        elif covered:
            covered[-1].end = n_lines  # absorb trailing whitespace

    # comments, includes, and gap content are context, not behavior — absorb
    # each run of them into the FOLLOWING block so they never become standalone
    # units (a trailing run attaches to the preceding block instead)
    ABSORB_KINDS = {"comment", "preproc_include", "gap"}
    absorbed: list[SeedBlock] = []
    pending_start: int | None = None
    for b in covered:
        if b.kind in ABSORB_KINDS:
            if pending_start is None:
                pending_start = b.start
            continue
        if pending_start is not None:
            b.start = pending_start
            pending_start = None
        absorbed.append(b)
    if pending_start is not None:
        if absorbed:
            absorbed[-1].end = covered[-1].end
        else:  # file of nothing but comments/includes
            absorbed = covered
    blocks = absorbed

    # annotate call info per block
    defined = set(fn_calls.keys())
    for b in blocks:
        if b.function and b.function in fn_calls:
            calls = fn_calls[b.function]
            # A function-pointer PARAMETER is called by name in the body but is
            # neither defined here nor implemented elsewhere — the caller
            # supplies it. Pull those out before the external split, or they are
            # handed to stage T as external domain functions and stubbed in
            # `<stem>_deps`, which is unfillable and has been observed being
            # fabricated instead. `calls_internal` is deliberately left alone: a
            # parameter shadowing a same-file function name is pathological, and
            # the existing behaviour there is no worse than before.
            b.callback_params = _callback_params(b.c_signature, fn_ptr_types)
            b.calls_internal = [c for c in calls if c in defined and c != b.function]
            b.calls_external = [c for c in calls
                                if c not in defined and c not in b.callback_params]
            # every callee, not just same-file ones: in a multi-file project
            # roughly half the call graph crosses a file boundary (18/42 of
            # binary_heap's functions, 24/49 of double_linked_list's), and
            # those call sites are exactly what a unit needs to see. Consumers
            # index this by the callees they care about, so the extra keys
            # (libc, siblings) cost storage here and nothing downstream.
            b.call_sites = dict(fn_call_sites.get(b.function, {}))

    # call edges between blocks (caller's blocks -> callee's blocks)
    edges: list[tuple[str, str]] = []
    for caller, callees in fn_calls.items():
        for callee in callees:
            if callee in fn_block_ids and callee != caller:
                for cb in fn_block_ids.get(caller, []):
                    for eb in fn_block_ids[callee]:
                        edges.append((cb, eb))

    scc_groups, topo = _condense(
        [b.id for b in blocks], edges
    )
    return SeedGraph(blocks=blocks, call_edges=edges,
                     scc_groups=[g for g in scc_groups if len(g) > 1],
                     topo_order=topo)


def _condense(node_ids: list[str], edges: list[tuple[str, str]]) -> tuple[list[list[str]], list[str]]:
    """Tarjan SCC + topological order of the condensation, callees first.

    Returns (scc_groups, topo_order_of_block_ids). Within the topo order, all
    members of an SCC appear consecutively.
    """
    adj: dict[str, list[str]] = {n: [] for n in node_ids}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)

    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = [0]

    def strongconnect(v: str) -> None:
        # iterative Tarjan to avoid recursion limits on big files
        work = [(v, iter(adj[v]))]
        index_of[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        while work:
            node, it = work[-1]
            advanced = False
            for w in it:
                if w not in index_of:
                    index_of[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(adj[w])))
                    advanced = True
                    break
                elif w in on_stack:
                    low[node] = min(low[node], index_of[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                group = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    group.append(w)
                    if w == node:
                        break
                sccs.append(group)

    for n in node_ids:
        if n not in index_of:
            strongconnect(n)

    # Tarjan emits SCCs in reverse topological order of the condensation
    # (callees/sinks first) — exactly the order we want (leaves first).
    topo: list[str] = [n for group in sccs for n in group]
    return sccs, topo
