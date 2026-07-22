"""Stage 0 — deterministic syntax seeding with tree-sitter (no LLM).

Produces:
  - seed blocks: ordered, non-overlapping line ranges covering the whole file
  - intra-file call graph over function-defining seed blocks
  - SCC condensation of that graph + a topological order (leaves first)

Line numbers are 1-indexed and inclusive throughout, matching editor conventions.
"""

from __future__ import annotations

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
    is_public: bool = False      # function without `static` (part of the C ABI surface)
    c_signature: str = ""        # declaration text up to the body, for FFI shims

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
                    "is_public": b.is_public,
                    "c_signature": b.c_signature,
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


def chunk(source: str, split_over: int = 40) -> SeedGraph:
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(source.encode())
    source_lines = source.split("\n")
    n_lines = len(source_lines)

    blocks: list[SeedBlock] = []
    fn_calls: dict[str, list[str]] = {}          # function name -> called identifiers
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
            b.calls_internal = [c for c in calls if c in defined and c != b.function]
            b.calls_external = [c for c in calls if c not in defined]

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
