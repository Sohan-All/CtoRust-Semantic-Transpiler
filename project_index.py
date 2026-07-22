"""Project symbol index — the cross-file bridging map.

Multi-file translation needs three facts no single-file stage can see:
  1. which file DEFINES each function (so a file's external_deps split into
     NATIVE / EXTERNAL / SIBLING — sibling deps become plain Rust calls into
     the co-translated module, not extern "C"),
  2. which types are SHARED (declared in headers, referenced from more than
     one file) — these must be synthesized ONCE at project level so every
     file's spec/code stage builds against the same canonical Rust type,
  3. the FILE DEPENDENCY GRAPH and its SCC condensation — translation order
     is topological over SCCs; mutually-referencing files form one SCC and
     are CO-TRANSLATED (spec stage for the whole group first, then codegen —
     the same signature trick that breaks cycles within a file).

TRUST NOTE (adopted from forclift/producers/defn_graph.py): this index is a
PARTITION HINT, never an oracle. A missed edge (function pointers, macro
calls) only changes how work is grouped/ordered; behavioral validation
re-checks every symbol regardless. Approximate extraction is therefore safe.

The extractor is tree-sitter (shared with chunker.py), not a regex heuristic:
we already depend on it and it parses preprocessor-heavy C without a build.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_c
from tree_sitter import Language, Parser

from chunker import chunk

C_LANGUAGE = Language(tree_sitter_c.language())


# ---------------------------------------------------------------- tarjan ----
# Iterative Tarjan SCC, ported from forclift/core/graph.py (Nils): a cycle
# cannot be split ("you can't freeze a seam inside it"), so every SCC is one
# co-translation group. Iterative so symbol-scale graphs can't overflow the
# recursion limit; file-scale graphs get it for free.

def tarjan_scc(graph: dict[str, list[str]]) -> list[list[str]]:
    """SCCs in REVERSE topological order (a component precedes the components
    it depends on — leaves LAST). Deterministic for deterministic input."""
    adj: dict[str, list[str]] = {}
    for u, succs in graph.items():
        adj.setdefault(u, [])
        for v in succs:
            adj[u].append(v)
            adj.setdefault(v, [])

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = 0

    for root in adj:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack.add(v)
            recursed = False
            succs = adj[v]
            i = pi
            while i < len(succs):
                w = succs[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recursed = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
                i += 1
            if recursed:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(sorted(comp))
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    return sccs


# ----------------------------------------------------------------- index ----

@dataclass
class ProjectIndex:
    c_root: Path
    files: list[str]                            # .c files, project-relative
    defines: dict[str, str]                     # function -> defining file
    references: dict[str, set[str]]             # file -> function names used
    header_types: dict[str, str]                # type name -> declaring header
    type_refs: dict[str, set[str]]              # type name -> files using it
    file_graph: dict[str, list[str]] = field(default_factory=dict)
    groups: list[list[str]] = field(default_factory=list)  # SCCs, leaves first

    @property
    def shared_types(self) -> dict[str, str]:
        """Header-declared types referenced from more than one .c file — the
        set that project-level type synthesis must own."""
        return {t: h for t, h in self.header_types.items()
                if len(self.type_refs.get(t, ())) > 1}

    def dep_class(self, from_file: str, name: str) -> str:
        """NATIVE/EXTERNAL are downstream policy; here only the structural
        question: SIBLING (defined in another indexed file), SELF, or
        UNKNOWN (libc / genuinely external)."""
        owner = self.defines.get(name)
        if owner is None:
            return "UNKNOWN"
        return "SELF" if owner == from_file else "SIBLING"

    def to_record(self) -> dict:
        return {
            "type": "project_index",
            "c_root": str(self.c_root),
            "files": self.files,
            "defines": self.defines,
            "references": {f: sorted(s) for f, s in self.references.items()},
            "header_types": self.header_types,
            "type_refs": {t: sorted(s) for t, s in self.type_refs.items()},
            "file_graph": self.file_graph,
            "groups": self.groups,
        }


_TYPE_NODE_KINDS = ("struct_specifier", "union_specifier", "enum_specifier")


def _header_declared_types(header_src: str) -> list[str]:
    """Type names a header declares: struct/union/enum tags and typedef
    names. Tree-sitter walk, no preprocessing."""
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(header_src.encode())
    names: list[str] = []

    def walk(node):
        if node.type in _TYPE_NODE_KINDS:
            tag = node.child_by_field_name("name")
            if tag is not None:
                names.append(tag.text.decode())
        if node.type == "type_definition":
            decl = node.child_by_field_name("declarator")
            if decl is not None and decl.type == "type_identifier":
                names.append(decl.text.decode())
        for ch in node.children:
            walk(ch)

    walk(tree.root_node)
    return names


def _type_identifiers(src: str) -> set[str]:
    """All type identifiers a source file mentions (usage detection for the
    shared-type set)."""
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(src.encode())
    out: set[str] = set()

    def walk(node):
        if node.type == "type_identifier":
            out.add(node.text.decode())
        elif node.type in _TYPE_NODE_KINDS:
            tag = node.child_by_field_name("name")
            if tag is not None:
                out.add(tag.text.decode())
        for ch in node.children:
            walk(ch)

    walk(tree.root_node)
    return out


def build_index(c_root: Path, split_over: int = 40) -> ProjectIndex:
    c_root = Path(c_root)
    c_files = sorted(p for p in c_root.rglob("*.c"))
    h_files = sorted(p for p in c_root.rglob("*.h"))

    defines: dict[str, str] = {}
    references: dict[str, set[str]] = {}
    rel = {p: str(p.relative_to(c_root)) for p in c_files}

    for p in c_files:
        graph = chunk(p.read_text(), split_over)
        refs: set[str] = set()
        for b in graph.blocks:
            # a block carries .function iff it belongs to a definition —
            # large functions are split into sub_statement blocks that must
            # still register the definition
            if b.function:
                defines.setdefault(b.function, rel[p])
            refs.update(b.calls_internal)
            refs.update(b.calls_external)
        references[rel[p]] = refs

    header_types: dict[str, str] = {}
    for h in h_files:
        hname = str(h.relative_to(c_root))
        for t in _header_declared_types(h.read_text()):
            header_types.setdefault(t, hname)

    type_refs: dict[str, set[str]] = {t: set() for t in header_types}
    for p in c_files:
        used = _type_identifiers(p.read_text())
        for t in header_types:
            if t in used:
                type_refs[t].add(rel[p])

    # file graph: A -> B when A references a function B defines
    file_graph: dict[str, list[str]] = {f: [] for f in references}
    for f, refs in references.items():
        deps = {defines[n] for n in refs if n in defines and defines[n] != f}
        file_graph[f] = sorted(deps)

    # SCC condensation. With edges pointing AT dependencies (A -> B when A
    # calls into B), Tarjan emits sink components first — i.e. dependencies
    # before their callers, which IS the translation order (leaves first).
    groups = tarjan_scc(file_graph)

    return ProjectIndex(c_root=c_root, files=sorted(references), defines=defines,
                        references=references, header_types=header_types,
                        type_refs=type_refs, file_graph=file_graph,
                        groups=groups)


def main() -> None:
    root = Path(sys.argv[1])
    idx = build_index(root)
    print(f"files: {len(idx.files)}")
    print("\ntranslation order (SCC groups, leaves first):")
    for i, g in enumerate(idx.groups):
        tag = "  CO-TRANSLATE" if len(g) > 1 else ""
        print(f"  {i}: {', '.join(g)}{tag}")
    print("\nfile graph:")
    for f, deps in sorted(idx.file_graph.items()):
        print(f"  {f} -> {', '.join(deps) or '(leaf)'}")
    print(f"\nshared types ({len(idx.shared_types)}):")
    for t, h in sorted(idx.shared_types.items()):
        print(f"  {t} ({h}) used by: {', '.join(sorted(idx.type_refs[t]))}")
    sib = 0
    for f, refs in idx.references.items():
        for n in refs:
            if idx.dep_class(f, n) == "SIBLING":
                sib += 1
    print(f"\nsibling call edges: {sib}")


if __name__ == "__main__":
    main()
