"""Ground-truth tests for exit_status.py — run directly:

    PYTHONPATH=. venv/bin/python test_exit_status.py

The negative cases matter as much as the positive ones. `handle_script_command`
IS called by a function in the propagation set, but its value is consumed by an
`if` and never returned, so it must NOT be a member. If this analysis ever
regresses to walking the call graph instead of return-flow, that assertion and
the four `handle_*` ones below are what catch it.
"""

from __future__ import annotations

import sys

from exit_status import analyze_project, interesting
from testpaths import CORPUS, skip_unless

_failures: list[str] = []


def check(label: str, cond: bool) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        _failures.append(label)


def load(project: str) -> dict[str, str]:
    src = CORPUS / project / "test_case" / "src"
    return {p.name: p.read_text(errors="replace") for p in sorted(src.glob("*.c"))}


def test_double_linked_list() -> None:
    """The project the analysis was written against: C's run_script returns a
    distinct 2 that the generated Rust flattened to 1."""
    print("\n=== double_linked_list: membership and codes ===")
    res = analyze_project(load("double_linked_list"))
    names = {n for _, n in res}

    check("membership == {main, cli_run, run_script}",
          names == {"main", "cli_run", "run_script"})
    check("handle_script_command NOT a member (value consumed by `if`)",
          "handle_script_command" not in names)
    for h in ("handle_append", "handle_replace", "handle_undo", "handle_redo"):
        check(f"{h} NOT a member (tail-called under handle_script_command)",
              h not in names)

    check("run_script codes == [0, 2]",
          sorted(res[("cli.c", "run_script")].codes) == [0, 2])
    check("cli_run codes == [0, 1, 2] (inherits run_script's 2)",
          sorted(res[("cli.c", "cli_run")].codes) == [0, 1, 2])
    check("main codes == [0, 1, 2]",
          sorted(res[("main.c", "main")].codes) == [0, 1, 2])
    check("run_script path == run_script -> cli_run -> main",
          res[("cli.c", "run_script")].path == ["run_script", "cli_run", "main"])
    check("all three emit a block", all(interesting(v) for v in res.values()))
    check("all three report an exhaustive code set",
          all(v.exhaustive for v in res.values()))


def test_return_forms() -> None:
    """One synthetic case per return form the analysis must classify."""
    print("\n=== return forms ===")
    cases = {
        # label:            (source,                                          codes,   emit, exhaustive)
        "direct literal":   ("int h(void){return 2;}\nint main(void){return h();}",
                             [2], True, True),
        "negative literal": ("int h(void){return -1;}\nint main(void){return h();}",
                             [-1], True, True),
        "via local var":    ("int h(void){int rc=0; rc=7; return rc;}\nint main(void){return h();}",
                             [0, 7], True, True),
        "only 0/1":         ("int h(void){return 1;}\nint main(void){return h();}",
                             [1], False, True),
        "computed ternary": ("int h(int n){return n>3?2:0;}\nint main(void){return h(5);}",
                             [], True, False),
        "macro constant":   ("int h(void){return EXIT_BAD;}\nint main(void){return h();}",
                             [], True, False),
        "void return":      ("void h(void){return;}\nint main(void){h(); return 0;}",
                             None, None, None),   # h must not be a member
    }
    for label, (src, codes, emit, exh) in cases.items():
        res = analyze_project({"a.c": src})
        got = {n: v for (_f, n), v in res.items()}
        if codes is None:
            check(f"{label}: h is not a member", "h" not in got)
            continue
        v = got.get("h")
        if v is None:
            check(f"{label}: h is a member", False)
            continue
        check(f"{label}: codes == {codes}", sorted(v.codes) == codes)
        check(f"{label}: emit == {emit}", interesting(v) is emit)
        check(f"{label}: exhaustive == {exh}", v.exhaustive is exh)


def test_no_main() -> None:
    """A library with no `main` has no exit status, so nothing may be emitted —
    half the corpus is such a project and a false positive there is pure noise."""
    print("\n=== library projects (no main) ===")
    for proj in ("double_linked_list_lib", "binary_heap_lib", "cc_hashset_lib"):
        res = analyze_project(load(proj))
        check(f"{proj}: no functions reach an exit status", res == {})


def test_corpus_noise() -> None:
    """The emission rate is the guard against this becoming prompt noise."""
    print("\n=== corpus-wide noise ===")
    total = members = emitted = 0
    import exit_status as es
    from tree_sitter import Parser
    parser = Parser(es.C_LANGUAGE)
    for proj in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        if not (proj / "test_case" / "src").is_dir():
            continue
        sources = load(proj.name)
        if not sources:
            continue
        for text in sources.values():
            for n in es._walk(parser.parse(text.encode()).root_node):
                if n.type == "function_definition":
                    total += 1
        res = analyze_project(sources)
        members += len(res)
        emitted += sum(1 for v in res.values() if interesting(v))
    rate = 100 * emitted / max(total, 1)
    print(f"  {total} C functions, {members} reach an exit status, "
          f"{emitted} emit a block ({rate:.1f}%)")
    check("emission rate stays under 5% of all C functions", rate < 5.0)
    check("every emitting project is a CLI (emitted > 0 somewhere)", emitted > 0)


def main() -> int:
    # Every test here reads the corpus through load(); without it the
    # ground-truth assertions raise KeyError on a project that was never
    # loaded, which reads as an analysis bug rather than a missing checkout.
    if skip_unless(CORPUS, "corpus"):
        print("\nALL PASS (skipped: no corpus)")
        return 0
    test_double_linked_list()
    test_return_forms()
    test_no_main()
    test_corpus_noise()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
