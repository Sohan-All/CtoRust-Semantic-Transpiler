"""Tests for C function-pointer PARAMETER detection (bug class 7).

Why this exists. `calls_external` is "callees not defined in this file", which
is right for a sibling or a libc call and WRONG for a callback parameter: the
body calls `pred(item)`, `pred` is a parameter, and it lands in external deps.
Stage T then stubs it in `<stem>_deps` as one global function — unfillable,
because callers pass a different predicate at each site — and `sibling_deps`
has been observed INVENTING a body for one (`cp` -> `Ok(item.clone())`), which
compiles and is silently wrong. That voided every `cc_array` run ever recorded.

Per the project convention, the detector is asserted against known-good AND
known-bad: it must fire on real callback parameters and stay silent on ordinary
pointer parameters, arrays, and libc calls.

Run: PYTHONPATH=. venv/bin/python test_callback_params.py
"""

from pathlib import Path

from chunker import chunk, fn_pointer_typedefs, _callback_params

CORPUS = Path("/nobackup2/alleshwaram/CtoRust/Test-Corpus/Public-Tests/B03_organic")

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))
        _failures.append(label)


def test_signature_forms() -> None:
    """The literal `(*name)(...)` form, and the shapes that must NOT match."""
    cases_hit = [
        ("void f(int (*cmp)(const void *, const void *))", {"cmp"}),
        ("bool f(CC_Array *a, bool (*pred)(const void *))", {"pred"}),
        # two callbacks in one signature
        ("void f(int (*cmp)(void *), void (*cb)(void *))", {"cmp", "cb"}),
        # pointer-returning callback
        ("void f(char *(*dup)(const char *))", {"dup"}),
        # callback beside ordinary params
        ("int f(Array *a, size_t n, int (*cmp)(const void *, const void *), void *ctx)",
         {"cmp"}),
    ]
    for sig, want in cases_hit:
        got = set(_callback_params(sig))
        check(f"detects {sorted(want)} in `{sig[:46]}...`", got == want, f"got {sorted(got)}")

    cases_miss = [
        # ordinary pointer params
        ("void f(char *s, int n)", "plain pointer"),
        ("void f(const void **items, size_t n)", "pointer-to-pointer"),
        # array param
        ("void f(int a[], int n)", "array"),
        # no params
        ("int main(void)", "void params"),
        # UNNAMED fn-pointer param: cannot be called from the body, so it can
        # never be misread as an external call — silence here is correct
        ("void f(void (*)(void))", "unnamed callback"),
        # a call in the RETURN type, not a param
        ("void (*get_handler(int n))(void)", "fn-ptr return type"),
    ]
    for sig, why in cases_miss:
        got = set(_callback_params(sig))
        check(f"silent on {why}", got == set(), f"got {sorted(got)}")


def test_typedef_form() -> None:
    """`ArrayListCompareFunc cmp` — the same parameter through a typedef.

    All five corpus projects that do this keep the typedef in a HEADER, so the
    names have to be supplied; without them the detector is blind to the form
    `array_list`, `binary_heap` and `binomial_heap` actually use.
    """
    header = "typedef int (*ArrayListCompareFunc)(ArrayListValue a, ArrayListValue b);"
    names = fn_pointer_typedefs(header)
    check("typedef name is extracted", names == {"ArrayListCompareFunc"}, str(names))

    sig = "static void arraylist_sort_internal(ArrayListValue *list, ArrayListCompareFunc compare_func)"
    check("typedef'd param is MISSED without the names",
          set(_callback_params(sig)) == set())
    check("typedef'd param is FOUND with them",
          set(_callback_params(sig, frozenset(names))) == {"compare_func"},
          str(set(_callback_params(sig, frozenset(names)))))

    # a plain type must not be captured just because a typedef set was supplied
    plain = "void f(ArrayListValue v, int n)"
    check("  a non-callback param of a known project is untouched",
          set(_callback_params(plain, frozenset(names))) == set())


def test_removed_from_calls_external() -> None:
    """The point of the whole exercise: the name must leave `calls_external`
    while genuine external calls stay."""
    src = """
static bool arraylist_filter(int *list, int n, bool (*pred)(int)) {
    int *copy = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) { if (pred(list[i])) copy[i] = list[i]; }
    memmove(list, copy, n);
    free(copy);
    return true;
}
"""
    g = chunk(src)
    b = [x for x in g.blocks if x.function == "arraylist_filter"][0]
    check("the callback is recorded", set(b.callback_params) == {"pred"},
          str(b.callback_params))
    check("  and is GONE from calls_external", "pred" not in b.calls_external,
          str(b.calls_external))
    check("  while malloc/free/memmove survive",
          {"malloc", "free", "memmove"} <= set(b.calls_external),
          str(b.calls_external))


def test_corpus() -> None:
    """Known-good/known-bad at corpus scale.

    `cc_array` is the crate this bug was found on: 8 callback-taking functions,
    none of which may leak. The 11 projects with no function-pointer parameters
    must stay completely silent — a detector that fires everywhere is useless
    as a gate.
    """
    if not CORPUS.is_dir():
        check("corpus unavailable — skipped", True)
        return
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_project import project_fn_ptr_types

    fired, total, leaks, libc_hits = 0, 0, [], []
    for proj in sorted(p for p in CORPUS.iterdir() if (p / "test_case" / "src").is_dir()):
        total += 1
        tds = project_fn_ptr_types(proj / "test_case")
        n = 0
        for c in (proj / "test_case" / "src").glob("*.c"):
            g = chunk(c.read_text(errors="replace"), fn_ptr_types=tds)
            for b in g.blocks:
                n += len(b.callback_params)
                leaks += [(proj.name, x) for x in b.callback_params
                          if x in b.calls_external]
                libc_hits += [(proj.name, x) for x in b.callback_params
                              if x in ("malloc", "free", "printf", "strlen",
                                       "memmove", "qsort", "memcpy")]
        if n:
            fired += 1
    check(f"corpus scanned ({total} projects)", total >= 25, f"{total}")
    check("a callback NEVER remains in calls_external", leaks == [], str(leaks[:5]))
    check("no libc function is mistaken for a callback", libc_hits == [],
          str(libc_hits[:5]))
    check("the detector is selective, not universal", 0 < fired < total,
          f"fired on {fired}/{total}")

    # the crate the bug was found on
    tds = project_fn_ptr_types(CORPUS / "cc_array" / "test_case")
    src = (CORPUS / "cc_array" / "test_case" / "src" / "cc_array.c").read_text()
    names = {n for b in chunk(src, fn_ptr_types=tds).blocks for n in b.callback_params}
    check("  cc_array's four callback names are all found",
          {"pred", "cmp", "cb", "cp"} <= names, str(sorted(names)))


def test_stubbed_callbacks_gate() -> None:
    """The gate that refuses a `_deps` stub for a callback parameter.

    Needed as well as the prompt rule because prompt compliance is stochastic
    at temperature 1.0 (the EXIT STATUS block landed in 4 of 8 arms), and this
    mistake does not stay an honest stub: `sibling_deps` walks every `*_deps`
    stub and asks the model for a body, and for `cc_array`'s `cp` it produced
    `Ok(item.clone())` — type-correct, compiles, flagged by nothing, silently
    wrong in a SCORED crate.
    """
    from rustgen.common import stubbed_callbacks, illegal_stubs

    bad = ("pub mod cc_array_deps {\n"
           "    pub fn pred<T>(item: &T) -> bool { todo!() }\n"
           "    pub fn cp<T: Clone>(i: &T) -> Result<T, E> { Ok(i.clone()) }\n"
           "    pub fn git_fetch(x: i32) -> i32 { todo!() }\n}")
    names = frozenset({"pred", "cmp", "cb", "cp", "fn"})

    hit = stubbed_callbacks(bad, names)
    check("a callback deps stub is refused", "pred" in hit, hit)
    check("  and so is one with a FABRICATED body, not just todo!()",
          "cp" in hit, hit)
    check("  a genuinely external stub is left alone", "git_fetch" not in hit, hit)

    ok = "pub mod cc_array_deps {\n    pub fn git_fetch(x: i32) -> i32 { todo!() }\n}"
    check("a clean deps module passes", stubbed_callbacks(ok, names) == "")
    check("no callback names = nothing to check",
          stubbed_callbacks(bad, frozenset()) == "")

    # the reason this needs its own checker: illegal_stubs PERMITS these by
    # design, because a `todo!()` inside a *_deps module is the legal form
    check("  illegal_stubs alone cannot see it (hence a separate gate)",
          illegal_stubs(bad) == "", illegal_stubs(bad))


def main() -> int:
    test_signature_forms()
    test_typedef_form()
    test_removed_from_calls_external()
    test_stubbed_callbacks_gate()
    test_corpus()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
