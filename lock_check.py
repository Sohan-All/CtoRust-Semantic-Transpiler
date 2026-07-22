"""Lock-check stack: regex blocklist -> C-mention check -> round-trip probe.

Runs over the WHOLE unit (description + invariants). Stages run cheapest-first;
a unit locks only if all enabled stages pass. Also provides the invariant
rephrase call (behavioral restatement of a C-phrased invariant).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import Config
from llm import LLM
from state import Explanation

# --- Stage 1: regex blocklist -------------------------------------------------
# Curated C-isms. Mechanism words fail; behavioral phrasings pass. Grows as we
# observe escapes. Word-boundary matched, case-insensitive.
BLOCKLIST_TERMS = [
    r"pointer(?:s)?", r"malloc", r"calloc", r"realloc", r"\bfree(?:s|d|ing)?\b",
    r"#define", r"#include", r"preprocessor", r"macro(?:s)?",
    r"null[- ]termin\w+", r"\bchar\s*\*", r"void\s*\*", r"\bstruct\b",
    r"\berrno\b", r"segfault", r"memcpy", r"memset", r"strcpy", r"strlen",
    r"header file", r"\.h\b file", r"typedef",
    r"dereferenc\w+", r"address[- ]of", r"\bC\s+string\b",
    r"stack[- ]allocat\w+", r"heap[- ]allocat\w+",
    r"undefined behavi\w+", r"uninitialized memory",
    r"\bsizeof\b", r"\w+->\w+",   # quoted C expressions / member access
]
# case-sensitive terms: uppercase FILE is the C type; lowercase "file" is prose
BLOCKLIST_CASE_SENSITIVE = [r"\bFILE\b"]
BLOCKLIST = [re.compile(t, re.IGNORECASE) for t in BLOCKLIST_TERMS] + [
    re.compile(t) for t in BLOCKLIST_CASE_SENSITIVE
]


@dataclass
class LockResult:
    passed: bool
    stage: str = ""          # stage that failed ("" if passed)
    detail: str = ""
    offending_invariant: str | None = None  # set when the failure localizes to an invariant
    offending_description: bool = False      # set when the failure localizes to the description
    questions: list[str] = field(default_factory=list)  # round-trip probe's unanswered questions


def regex_scan(text: str) -> str | None:
    """Return the first blocklisted term matched, or None."""
    for pat in BLOCKLIST:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


# --- Stage 2 & 3 prompts -------------------------------------------------------

C_MENTION_PROMPT = """\
You are checking a specification for language portability. Below are numbered
items from the spec. Classify EACH item as ALLOWED or FLAGGED.

FLAGGED only when the item matches one of these categories:
1. It NAMES a C library function, type, or keyword: malloc/calloc/realloc/free,
   memcpy/memset/str* functions, char*/void*/FILE*, errno, struct, typedef,
   #define.
2. It describes manipulating raw pointers or memory addresses as such:
   "advance the pointer", "dereference", "the address of".
3. It relies on undefined behavior, uninitialized memory, or aliasing rules.
4. It references the C preprocessor or header files.

Anything else is ALLOWED — including everything about memory, bytes, and data
formats phrased behaviorally. Worked examples:

- "the name is terminated by a zero byte"            -> ALLOWED (data format)
- "capacity doubles when the sequence is full"        -> ALLOWED (growth policy)
- "allocates a zeroed record for each child"          -> ALLOWED (abstract allocation)
- "releases the resources owned by all descendants"   -> ALLOWED (abstract release)
- "a position advances through the input as bytes are consumed" -> ALLOWED (cursor)
- "returns -1 if storage cannot be obtained"          -> ALLOWED (error sentinel)
- "the entry count is a big-endian 32-bit integer"    -> ALLOWED (data format)
- "copies the bytes using memcpy"                     -> FLAGGED (category 1)
- "advances the pointer past the slash"               -> FLAGGED (category 2)
- "calling this twice is undefined behavior"          -> FLAGGED (category 3)
- "the constant is defined in the header file"        -> FLAGGED (category 4)

Reply with ONLY a JSON object, one entry per item, in order:
{{"items": [{{"n": 1, "verdict": "ALLOWED"}}, {{"n": 2, "verdict": "FLAGGED", "category": 2, "quote": "<offending words>"}}, ...]}}

ITEMS:
{items}
"""

ROUND_TRIP_PROMPT = """\
You are asked to implement the following behavior in Rust. You have NOT seen
the original program, only this description and these invariants.
{context}
You are free to make all idiomatic implementation decisions yourself: type
choices (String vs Vec<u8>, usize widths), error representation (Result vs
sentinel), naming, standard-library usage, and how to structure the code.
List the questions you would need answered, and label each one:
- "behavior": missing or ambiguous behavior — two reasonable programmers would
  build observably different things without the answer
- "format": missing detail of an input/output data format
- "idiom": a language/idiom/style choice you could actually decide yourself

If the behavior is fully determined, reply with an empty list.

Reply with ONLY a JSON object:
{{"questions": [{{"question": "...", "kind": "behavior" | "format" | "idiom"}}, ...]}}

DESCRIPTION:
{description}

INVARIANTS:
{invariants}
"""

CONTEXT_BLOCK = """
The program's OTHER components are specified separately and will be available
to you; do not ask questions their summaries below already answer:
{siblings}
"""

DEPS_BLOCK = """
The implementation may call these externally-specified functions, whose exact
behavior is documented elsewhere — do not ask about their internals, formats,
or return conventions: {deps}
"""

QUESTION_FILTER_PROMPT = """\
An implementer reviewing the specification below asked the questions listed
after it. Classify EACH question:

- "ANSWERED": the specification text (description or invariants) already
  contains the answer, even partially or implicitly.
- "CHOICE": it is an implementation decision the implementer makes themselves —
  type choices, error representation (Result vs codes), naming, code structure,
  standard-library selection, and how to represent data internally.
- "OPEN": genuinely missing behavioral or format information — two reasonable
  implementations would observably differ without the answer.

SPECIFICATION:
{description}

INVARIANTS:
{invariants}

QUESTIONS:
{questions}

Reply with ONLY a JSON object, one verdict per question, in order:
{{"verdicts": ["ANSWERED" | "CHOICE" | "OPEN", ...]}}
"""

REPAIR_PROMPT = """\
A specification of the following C code was judged insufficient to reimplement
from: a reviewer (who could not see the code) asked the questions listed below.
Answer each question by consulting the code, and express each answer as a new
invariant: observable behavior only, no C vocabulary (no pointer/malloc/free/
struct/null-terminated/macro or C function names — say "terminated by a zero
byte", "releases the resources it owns", "a position within the input"). Never
quote C code or identifiers; refer to inputs by role. If a question cannot be
answered from this code (it concerns an external component), skip it.

CODE (lines {start}-{end}):
```c
{code}
```

CURRENT DESCRIPTION:
{description}

REVIEWER QUESTIONS:
{questions}

Reply with ONLY a JSON object: {{"invariants": ["...", ...]}}
"""

REPHRASE_PROMPT = """\
The following fact about a program's behavior is phrased in terms of C
mechanisms. Restate it purely as observable behavior, without reference to C
constructs, memory management functions, pointers, or the C standard library.
Preserve the exact semantic content — do not weaken or generalize it. If the
fact fundamentally cannot be expressed without C's memory model, reply with the
single word IMPOSSIBLE.

Useful substitutions: memory release -> "releases the resources it owns";
malloc/realloc failure -> "if storage cannot be obtained"; null-terminated ->
"terminated by a zero byte"; pointer into a buffer -> "a position within the
input".

FACT: {invariant}

Reply with ONLY the restated fact (one or two sentences) or IMPOSSIBLE.
"""

REPHRASE_DESC_PROMPT = """\
The following description of program behavior mentions C-specific mechanisms.
Rewrite it so it describes only observable behavior, with no reference to
pointers, manual memory management functions, the preprocessor, or the C
standard library (write "a growable byte sequence", not "a malloc'd char
buffer"; "terminated by a zero byte", not "null-terminated"). Keep it 1-5
sentences and preserve the exact semantic content — do not weaken it. Useful
substitutions: memory release -> "releases the resources it owns"; freeing
recursively -> "releases all descendants' resources"; allocation failure ->
"if storage cannot be obtained". If the behavior fundamentally cannot be
described without C's memory model, reply with the single word IMPOSSIBLE.

DESCRIPTION: {description}

Reply with ONLY the rewritten description or IMPOSSIBLE.
"""


def _fmt_invariants(invariants: list[str]) -> str:
    return "\n".join(f"- {inv}" for inv in invariants) if invariants else "(none)"


class LockChecker:
    def __init__(self, cfg: Config, llm: LLM):
        self.cfg = cfg
        self.llm = llm

    async def check(self, exp: Explanation, siblings: list[str] | None = None) -> LockResult:
        """`siblings` are one-line descriptions of the program's other units,
        given to the round-trip probe as context (at Rust-spec time all MTUs
        exist together, so isolation-only judging is too strict)."""
        # Stage 1: regex blocklist, description then each invariant
        if self.cfg.lock_regex_enabled:
            hit = regex_scan(exp.text)
            if hit:
                return LockResult(False, "regex", f"description matched blocklist term '{hit}'",
                                  offending_description=True)
            for inv in exp.invariants:
                hit = regex_scan(inv)
                if hit:
                    return LockResult(False, "regex",
                                      f"invariant matched blocklist term '{hit}'",
                                      offending_invariant=inv)

        # Stage 2: C-mention check — one item per call (the configuration that
        # measured 0 FP / 100% recall in tests/judge_eval.py; batched lists made
        # the judge flag its own ALLOWED examples). A FLAGGED verdict must be
        # confirmed by one independent re-judgment before it counts.
        if self.cfg.lock_c_mention_enabled:
            import asyncio as _asyncio

            items = [exp.text] + list(exp.invariants)

            async def judge(text: str) -> dict | None:
                resp = await self.llm.ask_json(C_MENTION_PROMPT.format(items=f"1. {text}"))
                entries = resp.get("items", []) if isinstance(resp, dict) else []
                f = entries[0] if entries and isinstance(entries[0], dict) else None
                return f if f and f.get("verdict") == "FLAGGED" else None

            flags = await _asyncio.gather(*(judge(t) for t in items))
            for n, (text, f) in enumerate(zip(items, flags), start=1):
                if f is None:
                    continue
                confirm = await judge(text)  # independent second opinion
                if confirm is None:
                    continue
                return LockResult(False, "c_mention",
                                  f"item {n} [{f.get('category', '?')}]: "
                                  f"{f.get('quote', text[:80])}",
                                  offending_invariant=text if n > 1 else None,
                                  offending_description=(n == 1))

        # Stage 3: round-trip probe. "idiom" questions are the probe's own
        # decisions to make and don't count against the unit; a pile of
        # behavior/format questions means the description is underspecified.
        if self.cfg.lock_round_trip_enabled:
            ctx = ""
            if siblings:
                ctx = CONTEXT_BLOCK.format(
                    siblings="\n".join(f"- {s}" for s in siblings))
            if exp.external_deps:
                ctx += DEPS_BLOCK.format(deps=", ".join(exp.external_deps))
            resp = await self.llm.ask_json(ROUND_TRIP_PROMPT.format(
                context=ctx, description=exp.text,
                invariants=_fmt_invariants(exp.invariants)))
            raw = resp.get("questions", []) if isinstance(resp, dict) else []
            substantive = [q.get("question", "") for q in raw
                           if isinstance(q, dict) and q.get("kind") in ("behavior", "format")]
            if len(substantive) > self.cfg.lock_round_trip_max_open:
                # arbitration: the probe over-asks (mislabels idiom choices,
                # re-asks what invariants already answer). Only OPEN questions
                # count against the unit. (Skipped when substantive can't exceed
                # the threshold — it then can't fail the gate below regardless.)
                fresp = await self.llm.ask_json(QUESTION_FILTER_PROMPT.format(
                    description=exp.text,
                    invariants=_fmt_invariants(exp.invariants),
                    questions="\n".join(f"{i+1}. {q}" for i, q in enumerate(substantive))))
                verdicts = fresp.get("verdicts", []) if isinstance(fresp, dict) else []
                if len(verdicts) == len(substantive):
                    open_qs = [q for q, v in zip(substantive, verdicts)
                               if str(v).upper() == "OPEN"]
                else:  # malformed filter reply: count everything (conservative)
                    open_qs = substantive
                if len(open_qs) > self.cfg.lock_round_trip_max_open:
                    return LockResult(False, "round_trip",
                                      f"{len(open_qs)} OPEN questions (of {len(substantive)} asked); "
                                      "description underspecified: " + "; ".join(open_qs[:3]),
                                      questions=open_qs)

        return LockResult(True)

    async def sanitize(self, exp: Explanation) -> bool:
        """Regex-scan the description and ALL invariants; batch-rephrase every
        offender in parallel. Runs before lock attempts so obvious C phrasing
        never costs an attempt. Returns True if anything changed. Respects the
        per-item rephrase budget; items that can't be laundered are left in
        place for the lock check to fail honestly."""
        import asyncio

        changed = False
        jobs: list[tuple[str, str]] = []  # (kind, original)
        if regex_scan(exp.text) and exp.invariant_rephrases.get("__description__", 0) < self.cfg.max_invariant_rephrases:
            jobs.append(("desc", exp.text))
        for inv in exp.invariants:
            if regex_scan(inv) and exp.invariant_rephrases.get(inv, 0) < self.cfg.max_invariant_rephrases:
                jobs.append(("inv", inv))
        if not jobs:
            return False

        async def do(kind: str, original: str):
            if kind == "desc":
                return kind, original, await self.rephrase_description(original)
            return kind, original, await self.rephrase_invariant(original)

        for kind, original, replacement in await asyncio.gather(*(do(k, o) for k, o in jobs)):
            key = "__description__" if kind == "desc" else original
            exp.invariant_rephrases[key] = exp.invariant_rephrases.get(key, 0) + 1
            if replacement is None:
                continue
            if kind == "desc":
                exp.text = replacement
            else:
                exp.invariants = [replacement if i == original else i for i in exp.invariants]
            changed = True
        return changed

    async def _rephrase(self, prompt_template: str, key: str, original: str,
                        max_tokens: int) -> str | None:
        """Shared rephrase driver: names the banned term explicitly, and retries
        once quoting any new violation in the model's own output."""
        term = regex_scan(original)
        banned = (f"\nThe term '{term}' is banned — your restatement must not "
                  "contain it or any variant of it.\n") if term else ""
        prompt = prompt_template.format(**{key: original}) + banned
        for _ in range(2):
            text = (await self.llm.ask(prompt, max_tokens=max_tokens)).strip()
            if not text or text.upper().startswith("IMPOSSIBLE"):
                return None
            hit = regex_scan(text)
            if hit is None:
                return text
            prompt = (prompt_template.format(**{key: original})
                      + f"\nYour previous restatement used the banned term '{hit}'. "
                        "Produce a version that avoids it (e.g. 'releases the "
                        "resources it owns' instead of naming memory operations).\n")
        return None

    async def rephrase_invariant(self, invariant: str) -> str | None:
        """Behavioral restatement of a C-phrased invariant. None if impossible."""
        return await self._rephrase(REPHRASE_PROMPT, "invariant", invariant, 300)

    async def repair(self, exp: Explanation, source_lines: list[str],
                     questions: list[str]) -> list[str]:
        """Answer the round-trip probe's questions from the unit's own source,
        returning new invariants to append. This is the underspecification
        repair path: the probe's questions are a to-do list of what's missing,
        and provenance gives us the code to answer them from."""
        code = "\n".join(
            "\n".join(source_lines[s - 1:e]) for s, e in exp.ranges)
        start = exp.ranges[0][0]
        end = exp.ranges[-1][1]
        resp = await self.llm.ask_json(REPAIR_PROMPT.format(
            start=start, end=end, code=code, description=exp.text,
            questions="\n".join(f"- {q}" for q in questions)),
            max_tokens=2000)
        raw = resp.get("invariants", []) if isinstance(resp, dict) else []
        new = [str(i).strip() for i in raw]
        return [i for i in new if i and i not in exp.invariants]

    async def rephrase_description(self, description: str) -> str | None:
        """Behavioral rewrite of a C-phrased description. None if impossible."""
        return await self._rephrase(REPHRASE_DESC_PROMPT, "description", description, 500)
