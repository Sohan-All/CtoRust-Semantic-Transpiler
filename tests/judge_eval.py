"""Targeted eval of the C-mention judge against known cases.

FALSE_POSITIVES: texts the old detection-style judge wrongly flagged on
tree-cache.c / strbuf.c runs — all must be ALLOWED.
TRUE_POSITIVES: genuine C-isms — all must be FLAGGED.

Usage: python3 tests/judge_eval.py [runs_per_case]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from llm import LLM
from lock_check import LockChecker
from state import Explanation

# wrongly flagged by the detection-style judge (verbatim from state.jsonl)
FALSE_POSITIVES = [
    "the number of bytes processed equals the count of bytes up to and including the first zero-valued byte",
    "Name is followed by a zero terminator",
    "The node's internal buffer stores the name terminated by a zero byte",
    "allocates a zeroed byte sequence",
    "cache.children references a zeroed byte sequence",
    "memory allocation failed during the serialization process",
    "the entire tree is deallocated",
    "Release all storage associated with the buffer and reset its state to uninitialized",
    "growable string buffer",
    "Buffer cursor is advanced through consumption and passed by reference to callers",
    "the child cache record is left allocated and partially initialized in the parent's children array",
    "When the children array grows, any prior numerical indexing scheme for accessing specific elements becomes invalid",
]

# genuine C-isms the judge must catch
TRUE_POSITIVES = [
    "copies the field content using memcpy into the output buffer",
    "advances the pointer past the comma before scanning the next field",
    "calling this function twice on the same sequence is undefined behavior",
    "INITIAL_CAP is a preprocessor constant defined with #define",
    "the buffer is a char* obtained from malloc",
    "reads until the FILE* handle reports end of file",
]


async def main(runs: int) -> None:
    cfg = Config()
    cfg.lock_c_mention_enabled = True
    cfg.lock_regex_enabled = False       # isolate the LLM judge
    cfg.lock_round_trip_enabled = False
    llm = LLM(cfg)
    checker = LockChecker(cfg, llm)

    async def judge(text: str) -> bool:
        """True = flagged."""
        exp = Explanation(id="t", ranges=[[1, 1]], text=text)
        res = await checker.check(exp)
        return not res.passed

    fp = tp = 0
    for label, cases, want_flagged in [("FALSE-POSITIVE set (want ALLOWED)", FALSE_POSITIVES, False),
                                       ("TRUE-POSITIVE set (want FLAGGED)", TRUE_POSITIVES, True)]:
        print(f"\n=== {label} ===")
        results = await asyncio.gather(*(judge(c) for c in cases for _ in range(runs)))
        for i, case in enumerate(cases):
            votes = results[i * runs:(i + 1) * runs]
            wrong = sum(1 for v in votes if v != want_flagged)
            status = "ok  " if wrong == 0 else f"BAD {wrong}/{runs}"
            print(f"  [{status}] {case[:90]}")
            if want_flagged:
                tp += sum(1 for v in votes if v)
            else:
                fp += sum(1 for v in votes if v)

    n_fp = len(FALSE_POSITIVES) * runs
    n_tp = len(TRUE_POSITIVES) * runs
    print(f"\nfalse-positive rate: {fp}/{n_fp}   true-positive recall: {tp}/{n_tp}")
    print(f"calls: {llm.calls}, tokens: {llm.input_tokens}+{llm.output_tokens}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
