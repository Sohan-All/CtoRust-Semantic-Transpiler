"""Coverage validator — mechanical, no LLM.

Checks that a set of line ranges covers a file with no gaps and no overlaps.
Used to validate whole-file-direct model output and to assert the terminal
invariant of the diffusion strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoverageResult:
    ok: bool
    gaps: list[tuple[int, int]] = field(default_factory=list)
    overlaps: list[tuple[int, int]] = field(default_factory=list)
    out_of_bounds: list[tuple[int, int]] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return "coverage OK"
        parts = []
        if self.gaps:
            parts.append("uncovered lines: " + ", ".join(f"{s}-{e}" for s, e in self.gaps))
        if self.overlaps:
            parts.append("overlapping lines: " + ", ".join(f"{s}-{e}" for s, e in self.overlaps))
        if self.out_of_bounds:
            parts.append("out-of-bounds ranges: " + ", ".join(f"{s}-{e}" for s, e in self.out_of_bounds))
        return "; ".join(parts)


def check_coverage(
    ranges: list[list[int]] | list[tuple[int, int]],
    n_lines: int,
    ignorable: set[int] | None = None,
) -> CoverageResult:
    """ranges: [(start, end), ...] 1-indexed inclusive, possibly from many units.

    `ignorable` are line numbers allowed to be uncovered (blank lines) — used
    when validating model output in strategy B, where demanding coverage of
    blank separator lines would cause pointless retries.
    """
    ignorable = ignorable or set()
    counts = [0] * (n_lines + 2)  # 1-indexed
    out_of_bounds: list[tuple[int, int]] = []
    for start, end in ranges:
        if start < 1 or end > n_lines or start > end:
            out_of_bounds.append((start, end))
            continue
        for i in range(start, end + 1):
            counts[i] += 1

    gaps = _runs([i for i in range(1, n_lines + 1) if counts[i] == 0 and i not in ignorable])
    overlaps = _runs([i for i in range(1, n_lines + 1) if counts[i] > 1])
    ok = not gaps and not overlaps and not out_of_bounds
    return CoverageResult(ok=ok, gaps=gaps, overlaps=overlaps, out_of_bounds=out_of_bounds)


def blank_lines(source: str) -> set[int]:
    return {i + 1 for i, line in enumerate(source.split("\n")) if not line.strip()}


def _runs(sorted_lines: list[int]) -> list[tuple[int, int]]:
    """Compress a sorted list of line numbers into inclusive runs."""
    runs: list[tuple[int, int]] = []
    for n in sorted_lines:
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs
