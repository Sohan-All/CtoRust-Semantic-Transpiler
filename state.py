"""JSONL state store (source of truth) + explanations.md renderer (view only).

Every record is one JSON object per line with a "type" field:
  config      — snapshot of the Config for this run
  seed_graph  — call edges + topological order from the chunker
  explanation — the unit records (see SPEC.md for the schema)
  event       — run events (pass boundaries, retries, failures, usage totals)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

OPEN = "open"
LOCKED = "locked"
IRREDUCIBLE = "irreducible"


@dataclass
class Explanation:
    id: str
    ranges: list[list[int]]              # [[start, end], ...] 1-indexed, inclusive
    text: str
    invariants: list[str] = field(default_factory=list)
    status: str = OPEN
    pass_num: int = 0
    parent_ids: list[str] = field(default_factory=list)
    external_deps: list[str] = field(default_factory=list)
    model: str = ""
    strategy: str = ""
    lock_attempts: int = 0
    changed_since_lock_attempt: bool = True
    invariant_rephrases: dict[str, int] = field(default_factory=dict)  # invariant text -> count
    lock_failures: list[dict] = field(default_factory=list)

    def total_lines(self) -> int:
        return sum(end - start + 1 for start, end in self.ranges)

    def sort_key(self) -> int:
        return min(start for start, _ in self.ranges)

    def to_record(self) -> dict:
        d = asdict(self)
        d["pass"] = d.pop("pass_num")
        return {"type": "explanation", **d}


class Store:
    """Append-only JSONL writer plus the in-memory working set.

    The JSONL file is a log: explanations are re-appended each time they change,
    and the latest record per id wins on load.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "state.jsonl"
        self.explanations: dict[str, Explanation] = {}
        self._counter = 0

    @classmethod
    def load(cls, out_dir: Path) -> "Store":
        """Rebuild the working set from an existing state.jsonl — the file is a
        log, so the latest record per explanation id wins. Also restores the id
        counter and stashes non-explanation records on .records for callers
        that need run metadata (config, events, seed_graph)."""
        store = cls(out_dir)
        store.records: list[dict] = []
        if not store.path.exists():
            raise FileNotFoundError(f"no state.jsonl in {out_dir}")
        for line in store.path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") != "explanation":
                store.records.append(r)
                continue
            fields = {k: v for k, v in r.items() if k not in ("type", "ts", "pass")}
            exp = Explanation(pass_num=r.get("pass", 0), **fields)
            store.explanations[exp.id] = exp
            num = int(exp.id.split("_")[1])
            store._counter = max(store._counter, num)
        return store

    # --- ids ---
    def new_id(self) -> str:
        self._counter += 1
        return f"exp_{self._counter:04d}"

    # --- writing ---
    def _append(self, record: dict) -> None:
        record.setdefault("ts", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_record(self, record: dict) -> None:
        self._append(record)

    def put(self, exp: Explanation) -> None:
        self.explanations[exp.id] = exp
        self._append(exp.to_record())

    def event(self, name: str, **kw) -> None:
        self._append({"type": "event", "event": name, **kw})

    # --- queries ---
    def by_status(self, status: str) -> list[Explanation]:
        return sorted(
            (e for e in self.explanations.values() if e.status == status),
            key=Explanation.sort_key,
        )

    def open_units(self) -> list[Explanation]:
        return self.by_status(OPEN)

    def final_units(self) -> list[Explanation]:
        return sorted(
            (e for e in self.explanations.values() if e.status in (LOCKED, IRREDUCIBLE)),
            key=Explanation.sort_key,
        )

    def live_units(self) -> list[Explanation]:
        """Units that currently own source lines (merged-away parents excluded)."""
        merged_away = {pid for e in self.explanations.values() for pid in e.parent_ids}
        return sorted(
            (e for e in self.explanations.values() if e.id not in merged_away),
            key=Explanation.sort_key,
        )

    # --- rendering ---
    def render_markdown(self, source_name: str) -> Path:
        lines = [f"# Explanations — {source_name}", ""]
        for exp in self.live_units():
            ranges = ", ".join(f"{s}-{e}" for s, e in exp.ranges)
            lines.append(f"## Lines {ranges}  [{exp.status.upper()}]")
            lines.append("")
            lines.append(exp.text.strip())
            if exp.invariants:
                lines.append("")
                lines.append("Invariants:")
                lines.extend(f"- {inv}" for inv in exp.invariants)
            if exp.external_deps:
                lines.append("")
                lines.append(f"External dependencies: {', '.join(exp.external_deps)}")
            lines.append("")
            lines.append("=" * 72)
            lines.append("")
        out = self.out_dir / "explanations.md"
        out.write_text("\n".join(lines))
        return out
