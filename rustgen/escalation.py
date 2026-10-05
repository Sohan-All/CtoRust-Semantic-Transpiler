"""Metering for model escalation. See ../../plan.md.

This module is the SINGLE WRITE POINT for the `escalation` record, for the same
reason `compile_loop.set_section` is the single write point for repairs: a
record shape assembled independently at three call sites drifts, and the drift
shows up as a summary that silently under-counts one site.

Nothing here decides whether to escalate — that is each site's business. This
only measures a firing that is already happening, and it measures it in DELTAS
because one escalation `LLM` instance serves every firing in a run: its
`usage_record()` gives the run total, and subtracting a snapshot is the only way
to attribute cost to the individual firing that incurred it.

A REJECTED firing is recorded exactly like an accepted one. That is the whole
point — a rejected escalation is indistinguishable from one that never fired
unless it is written down, except that it was paid for. Same reasoning as
`CompileReport.rejected_repairs`.
"""
from __future__ import annotations

import time

# The three sites in plan.md. Kept as a tuple so a typo in a site name fails a
# lookup here rather than quietly creating a fourth category in the summary.
SITES = ("types", "compile", "stubs")


class Escalation:
    """Brackets one firing and produces its record.

        with Escalation("compile", llm, trigger="final", value=2) as e:
            ok, why = ...                      # do the work
            e.done(accepted=ok, rejected_by=why)
        record(pdir, e.record)

    `done()` is what marks the outcome; a block that raises leaves
    `accepted=False` with `error` set, so a crashed escalation still costs a
    record rather than vanishing.
    """

    def __init__(self, site: str, llm, *, trigger: str, value):
        if site not in SITES:
            raise ValueError(f"unknown escalation site {site!r}; "
                             f"expected one of {', '.join(SITES)}")
        self.site = site
        self.llm = llm
        self.trigger = trigger
        self.value = value
        self.accepted = False
        self.rejected_by = ""
        self.error = ""
        # Seeded HERE, not in __enter__. If __enter__ itself raises — a broken
        # llm object, a counter that throws — the record is still built by the
        # caller's `finally`, and a `_t0` of 0.0 would report a `seconds` of
        # roughly the machine's uptime. A record that is wrong is worse than
        # one that is missing, because it gets averaged into a cost number.
        self._t0 = time.monotonic()
        self._snap = self._counters_safe()

    def _counters(self):
        return (getattr(self.llm, "calls", 0),
                getattr(self.llm, "input_tokens", 0),
                getattr(self.llm, "output_tokens", 0))

    def _counters_safe(self):
        """`_counters` but never raising. The metering must not be the thing
        that kills a run: if the counters are unreadable the honest record is
        zero cost with the error attached, not a traceback out of a `finally`.
        """
        try:
            return self._counters()
        except Exception:
            return (0, 0, 0)

    def __enter__(self):
        self._t0 = time.monotonic()
        self._snap = self._counters_safe()
        return self

    def done(self, *, accepted: bool, rejected_by: str = "") -> None:
        self.accepted = bool(accepted)
        self.rejected_by = rejected_by or ""

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.accepted = False
            self.error = f"{exc_type.__name__}: {exc}"
        return False        # never swallow: a site decides its own recovery

    @property
    def record(self) -> dict:
        calls, tin, tout = self._counters_safe()
        c0, i0, o0 = self._snap
        return {
            "type": "escalation",
            "site": self.site,
            "trigger": self.trigger,
            "trigger_value": self.value,
            "model": getattr(getattr(self.llm, "cfg", None), "worker_model", ""),
            "accepted": self.accepted,
            "rejected_by": self.rejected_by,
            "error": self.error,
            "seconds": round(time.monotonic() - self._t0, 1),
            "calls": calls - c0,
            "input_tokens": tin - i0,
            "output_tokens": tout - o0,
        }


def should_escalate_compile(cfg, sites, final_errors: int) -> bool:
    """Site B's gate: did the compile loop end few enough errors above zero to
    be a REPAIR rather than a rewrite?

    Swept over 140 recorded per-run logs (mtu_runs/abl2/sweep_finals.py): of 37
    build failures, 18 sit at exactly `final: 1` and 25 at <= 3, and the tail
    runs 10/19/19/24/34. Buying a rewrite from an expensive model defeats the
    point of a cheap pipeline.

    `final_errors` is -1 when the loop never recorded one and 0 when the crate
    built; neither is a firing. Kept here rather than inline at the call site so
    the threshold has one home and can be tested without a cargo toolchain.
    """
    if "compile" not in sites:
        return False
    return 0 < final_errors <= getattr(cfg, "escalation_max_errors", 3)


def spent_tokens(records: list[dict]) -> int:
    """Total tokens across firings so far. Input + output, because both are
    billed — counting only output understates a repair loop, whose prompts
    carry the whole crate."""
    return sum(r.get("input_tokens", 0) + r.get("output_tokens", 0)
               for r in records)


def over_budget(records: list[dict], cfg) -> bool:
    """Has this RUN spent its escalation budget?

    Checked before each firing, so one pathological crate cannot eat a batch.
    A per-call bound (`call_deadline`) and a per-firing turn cap do not compose
    into a per-run bound — three sites x several rounds each multiply, and this
    project has already paid once for assuming a retry budget bounds a timeout
    rather than multiplying it. The failure mode also changed when the backend
    started billing: an unbounded vLLM loop burned GPU time already paid for,
    an unbounded Vertex loop bills.

    Budget <= 0 disables the cap, which is deliberate: a run explicitly
    configured without a ceiling should not silently get a default one.
    """
    budget = getattr(cfg, "escalation_run_token_budget", 0)
    if budget <= 0:
        return False
    return spent_tokens(records) >= budget


def compile_outcome(before: int, after: int) -> tuple[bool, bool]:
    """(improved, resolved) for a site-B firing. Two DIFFERENT questions.

    `improved` — is the crate closer? Then keep the escalated result; the
      compile loop's own best-tracking guarantees it is not worse.
    `resolved` — did the VERDICT change? Only `final: 0` does.

    Conflating them overstates the one number this work is judged on. A crate
    at 1 error and a crate at 3 both score BUILD_FAILED, so `3 -> 1` bought
    nothing, and recording it as accepted would fill the accept rate with
    firings that changed no outcome. The same step function as site C's
    `after == 0`, for the same reason — and observed live on the first
    multi-site run, which went 3 -> 1 and was still BUILD_FAILED.
    """
    return (0 <= after < before, after == 0)


def enabled_sites(cfg) -> list[str]:
    """Sites switched on for this run, [] when escalation is off entirely.

    `escalation_model` is the master switch: with no model there is nothing to
    escalate TO, and the per-site booleans are subordinate to it. Reported by
    the summary so an arm that escalated nothing is distinguishable from an arm
    that was never configured to.
    """
    if not getattr(cfg, "escalation_model", ""):
        return []
    return [s for s in SITES
            if getattr(cfg, f"escalate_{s}", True)]
