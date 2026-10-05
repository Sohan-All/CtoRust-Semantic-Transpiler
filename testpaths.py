"""Filesystem roots the tests read, resolved from the environment.

These used to be absolute paths written into each test file. That made every
test fail for anyone whose tree is not laid out exactly like the author's, and
it published a username and directory layout nobody else can use. Same reasoning
as `config.py`'s `_KEYS_DIR`, and the same defaulting trick: each root defaults
to the existing layout relative to this checkout's parent, so local runs need no
environment at all.

  DIFFUSIONMTUS_CORPUS  the B03_organic corpus root, holding <project>/test_case.
                        Defaults to <tree>/Test-Corpus/Public-Tests/B03_organic.
  DIFFUSIONMTUS_RUNS    recorded ablation runs (<tag>/_project_*/...).
                        Defaults to <tree>/mtu_runs/abl2/runs.
  DIFFUSIONMTUS_SCORE   recorded verdicts (<tag>.verdict).
                        Defaults to <tree>/mtu_runs/abl2/score.

`mtu_runs/` is not a git repo and the corpus is a separate checkout, so all
three are absent on a fresh clone. That is expected: a test whose fixture is a
recorded run SKIPs when the recording is not on disk rather than failing, since
"the evidence is not here" and "the code is wrong" are opposite findings.
"""

import os
from pathlib import Path

_TREE = Path(__file__).resolve().parent.parent

CORPUS = Path(os.environ.get(
    "DIFFUSIONMTUS_CORPUS", _TREE / "Test-Corpus/Public-Tests/B03_organic"))
RUNS = Path(os.environ.get(
    "DIFFUSIONMTUS_RUNS", _TREE / "mtu_runs/abl2/runs"))
SCORE = Path(os.environ.get(
    "DIFFUSIONMTUS_SCORE", _TREE / "mtu_runs/abl2/score"))
