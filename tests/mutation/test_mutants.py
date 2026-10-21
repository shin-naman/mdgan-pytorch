"""Mutation tests: every hand-written mutant must be killed by its target test file.

Each case runs the target file in a fresh interpreter with the mutant patched in
(see mutants.py). Verdicts are strict:
  exit 1  -> KILLED (some test failed or errored)          -> pass
  exit 0  -> SURVIVED (the suite has a hole)                -> fail
  other   -> the run itself broke (bad patch, collection)   -> fail, output shown
A baseline case per target file first proves the unmutated file passes, so a
mutant cannot be "killed" by an already-broken suite.

Excluded from the default run (marker `mutation`). Run with:
    python -m pytest -m mutation tests/mutation -v
Real-data tests are disabled in the subprocesses (MDGAN_PE_DIR is removed).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mutants import MUTANTS

pytestmark = pytest.mark.mutation

REPO = Path(__file__).resolve().parents[2]
TARGETS = sorted({target for _, _, target, _ in MUTANTS})

RUNNER = """
import sys
{patch}
import pytest
sys.exit(pytest.main([{target!r}, "-x", "-q", "-p", "no:cacheprovider", "--no-header"]))
"""


def _run(target: str, patch: str = "") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "MDGAN_PE_DIR"}
    return subprocess.run([sys.executable, "-c", RUNNER.format(patch=patch, target=target)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("target", TARGETS)
def test_baseline_target_passes_unmutated(target):
    proc = _run(target)
    assert proc.returncode == 0, f"{target} fails without any mutant:\n{proc.stdout[-3000:]}{proc.stderr[-3000:]}"


@pytest.mark.parametrize("group,name,target,patch", MUTANTS, ids=[f"{g}:{n}" for g, n, _, _ in MUTANTS])
def test_mutant_is_killed(group, name, target, patch):
    proc = _run(target, patch)
    if proc.returncode == 0:
        pytest.fail(f"SURVIVED: {group}:{name} is not detected by {target}")
    assert proc.returncode == 1, (
        f"{group}:{name}: the mutated run did not complete normally (exit {proc.returncode}); "
        f"the patch is probably broken.\n{proc.stdout[-3000:]}{proc.stderr[-3000:]}")