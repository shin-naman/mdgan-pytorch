# Testing

The suite answers two different questions and keeps them apart:

1. **Is the code what we decided?** Unit tests per source module, pinned to the
   architecture in `docs/architecture.md` and the choices in `docs/decisions.md`.
2. **Would the tests notice if it stopped being?** Mutation tests: deliberate
   breakages that the unit tests must catch.

A third question, **does the model learn?**, is answered by `learning` tests, which
arrive with the trainer (synthetic Brownian walk with known D, see decisions.md).

## Running

```
python -m pytest                                  # fast tier (default)
python -m pytest -m mutation tests/mutation       # mutation tier, ~1 min
python -m pytest -m gpu tests/test_devices.py     # CUDA / MPS smoke
python -m pytest -m "slow or learning"            # nightly tier
python -m pytest -m ""                            # everything
MDGAN_PE_DIR=data/raw/wat_in_polymer_mdgan/MD_data/PE python -m pytest   # + real data
```

A `-m` on the command line replaces the one in `pyproject.toml`'s `addopts`.

## Tiers and markers

Registered in `pyproject.toml` with `--strict-markers` (an unregistered marker is an error).

| Marker | Default run | Meaning |
| --- | --- | --- |
| *(none)* | yes | Fast unit and integration tests. Whole tier runs in a few seconds. |
| `real_data` | yes, skips itself | Needs `MDGAN_PE_DIR` = `MD_data/PE`. Never excluded by marker; the env var is the gate. |
| `statistical` | yes | Asserts a sampling tolerance. The docstring derives the tolerance (see policy below). |
| `ablation` | yes | Exercises an option the baseline does not use (G_z running-stats BatchNorm). Cheap, so it runs. |
| `gpu` | no | Needs CUDA or MPS. Smoke only. |
| `slow` | no | More than ~10 s. Nightly. |
| `learning` | no | Trains something and asserts what it learned. Nightly. |
| `mutation` | no | Applies a deliberate breakage and asserts the suite catches it. |

## Layout

One test file per source module, flat under `tests/`; cross-module tests in subfolders.

| Source | Tests |
| --- | --- |
| `models/latent_transition.py` | `tests/test_latent_transition.py` |
| `models/trajectory_generator.py` | `tests/test_trajectory_generator.py` |
| `models/critic.py` | `tests/test_critic.py` |
| `ops.py` | `tests/test_ops.py` |
| `data/water.py` | `tests/test_water.py` |
| `data/xtc.py` | `tests/test_xtc.py` |
| `data/preprocess.py` | `tests/test_preprocess.py` |
| `data/msd.py` | `tests/test_msd.py` |
| `data/dataset.py` | `tests/test_dataset.py` |
| three networks together | `tests/integration/test_three_networks.py` |
| devices | `tests/test_devices.py` |
| mutation | `tests/mutation/` (`mutants.py` + `test_mutants.py`) |

Shared pieces live in `tests/conftest.py` (the only conftest; a second one would
replace `sys.modules["conftest"]` mid-collection and break `from conftest import SMALL`)
and `tests/synthetic_water.py` (rigid-water ground truth, `write_xtc`).

`SMALL` (n_z = 7, n_r = 3, M = 8, C = 3, H = 24) is the one set of tiny model
dimensions. M ≠ C on purpose: a (B, T, C) ↔ (B, C, T) transpose bug cannot pass a
shape check by accident.

### Parked tests

`tests/integration/test_three_networks.py` holds tests whose home module does not
exist yet. They move when it does:

| Tests | Moves to | When |
| --- | --- | --- |
| `_gp` reference, GP reaches weights not biases | `test_losses.py` (must agree with `losses.gradient_penalty`) | `losses.py` |
| fake-window pairing, consecutive blocks differ, cumsum, 80 ns = 125 blocks, 8192 chains | `test_rollout.py` | `rollout.py` |
| cross-module state-dict hazards | `test_checkpoint.py` (as tests of the strict loader) | `checkpoint.py` |

## What a test must do to be kept

- Test **our** code or **our** decisions. Tests that pass for every `nn.Module`
  (input not modified, deepcopy independent, pickle round trip, fp64 runs, ...)
  were removed; they carry no information.
- Prefer **behaviour** to structure. Example: "output of sample i does not depend
  on sample j" instead of "no BatchNorm class in `modules()`".
- Prefer a **hand-written reference forward** to hooks that record call order. It
  pins concat order, layer order and activation placement and survives refactors.
  It can only see an activation after the output if some outputs are negative:
  the critic's reference test shifts the scores to straddle zero for that reason
  (found by the `D_G:act_after_output` mutant).
- Every exact-number assertion states where the number comes from: a paper, a
  closed form, or a measurement recorded in `decisions.md`.

## Determinism policy

- Every test runs under a seed derived from its node id (autouse fixture in
  `conftest.py`). No assertion depends on an unseeded RNG.
- Tests of our explicit-generator design (`sample_noise`, `generate`, `unroll`,
  `WindowDataset.sample`) assert that the global RNG is untouched when a generator
  is passed. Plain forward passes are not tested for this.
- `statistical` tests use a fixed seed **and** a derived tolerance. The tolerance is
  5 standard errors of the estimator, derived in the docstring, so a correct
  implementation passes under any seed while the bugs the test exists for (factors
  of 2/3/6, wrong dt, wrong axis count) are far outside it.
- GPU tests are smoke only: they assert the path runs and agrees with CPU to 1e-4,
  nothing about speed or bit-exactness.

## Mutation testing

`tests/mutation/mutants.py` lists hand-written mutants, each `(group, name, target
test file, patch)`. `test_mutants.py` runs the target file in a fresh interpreter
with the patch applied. Verdicts are strict: exit 1 = killed (pass), exit 0 =
survived (fail), anything else = the patch itself is broken (fail, output shown).
A baseline run per target proves the unmutated file passes first.
`MDGAN_PE_DIR` is removed in the subprocesses.

Current state: **60 mutants, all killed** (8 baselines pass), ~65 s.

| Group | Mutants | Examples |
| --- | --- | --- |
| G_z | 15 | concat order, activation placement, BatchNorm removed / running stats / no affine, singleton accepted, unroll detached / off by one / drops z0 |
| G_Y | 14 | channel-major flatten or reshape, temporal leak in the noise conv, residual without skip, activation after the add, noise ignored, uniform u |
| D_G | 15 | accepts 4M windows, squeeze without axis, BatchNorm added, input detached, sigmoid or LReLU on the score, second block ignored, stride 1 |
| data | 16 | minimum image with floor, molecule not made whole, equal masses, guard at L/4 or disabled, triclinic accepted, cache ignores source changes, time average over T, D without /6, linear err weights, σ from the full trajectory (leak), phase ignored |

Removed when the suites were consolidated, with reasons:

- `G_z:validate_finite_default_on`: the flag no longer exists.
- `D_G:forward_consumes_global_rng`: the critic's forward has no module-level RNG
  contract. Global-RNG reproducibility is to be covered by the trainer's bit-exact
  resume test (`test_checkpoint.py`).

Adding a mutant: append a tuple to `MUTANTS`; the patch runs before pytest imports
the target, so monkeypatching the module or class in place is enough. If it
survives, add the test that kills it; never weaken the mutant.

A generic tool (`mutmut`) has not been run yet; Python 3.14 support is uncertain.
Its survivors would be triaged into: new test, equivalent mutant, or accepted gap.

## CI

`.github/workflows/ci.yml`: ruff and the fast tier on every push and PR (Python 3.12,
CPU-only torch); the `slow`/`learning` tiers nightly (an empty tier is not a failure
until those tests exist). `.github/workflows/mutation.yml`: the mutation tier when
`src/mdgan/**`, `tests/**` or `pyproject.toml` change. Real-data tests skip in CI.

Ruff rules are pinned in `pyproject.toml` (`E4 E7 E9 F B`) so an upgrade cannot change
what CI enforces; tests may use compact one-line statements (`E701/E702`).

## Known gaps

- No `learning` tests yet: nothing verifies that the model learns. First one is the
  synthetic Brownian D-recovery test, with the trainer.
- No golden-output tests. Deliberately: comparing forward outputs across torch or
  BLAS versions produces spurious failures. Reference-forward tests cover the same
  regressions without depending on kernel arithmetic.
- Real-data tests cover PE only; the 0.45 L step guard is unverified on the other
  27 systems.