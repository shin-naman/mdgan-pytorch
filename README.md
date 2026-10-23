# MD-GAN PyTorch Reproduction

A research reproduction of MD-GAN for water diffusion in polymers: a generative model that
learns short MD trajectories and predicts long-time diffusion.

## Research basis
- Endo et al. (2018), AAAI: the original MD-GAN.
- Kawada et al. (2022), Soft Matter: MD-GAN with multi-particle input; architecture in ESI Fig. S2.
- Kawada et al. (2023), JCIM: water diffusion in 28 polymers. The first target here is its PE
  setup: single water center-of-mass displacements, 1.28 ns input, 80 ns prediction.

## Status
Implemented and tested: preprocessing of the published GROMACS trajectories, the three
networks (G_z, G_Y, D_G), WGAN-GP losses, rollout, checkpoints, evaluation (MSD, D, err) and the
training loop, with cluster support for CHTC. Learning is verified on synthetic Brownian
data; the paper-scale PE runs are next. Findings and open questions: `docs/decisions.md`.

## Getting started (macOS/Linux)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,data]'      # data = mdtraj, for reading .xtc trajectories
python -m pytest -q                          # fast tier; see docs/testing.md for the others
```

Data: the trajectories are on Zenodo (10.5281/zenodo.7295407). Preprocess one system into a cache:

```sh
python -m mdgan.data.preprocess data/raw/wat_in_polymer_mdgan/MD_data/PE --out data/cache/PE.npz
```

Train and evaluate locally (small runs) or on CHTC (`docs/cluster.md`):

```sh
python -m mdgan.train data/cache/PE.npz --out runs/pe_test --set n_iterations=1000 --device mps
python -m mdgan.evaluate runs/pe_test/snap_001000.pt data/cache/PE.npz
```

## Layout
- `src/mdgan/`: the package
  - `models/`: G_z (latent transition), G_Y (trajectory decoder), D_G (critic)
  - `data/`: `.xtc` reading, water COM unwrapping, preprocessing, MSD, training windows
  - `config.py`, `losses.py`, `rollout.py`, `checkpoint.py`, `evaluate.py`, `train.py`
- `tests/`: one file per module; `integration/`; `mutation/` (deliberate breakages the suite must catch)
- `configs/`: run configurations (`pe_paper.json` = the paper's PE protocol)
- `tools/`: `chtc/` (HTCondor submit files and job scripts), `diag_brownian.py` (training-stability experiments)
- `docs/`: `architecture.md` (the networks), `decisions.md` (decisions, findings, open questions),
  `testing.md` (test tiers and policy), `cluster.md` (running on CHTC)

This repository does not contain MD trajectories, caches or trained weights.