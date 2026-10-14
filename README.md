# MD-GAN PyTorch Reproduction

A research-oriented reproduction of the MD-GAN method for water diffusion in polymer systems.

## Research basis
- Kawada et al. (2022), *MD-GAN with multi-particle input: the machine learning of long-time molecular behavior from short-time MD data* (Soft Matter), especially supplementary Fig. S2.
- Kawada et al. (2023), *Prediction of Water Diffusion in Wide Varieties of Polymers with All-Atom Molecular Dynamics Simulations and Deep Generative Models* (JCIM).

The initial target is the **single water center-of-mass displacement** in polyethylene, following the 2023 experiment. GROMACS integration will be added in a later stage.

## Getting started (macOS/Linux)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -c 'import torch; print("torch:", torch.__version__, "MPS:", torch.backends.mps.is_available())'
pytest
```

## Layout
- `src/mdgan/models/`: trainable networks (`latent_transition.py` is intentionally a placeholder)
- `tests/`: model and tensor contract tests
- `docs/architecture.md`: scientific architecture contract and reference
- `docs/decisions.md`: known unknowns and research decisions

## Roadmap
1. Implement and test `G_z`.
2. Implement `G_Y` and critic `D_G`.
3. Implement WGAN-GP training and synthetic-data tests.
4. Add GROMACS processing and physical evaluation.

This repository does not contain MD trajectories or trained weights.
