# Research decisions and unresolved questions

Status key: Confirmed / Decided / Open

- **Confirmed**: stated in a paper.
- **Decided**: not stated in any paper; our documented assumption, with reasoning. Revisit if a source or the mentor settles it.
- **Open**: not yet decided.

Sources: Endo et al. 2018 (AAAI), Kawada et al. 2022 (Soft Matter) + ESI, Kawada et al. 2023 (JCIM).

## Scope

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Target system | Confirmed | Water COM in PE, 2023 Fig. 4 setup: n_s = 1 (1.28 ns input), 80 ns MD, 10 ps sampling. Other polymers later. |
| Subsystem | Confirmed | Single-particle input: water COM per-step displacements, C = 3. Multi-particle (C = 9) deferred. |
| M = 64 | Confirmed | 2022, 2023. |
| n_z = 32 | Confirmed | 2023 PE baseline (Fig. 5 minimum). |
| Architecture reference | Confirmed | 2022 ESI Fig. S2. Affine-based G_Y (not the 2018 U-Net); no D_z critic (dropped in 2022). |
| Data representation | Confirmed | Per-step displacements, not positions; cumsum from origin for MSD. 2022 §3.1, 2023 Method. |
| Framework / layout | Decided | PyTorch. Tensors are (B, T, C); transpose to (B, C, T) only inside Conv1d calls. |

## G_z (latent transition) — implemented

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Layer stack | Confirmed | [z, r] → 3 × (Affine 2048 + LReLU) → Affine → BatchNorm1d → z'. ESI Fig. S2(B). |
| Final affine dimension | Decided | Outputs n_z. Fig. S2(B) labels it 2048, but BatchNorm cannot change dimension; the label is an error. |
| Concatenation order | Decided | [z, r]. Not specified; pinned by test so checkpoint weight layout is unambiguous. |
| BatchNorm statistics | Decided | **Batch statistics in both train and eval** (`track_running_stats=False`, default). Reasoning: (1) during the unrolled burn-in, train-mode BN sees z_1 (near uniform init) through z_7+ (stationary), so running stats would describe no real step; (2) output-layer BN with batch stats re-centers/re-scales the ensemble at every transition — this is the distribution-stabilization mechanism that let 2022 drop D_z; running stats disable it exactly during the long estimation rollout; (3) estimation batch is 8192, so batch stats are effectively exact. Running-stats mode kept as opt-in for ablation. Pending mentor confirmation. |
| n_r (G_z noise dim) | Decided | n_r = n_z = 32. Not given in any source. |
| Weight initialization | Decided | PyTorch default. Measured pre-BN variance at production width ≈ 1e-4 vs eps = 1e-5, so output variance at init ≈ 0.92, not 1. BatchNorm's role (zero mean, bounded O(1) spread, no drift) is unaffected. If a shared init policy is adopted for all nets it will live in a common helper, decided when D_G is built (gradient-penalty scale argument). |
| LReLU slope | Decided | 0.2 (common default; not given). Configurable. |
| Input validation | Decided | Cheap shape checks only (rank, feature counts, batch match, B ≥ 2 when BN uses batch stats). No NaN/inf checks in the hot path (`validate_finite=False`; opt-in forces a device sync). Divergence detection, checkpointing and rollback belong to the trainer. |
| Burn-in gradient | Decided | Full unroll (BPTT through all burn-in steps). Eight MLP steps are cheap and it lets G_z learn to land on the stationary distribution. `unroll()` returns all states; callers detach if they want to. |
| Burn-in length c | Confirmed | c = 7 (ESI §1: "after seven or more time evolutions"). Configurable. |
| Randomness | Decided | All randomness enters via explicit tensors or an explicit `torch.Generator`; `forward` never consumes the global RNG. Estimation runs are reproducible from a seed. |

## Training design — decided, not yet implemented

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Base loss | Confirmed | WGAN-GP (2022 eq. 5). Critic maximizes L; G_z and G_Y jointly minimize. |
| λ, optimizer, schedule | Decided | λ = 10, Adam lr = 1e-4, betas = (0.0, 0.9), 5 critic updates per generator update (WGAN-GP paper defaults; 2022 says Adam only). |
| "300,000 iterations" | Decided | Interpreted as generator updates; checkpoint/estimate every 10,000. |
| Batch sizes | Confirmed | 64 in training; 8192 in estimation. |
| Latent pair selection | Decided | Unroll each chain to c + K steps (K = 4), draw one random j ∈ [c, c+K−1] per iteration, decode (z_j, z_{j+1}), concatenate to one 128×3 window. One fake window per chain → 64 real vs 64 fake. Matches 2018's randomized post-burn-in pair; avoids training G_Y only on step-c latents. |
| Mean operation (2023 eq. 9) | Decided | Applied to the **generated** 128×3 displacement window: subtract batch mean per step per axis, inside the autograd graph, **before** gradient-penalty interpolation. Not applied to real data (as the paper states). Never applied to latents. Centering displacements per step ≡ centering positions per time (cumsum is linear). Note: at N = 8192 the removed mean is ~σ/90, so the paper's accuracy gain is a training-time regularizer. |
| Input scaling | Decided | Divide displacements by one scalar σ = pooled std over x,y,z of training step-displacements; rescale G_Y output by σ only when computing physical MSD/D. Reasoning: WGAN-GP's unit-gradient-norm penalty is in input units; water-in-PE step displacements are ~0.04 nm, which would force a nearly flat critic. One isotropic scalar preserves rotational symmetry. σ saved with checkpoints. |
| Checkpoint selection | Confirmed | Papers pick the checkpoint with lowest err against long MD (uses ground truth). For blind use, latent-distribution diagnostic (2023 Fig. 4B/C) is the proposed alternative — Open for us. |

## G_Y and D_G — spec agreed, to be built next

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| G_Y stack | Confirmed | u (64×3, normal) → Conv1d(3→3, k=1, s=1) → flatten 192; [r', z, u'] → Affine+LReLU 2048 → residual (Affine+LReLU 2048 → Affine 2048, skip-add) → LReLU → Affine 192 → (64, 3). ESI Fig. S2(A). |
| D_G stack | Confirmed | (y_k ‖ y_{k+1}) = 128×3 → Conv1d(3→2048, k=128, s=128) → LReLU → 3 × (Affine+LReLU 2048) → Affine 1. ESI Fig. S2(C). |
| G_Y / D_G details | Open | Concatenation order, flatten order, residual block as separate module, noise ownership — decide together before coding. |

## Data pipeline — open

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Source | Open | Own GROMACS runs planned; 2023 PE data on Zenodo (10.5281/zenodo.7295407) available for validation first. |
| Preprocessing | Decided | Unwrap PBC before water COM; diff positions → displacements; apply step skip n_s; windows of 2M = 128. Units nm. |

We will explicitly distinguish reproduced architecture from undocumented implementation assumptions.