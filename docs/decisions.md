# Research decisions and unresolved questions

Status key: Confirmed / Decided / Open

- **Confirmed**: stated in a paper, or a mathematical property we verified.
- **Decided**: not stated in any paper; our documented assumption, with reasoning. Revisit if a source or the mentor settles it.
- **Open**: not yet decided.

Sources: Endo et al. 2018 (AAAI), Kawada et al. 2022 (Soft Matter) + ESI, Kawada et al. 2023 (JCIM).

## Scope

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Target system | Confirmed | Water COM in PE, 2023 Fig. 4 setup: n_s = 1 (1.28 ns input), 80 ns MD, 10 ps sampling. Other polymers later. |
| Subsystem | Confirmed | Single-particle input: water COM per-step displacements, C = 3. Multi-particle (C = 9) deferred; all modules take `n_channels`. |
| M = 64 | Confirmed | 2022, 2023. |
| n_z = 32 | Confirmed | 2023 PE baseline (Fig. 5 minimum). |
| Architecture reference | Confirmed | 2022 ESI Fig. S2. Affine-based G_Y (not the 2018 U-Net); no D_z critic (dropped in 2022). |
| Data representation | Confirmed | Per-step displacements, not positions; cumsum from origin for MSD. 2022 §3.1, 2023 Method. |
| Framework / layout | Decided | PyTorch. Tensors are (B, T, C); transpose to (B, C, T) only inside Conv1d calls. |
| Module style | Decided | Named layer attributes, not `nn.Sequential`, in all three networks: readable state-dict keys, stable hook targets, index-free tests. |
| Randomness | Decided | All randomness enters via explicit tensors or an explicit `torch.Generator`; no `forward` consumes the global RNG. Each generator has `sample_noise`; G_z has `unroll`, G_Y has `generate`. |
| Input validation | Decided | Cheap shape checks only (rank, feature counts, batch match). No NaN/inf checks in the hot path (G_z `validate_finite=False`; opt-in forces a device sync). Divergence detection, checkpointing and rollback belong to the trainer. |
| Weight initialization | Decided | PyTorch default everywhere. Measured G_z pre-BN variance at production width ≈ 1e-4 (eps = 1e-5), output variance at init ≈ 0.92; harmless. A shared init helper may be revisited if WGAN-GP training proves sensitive to the critic's initial input-gradient scale (recorded by a test). |
| Test strategy | Decided | Per module: contract (architecture pinned to the figure), numerics (regimes met in a real run), adversarial (silent-failure modes). `tools/mutate_*.py` applies deliberate breakages and requires every mutant to be killed. A failing test on real code → investigate the product; a surviving mutant → strengthen the tests. |

## G_z (latent transition) — implemented, `src/mdgan/models/latent_transition.py`

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Layer stack | Confirmed | [z, r] → 3 × (Affine 2048 + LReLU) → Affine → BatchNorm1d → z'. ESI Fig. S2(B). |
| Final affine dimension | Decided | Outputs n_z. Fig. S2(B) labels it 2048, but BatchNorm cannot change dimension; the label is an error. |
| Concatenation order | Decided | [z, r]. Not specified; pinned by test. |
| BatchNorm statistics | Decided | **Batch statistics in both train and eval** (`track_running_stats=False`, default; `affine=True`). Reasoning: (1) during the unrolled burn-in, train-mode BN sees z_1 (near uniform init) through z_7+ (stationary), so running stats would describe no real step; (2) output-layer BN with batch stats re-centers/re-scales the ensemble at every transition — the distribution-stabilization mechanism that let 2022 drop D_z; running stats disable it exactly during the long estimation rollout; (3) estimation batch is 8192, so batch stats are effectively exact. Consequence: B ≥ 2 required in both modes. Running-stats mode kept as opt-in for ablation. Pending mentor confirmation. |
| n_r (G_z noise dim) | Decided | n_r = n_z = 32. Not given in any source. |
| LReLU slope | Decided | 0.2 (common default; not given). One stateless `act` shared by the three uses. |
| Burn-in gradient | Decided | Full unroll (BPTT through all burn-in steps). `unroll()` returns all states (steps+1, B, n_z); callers detach if they want to. |
| Burn-in length c | Confirmed | c = 7 (ESI §1: "after seven or more time evolutions"). Configurable. |

## G_Y (trajectory decoder) — implemented, `src/mdgan/models/trajectory_generator.py`

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Layer stack | Confirmed | u (M×C, normal) → Conv1d(C→C, k=1, s=1) → flatten M·C; [r', z, u'] → Affine+LReLU 2048 → residual (Affine+LReLU 2048 → Affine 2048, skip-add) → LReLU → Affine M·C → (M, C). ESI Fig. S2(A). |
| Noise mixing conv | Decided | Kept as a real `nn.Conv1d` (not the equivalent per-step `Linear(C, C)`) so parameter names/shapes match the reference. It is the same C×C affine at every step; no temporal receptive field. |
| Concatenation order | Decided | [r', z, u'], the figure's left-to-right order. Pinned by test with a negative control. |
| Flatten / reshape order | Decided | Step-major: flat index = t·C + axis, for both the u' flatten and the output reshape. Pinned by tests. |
| Residual block | Decided | Separate `ResidualAffineBlock` (fc1 → LReLU → fc2, identity skip, no activation after the add) so the skip is testable. The LReLU after the ⊕ is the main stack's `post_act`. |
| Activations | Decided | `input_act`, `post_act`, and the block's own `act` are separate instances so hooks can distinguish them; functionally one shared instance would be identical. |
| Normalization | Confirmed | None. eval ≡ train; B = 1 legal. |
| Noise ownership | Decided | `forward(z, r_prime, u)` takes explicit noise; `sample_noise(batch)` → (r', u); `generate(z)` is the convenience. |
| Mean operation | Decided | Not inside G_Y; it acts on the 2M window and lives in `mdgan.ops`. |

## D_G (critic) — implemented, `src/mdgan/models/critic.py`

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Layer stack | Confirmed | (y_k ‖ y_{k+1}) = 2M×C → Conv1d(C→2048, k=2M, s=2M) → LReLU → 3 × (Affine+LReLU 2048) → Affine 1. ESI Fig. S2(C). |
| Input | Decided | `forward(window)` takes one (B, 2M, C) window; the trainer concatenates G_Y's two blocks. Window length must equal 2M exactly (a multiple would silently yield extra conv positions). |
| Input conv | Decided | Kept as `nn.Conv1d`; with T = 2M it is one affine map over the step-major flattened window (weight (H, C, 2M) pairs with the window via `permute(0, 2, 1)`). Pinned by test. |
| Output shape | Decided | (B,), via `squeeze(-1)` with an explicit axis so B = 1 keeps its batch dimension (per-sample GP). |
| Normalization | Confirmed | None of any kind. WGAN-GP's penalty is per-sample; BatchNorm would couple samples. Tested via per-sample gradient independence, with a positive control showing the test detects coupling. |
| Naming | Decided | File `critic.py`, class `Critic` (Wasserstein score, not a probability); the paper's D_G in prose. |
| GP and biases | Confirmed (property) | For a piecewise-linear critic, ∂D/∂x is a product of weight matrices and slope masks; the gradient penalty therefore reaches every weight and no bias (autograd reports None or exact zero). Biases are trained by the Wasserstein term. Not a bug; a smooth activation would change this, but the paper specifies LReLU. |
| Critic output bias | Confirmed (property) | ∂(E[D(fake)] − E[D(real)])/∂b_out = 1 − 1 = 0 and the GP adds nothing: identically zero gradient, never updates. D is defined up to a constant under the Wasserstein objective. Kept to match "Affine → 1"; `bias=False` would be an equivalent cosmetic choice. |

## Ops — implemented, `src/mdgan/ops.py`

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Mean operation (2023 eq. 9) | Decided | `mean_operation(x)` subtracts the batch mean per step per axis. Applied by the trainer to the **generated** 128×3 displacement window, inside the autograd graph, **before** gradient-penalty interpolation. Not applied to real data (as the paper states). Never applied to latents. Centering displacements per step ≡ centering positions per time (cumsum is linear). Its Jacobian is a projection (w − mean w), not a detach. At N = 8192 the removed mean is ~σ/90, so the paper's accuracy gain is a training-time regularizer. |

## Training design — decided, not yet implemented

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Base loss | Confirmed | WGAN-GP (2022 eq. 5). Critic maximizes L; G_z and G_Y jointly minimize. |
| λ, optimizer, schedule | Decided | λ = 10, Adam lr = 1e-4, betas = (0.0, 0.9), 5 critic updates per generator update (WGAN-GP paper defaults; 2022 says Adam only). |
| "300,000 iterations" | Decided | Interpreted as generator updates; checkpoint/estimate every 10,000. |
| Batch sizes | Confirmed | 64 in training; 8192 in estimation. |
| Latent pair selection | Decided | Unroll each chain to c + K steps (K = 4), draw one random j ∈ [c, c+K−1] per iteration, decode (z_j, z_{j+1}), concatenate to one 128×3 window. One fake window per chain → 64 real vs 64 fake. Matches 2018's randomized post-burn-in pair; avoids training G_Y only on step-c latents. Encoded by `_fake_window` in `tests/integration`. |
| Gradient flow | Decided | Generator step: loss −E[D(fake)] backprops through the critic into G_Y and G_z; the critic's gradients are discarded (its optimizer does not step). Critic step: the fake window is `.detach()`ed before the critic so no gradient reaches the generators. |
| Gradient penalty | Decided | Reference implementation in `tests/D_G/test_dg_numerics.py::_gp` (and `tests/integration`). The trainer's `gradient_penalty` must agree with it; the integration tests will import the trainer's version when it exists. |
| Input scaling | Decided | Divide displacements by one scalar σ = pooled std over x,y,z of training step-displacements; rescale G_Y output by σ only when computing physical MSD/D. Reasoning: WGAN-GP's unit-gradient-norm penalty is in input units; water-in-PE step displacements are ~0.04 nm, which would force a nearly flat critic. One isotropic scalar preserves rotational symmetry. σ saved with checkpoints. |
| Checkpoint format and loading | Decided | One file keyed by module name (`G_z`, `G_Y`, `D_G`) each with class name + constructor config + state dict, plus optimizers, RNG states, iteration, σ. The loader rebuilds from config, verifies the class name, and uses `strict=True` only; no non-strict path exists. Reason: modules share key names (`input_layer.*`, `hidden1.*`, `output_layer.*`); a non-strict cross-load silently copies same-shaped tensors (tested in `tests/G_Y/test_gy_adversarial.py`). |
| Estimation rollout | Decided | 8192 chains; burn in c steps; then alternate G_z → G_Y, concatenate blocks; 80 ns at n_s = 1 is 125 blocks of M = 64; cumsum from origin; multiply by σ. Reproducible from a seed via explicit generators. |
| Checkpoint selection | Open | Papers pick the checkpoint with lowest err against long MD (uses ground truth). For blind use, the latent-distribution diagnostic (2023 Fig. 4B/C) is the proposed alternative. |
| Robustness policy | Decided | Modules never sanitize NaNs. The trainer owns: atomic checkpoints every N iterations, a NaN/inf guard on losses each step, and rollback to the last checkpoint on repeated divergence. |

## Data pipeline — open

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Source | Open | Own GROMACS runs planned; 2023 PE data on Zenodo (10.5281/zenodo.7295407) available for validation first. |
| Preprocessing | Decided | Unwrap PBC before water COM; diff positions → displacements; apply step skip n_s; windows of 2M = 128. Units nm. |
| First learning test | Decided | Synthetic Brownian walk with known D before any MD data: the full model must recover D. This is the first test of the model rather than the code. |

We explicitly distinguish reproduced architecture from undocumented implementation assumptions.