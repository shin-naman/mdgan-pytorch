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
| Randomness | Decided | All randomness enters via explicit tensors or an explicit `torch.Generator`; no `forward` consumes the global RNG. Each generator has `sample_noise`; G_z has `unroll`, G_Y has `generate`. Noise is drawn on the generator's device and moved to the model's, so one CPU generator gives the same stream on CPU, CUDA and MPS. |
| Input validation | Decided | Cheap shape checks only (rank, feature counts, batch match), via `_util.positive_int` for constructor arguments. No NaN/inf checks in the hot path (the G_z `validate_finite` option was removed). Divergence detection, checkpointing and rollback belong to the trainer. |
| Weight initialization | Decided | PyTorch default everywhere. Measured G_z pre-BN variance at production width ≈ 1e-4 (eps = 1e-5), output variance at init ≈ 0.92; harmless. A shared init helper may be revisited if WGAN-GP training proves sensitive to the critic's initial input-gradient scale (recorded by a test). |
| Test strategy | Decided | `docs/testing.md`. One test file per module; tiers by marker (fast default; `real_data`, `statistical`, `gpu`, `slow`, `learning`, `mutation`). Hand-written mutants in `tests/mutation/` must all be killed. A failing test on real code → investigate the product; a surviving mutant → strengthen the tests (twice it exposed a real product flaw: see Evaluation). |

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

## Training — implemented, `src/mdgan/{config,losses,rollout,checkpoint,evaluate,train}.py`

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Configuration | Decided | One frozen `Config` holds every default; module and function keyword defaults must agree with it (`TestNoDrift`). σ (derived from data), device and paths are deliberately not in it. Stored in every checkpoint; `configs/pe_paper.json` writes the PE protocol out in full. |
| Base loss | Confirmed | WGAN-GP (2022 eq. 5). Critic maximizes L; G_z and G_Y jointly minimize. |
| λ, optimizer, schedule | Decided | λ = 10, Adam lr = 1e-4, betas = (0.0, 0.9), 5 critic updates per generator update (WGAN-GP paper defaults; 2022 says Adam only). lr 1e-4 is now also **measured** to be more stable than 5e-4 (Learning experiments). |
| "300,000 iterations" | Decided | Interpreted as generator updates; checkpoint and estimate every 10,000. |
| Batch sizes | Confirmed | 64 in training; 8192 in estimation. |
| Latent pair selection | Decided | `rollout.make_fake_window`: one j ~ U{c, …, c+K−1} per call (c = 7, K = 4), unroll to j+1 (same distribution and gradients as unrolling to c+K, less compute), decode (z_j, z_{j+1}) into one 2M window. |
| Gradient flow | Decided | `critic_loss` detaches the fake window itself, and the trainer builds it under `no_grad` for critic steps (its graph would be discarded). Generator step: −E[D(fake)] through a critic frozen by a context manager (no parameter gradients; restored even on error). |
| Gradient penalty | Decided | `losses.gradient_penalty`: one ε ~ U[0,1) per sample broadcast over the window; the interpolate is detached (a critic-only term, never reaching the generators); `create_graph=True`. Pinned by closed forms on a linear critic. Reaches every critic weight and no bias. |
| Input scaling | Decided | Divide displacements by one scalar σ = std of the training windows; MSD in nm is the σ-unit MSD × σ². WGAN-GP's unit-gradient-norm penalty is in input units, and raw steps (~0.04 nm) would force a nearly flat critic. `WindowDataset(scale_inputs=False)` is the unscaled ablation (σ = 1); the raw spread `window_std_nm` is recorded either way. |
| Estimation rollout | Decided | `rollout.rollout`: z_0 → c burn-in steps → block k decoded from z_{c+k}; mean operation per block (2023, at estimation as in training); eval mode, modes restored; output on CPU. All chains run as **one batch**: G_z's batch-statistics BatchNorm couples chains, so chunking would change results. |
| Evaluation | Decided | `evaluate`: rollout, trim to `estimation_time_ps`, ensemble MSD (accumulated in time chunks: no 1.6 GB float64 copy), err and D against an MD reference computed once. `err` and `diffusion_coefficient` refuse a curve that ends inside their window, and `Config` refuses windows beyond the estimation time: both were silently shortening windows before mutation testing exposed it. |
| Randomness in training | Decided | One CPU generator drives data sampling and model noise. Evaluation uses a separate generator with a fixed seed at every checkpoint, so turning evaluation on or off leaves training bit-identical and every checkpoint is evaluated on the same noise. Bit-exact reproducibility is a CPU guarantee. |
| Checkpoints | Decided | Two kinds, one strict loader (format v2). `latest.pt` (full resume state, ≈375 MB at width 2048) is written at iteration 0 and every checkpoint, overwritten atomically; `snap_XXXXXX.pt` (G_z + G_Y, ≈72 MB) is kept per checkpoint for evaluation. Models are rebuilt from the stored Config; class names checked before any tensor; `strict=True` only; read with `weights_only=True`. |
| Divergence | Decided | One finiteness check per iteration. On failure: roll back to `latest.pt` and re-seed deterministically (seed + 1000 × rollback count); history stored in the checkpoint and `events.jsonl`. More than 3 consecutive rollbacks to the same checkpoint stop the run (a total limit would kill long runs with unrelated NaNs). |
| Resume | Decided | Config must equal the checkpoint's except `n_iterations`; dataset metadata (files' md5, windows, σ, raw spread) must be equal; logs truncated to the restored iteration. CLI overrides are applied on top of the checkpoint's config, so a restarted job repeating its arguments resumes cleanly. |
| Cluster execution | Decided | `docs/cluster.md`. CHTC GPU Lab via HTCondor, PyTorch runtime container, everything in `/home` (every file < 1 GB; `/staging` is for larger files). A job runs within its GPU Lab length class (sized by a short run); `--max-hours` + exit 85 + `--auto-resume` is a safety net, not the main mechanism, and `+is_resumable` (evictable backfill) is not used. Jobs train from the preprocessed `.npz`; the commit is stamped into the code copy; run names carry the cluster ID so resubmissions never overwrite results; progress lines on stdout for `condor_tail`. |
| Checkpoint selection | Open | Papers pick the checkpoint with lowest err against long MD (uses ground truth). Brownian runs show D fluctuating ≈ ±15 % between adjacent late checkpoints, so a single blind pick is noisy. Candidates: average over late checkpoints (Brownian, lr 1e-4: mean D_rel −0.04 over 6k–10k), the 2023 latent-distribution diagnostic. |

## Learning experiments — synthetic Brownian system (`tests/test_train.py`, `tools/diag_brownian.py`)

Miniature model (n_z = 8, M = 8, width 128). Known answer: MSD = 6Dt, no correlations.

| Finding | Status | Evidence / consequence |
| --- | --- | --- |
| Data starvation shrinks the variance | Confirmed (measured) | 64 training windows: step std 0.70, W stuck at 6.9, short-time MSD ≈ half. Exact W1 to 64 points in the 48-dim window space is smaller at std 0.5 than at 1.0, so the objective itself rewards shrinking. 2048 windows: std 0.99, W ≈ 0.5, err 6.4 %. Learning tests use 2048 trajectories. **PE has 60 windows in 384 dims: watch `step std`.** Many short MD trajectories beat few long ones. |
| Long-range block correlations drive D | Confirmed (measured) | D_rel tracks the sum of block-to-block displacement correlations (lags 1–10) together with the block variance. A two-block critic does not constrain correlations beyond one block, so they are what wanders. |
| Learning rate | Measured | 5e-4: best at 1k (err 6.4 %), then degrades (19.8 %, D_rel +0.81 at 3k). 1e-4: improves through 10k; second-half err 5.6–8.8 %, best err 5.6 % / D_rel −0.07. |
| Residual D fluctuation | Open | ≈ ±15 % between adjacent late checkpoints at 1e-4. Next: larger K (`--pair-window 16`), longer runs (does rising W turn into degradation?), possibly the 2018 latent critic D_z. |
| Memory across blocks | Open | Brownian motion has none (a decoder ignoring z could pass). A persistent-walk (AR(1) velocity) test with a known long-time D is planned. |
| Paper scale | Open | Everything above is at miniature scale; the PE runs (critic window 128 steps instead of 16) will show whether the instability carries over. |

## Data pipeline — implemented, `src/mdgan/data/`

Modules: `xtc.py` (read + validate), `water.py` (make whole, COM, unwrap), `preprocess.py`
(per-system cache with provenance), `dataset.py` (training windows + σ), `msd.py` (reference
MSD, ensemble MSD, D, err). Tests in `tests/test_{xtc,water,preprocess,msd,dataset}.py`; the
five real-data tests run when `MDGAN_PE_DIR` points at `MD_data/PE`.

| Item | Status | Decision / reasoning |
| --- | --- | --- |
| Source data | Confirmed | Zenodo 10.5281/zenodo.7295407 (`MD_data/<system>/trajectory/trajNNN.xtc`, `msd.out`), the MD of Kojima et al. 2021 used by the 2023 paper. Kept unzipped under `data/raw/` (gitignored); caches under `data/cache/`. Used to validate the pipeline; our own trajectories will follow the same layout. |
| File facts (PE) | Confirmed | 60 files × 8001 frames at 10 ps (0–80 ns), 3 atoms in (O, H, H) order (TIP3P geometry verified), cubic box ≈ 4.75 nm constant within a trajectory but **different between trajectories** (each run NPT-equilibrated separately), coordinates wrapped per frame. |
| Reader | Decided | mdtraj's topology-free `XTCTrajectoryFile`; ~0.01 s/trajectory. Validates ≥ 2 frames, finite data, uniform strictly increasing time, orthorhombic box, positive lengths. Does not require a constant box (reported as a property). |
| Unwrapping | Decided | Minimum image, vectorized in numpy: make the molecule whole by folding each H relative to O, mass-weighted COM, then fold frame-to-frame COM differences. Identical arithmetic to MDAnalysis `unwrap` + `NoJump` (checked once against MDAnalysis: agreement 2e-6 nm, float32 round-off), ~1000× faster because it is not per-frame Python. MDAnalysis is not a dependency. |
| Masses / geometry | Confirmed | 16.0 / 1.008 / 1.008 from the dataset's `water.top`; 15.999 is indistinguishable at the tested precision. Geometry check against SETTLE (O–H 0.09572, H–H 0.15139 nm, tol 0.01) rejects wrong atom order, non-water, or Å input. |
| Step guard | Decided | Hard error if any folded step exceeds **0.45 L**. A quality-control heuristic below the 0.5 L ambiguity bound, not a physical invariant. L/4 was rejected: real PE has 109 steps above L/4 (max 1.84 nm, cavity-to-cavity hops) that reproduce `msd.out`. Validity on other polymers Open until their data are processed (PE is the fastest). |
| Varying box | Decided | Rejected within a trajectory (minimum image across a changing box is ill-defined; nothing downstream could use it). Differences *between* trajectories are stored per trajectory, (N, 3). `water.gro`'s 10 nm box is never used; the XTC box is. |
| Cache | Decided | One `.npz` per system: displacements (N, T−1, 3) float32 (XTC precision is 1e-3 nm; nothing lost), dt, per-trajectory box, provenance (files, md5, frame count, constants, max step, version). Reused only when md5 list and `PREPROCESS_VERSION` match. |
| Reference MSD | Confirmed | **Time-averaged over all origins per trajectory, then averaged over trajectories.** Evidence: reproduces every column of `msd.out` to 5e-6 at all 8000 lags; an origin-based estimator cannot produce those numbers. `msd.out` columns: mean = mean of N; std = sample std (ddof 1); 95 % = 1.96·std/√N; units nm² (header says "ns²", a typo). FFT implementation with exact origin counts. |
| Generated MSD | Confirmed | Ensemble average from the origin over the generated chains (2023 eq. 5). Kept as a separately named function; never swapped with the reference estimator. |
| D | Confirmed | OLS line with intercept over 10–20 ns, slope/6 (eq. 8). PE reference **1.31593866e-10 m²/s** (`dc_result_md.txt`), slope 0.7896 nm²/ns, reproduced to 1e-4. |
| err | Confirmed + Decided | 2023 eqs. 6–7 with the 2022 ESI discretization: Σ f(t_k)/k over Σ 1/k on the generated grid, 0.2–78 ns, endpoints inclusive, k ≥ 1, reference looked up at exact times (dt_gen must be a multiple of dt_ref). On the authors' PE generated curves (100 ps grid) this gives best-per-run err 7.6/5.6/6.0/4.7/6.6 % with the mean operation and 11.2/16.5/17.5/16.7/14.9 % without; Fig. 4A reads ≈0.5–0.8 lower, attributable to their 10 ps grid. Run ordering identical. |
| Training region | Confirmed | First 2M·n_s displacement steps of each trajectory = the paper's "input length" (1.28 ns for PE = 128 steps from frames 0–128). One window per trajectory per phase; 60 windows for PE. The rest of each trajectory is the **reference region**, used only by `msd.py`; `WindowDataset` is built from a slice and cannot return it. Not a conventional held-out test set: the reference MSD spans the full trajectory, and its use for checkpoint selection (as in the paper) makes it a validation target (Open). |
| Step skip / phases | Decided | Coarse step = sum of n_s consecutive fine steps. `phase_mode="single"` (default, paper): phase 0, region exactly 2M·n_s. `"all"`: region extended by n_s−1 steps, every phase yields a window; those windows are time-shifted copies, not independent samples. Identical for n_s = 1. |
| σ | Decided | One scalar: std of the training windows over trajectories, steps and axes (isotropic), computed from the training windows only, per n_s. Ours, not the paper's (WGAN-GP's gradient penalty is in input units; PE steps are ~0.17 nm). Stored in `WindowDataset.metadata` for the checkpoint; the trainer must accept σ = 1 as an ablation. |
| Sampling | Confirmed | Batch 64 drawn with replacement from the W windows (W = 60 < 64 for PE). Iteration-based; "epoch" has no meaning. Reproducible via an explicit `torch.Generator`. |
| Scientific framing (PE) | Confirmed | In the training region the MSD exponent is 0.63 (anomalous); it reaches 0.97 only by 10–20 ns. The model must extrapolate through the anomalous→normal crossover from data that lie entirely before it. Step displacements are heavy-tailed (99.99th percentile 1.16 nm ≈ 7σ): a Gaussian step model is wrong for this system. |
| Authors' results (archive) | Confirmed, not used | `gen_data/MSD_gen/PE` holds 30 checkpoints × 5 runs × 7 d_z (index k ↔ t = 100·(k+1) ps); `dc_result_mdgan.txt` gives their PE MD-GAN D = 1.45025e-10 (run 1, checkpoint 190k, reproduced to 5e-5). Checkpoint-to-checkpoint err swings 7 → 27 → 7 % within one run: the paper's numbers are best-of-30. Reproduction target: best checkpoint of a 300k run at ≈5–7 % err and D within ~10 %. Not loaded by the pipeline; only used as expected values here. |
| Other systems | Decided | Nothing PE-specific in code: frame count, dt, box, trajectory count are read from files; n_s is a `WindowDataset` argument (paper values: PE 1, E-VDF 3, PVDF 5, PVAC/E-VAC 7, PAAM/E-AAM 20). `water.py` is water-specific by design (all 28 systems). |
| MDAnalysis cross-check | Decided | Done once during development (agreement 2e-6 nm); it settled the unwrapping design and is not kept. MDAnalysis is not a dependency. The synthetic wrapping tests in `tests/test_water.py` are the standing guarantee. |

We explicitly distinguish reproduced architecture from undocumented implementation assumptions.