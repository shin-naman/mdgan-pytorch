# Architecture contract

Reference: Kawada et al. (2022), ESI Fig. S2, for all three networks. Target and
hyperparameters from Kawada et al. (2023). Every row of Fig. S2 maps to one named
attribute in the corresponding module; nothing exists in a module that is not in the figure.
Decisions that the figure leaves open are recorded in `decisions.md`.

## Shared conventions
- M = 64 steps per block, C = 3 channels (xyz) for single-particle input; `n_channels` is a constructor argument in every module.
- Tensors are (B, T, C). Conv1d calls transpose to (B, C, T) internally and back.
- Flatten/reshape between (T, C) and T·C is step-major: index = t·C + c.
- Data are per-step displacements of the water COM, scaled by a single scalar σ (see `decisions.md`); positions are their cumsum from the origin.
- LeakyReLU slope 0.2 everywhere. No normalization layers except G_z's output BatchNorm.
- No `forward` consumes the global RNG; noise is passed explicitly or drawn through an explicit `torch.Generator`.

## G_z — `LatentTransitionGenerator` (latent_transition.py)
Inputs: z_i (B, n_z), r_i (B, n_r) ~ U[0,1). Output: z_{i+1} (B, n_z).

    [z, r] → input_layer (n_z+n_r → 2048) → act
           → hidden1 (2048 → 2048) → act
           → hidden2 (2048 → 2048) → act
           → output_layer (2048 → n_z)
           → output_norm BatchNorm1d(n_z, batch statistics in train and eval)

Helpers: `sample_noise(batch)` → r; `unroll(z0, steps)` → (steps+1, B, n_z), full BPTT.
Defaults: n_z = 32, n_r = 32, c = 7 burn-in steps before decoding.

## G_Y — `TrajectoryGenerator` (trajectory_generator.py)
Inputs: z (B, n_z), r' (B, n_r) ~ U[0,1), u (B, M, C) ~ N(0,1). Output: y (B, M, C).

    u → noise_mix Conv1d(C → C, k=1, s=1) → flatten M·C          (same C×C affine at every step)
    [r', z, u'] → input_layer (n_r+n_z+M·C → 2048) → input_act
                → residual: x + fc2(act(fc1(x)))                  (ResidualAffineBlock, identity skip)
                → post_act
                → output_layer (2048 → M·C) → reshape (M, C)

Helpers: `sample_noise(batch)` → (r', u); `generate(z)` → y.

## D_G — `Critic` (critic.py)
Input: one window (B, 2M, C) = y_k ‖ y_{k+1}. Output: Wasserstein score (B,).

    window → input_conv Conv1d(C → 2048, k=2M, s=2M) → squeeze → act   (one affine over the flattened window)
           → hidden1 → act → hidden2 → act → hidden3 → act
           → output_layer (2048 → 1) → squeeze(-1)

Window length must equal 2M exactly. No normalization: the gradient penalty is per-sample.

## Ops (ops.py)
`mean_operation(x)`: subtract the batch mean per step per axis (2023 eq. 9). Applied by the trainer to the generated window before the critic and before gradient-penalty interpolation.

## Training wiring (to be implemented)
Per generator iteration: 64 chains z_0 ~ U[0,1)^{n_z}; `unroll` to c+K; pick j ∈ [c, c+K−1];
fake = mean_operation(cat(G_Y.generate(z_j), G_Y.generate(z_{j+1}))); critic sees 64 real vs 64 fake windows.
WGAN-GP: 5 critic updates (fake detached) per generator update (gradient flows through the critic into both generators).
Estimation: 8192 chains, burn in c, then 125 alternating G_z → G_Y steps for 80 ns at n_s = 1.

## Sources
- Kawada, Endo, Yuhara, Yasuoka. Soft Matter 2022, 18, 8446 (ESI Fig. S2, §S1–S3).
- Kawada, Endo, Yasuoka, Kojima, Matubayasi. J. Chem. Inf. Model. 2023, 63, 76 (MD-GAN Model, Method, eq. 9).
- Endo, Tomobe, Yasuoka. AAAI 2018 (original MD-GAN; U-Net G_Y and D_z, both superseded).