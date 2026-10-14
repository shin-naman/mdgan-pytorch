# Architecture contract (draft)

## Scope
Target: water center-of-mass displacement sequences in polyethylene, informed by Kawada et al. (2023). Use 2022 supplementary Fig. S2 as the architecture reference.

### Baseline
- Displacement block length: M = 64
- Spatial components per step: 3
- Latent state dimension: n_z = 32
- Training sample: two adjacent blocks, shape (B, 128, 3)

### Latent transition generator G_z
- Inputs: latent vector z_i with shape (B, n_z); uniform random vector r_i with shape (B, n_r)
- Output: next latent vector z_(i+1) with shape (B, n_z)
- Diagram: concatenate z_i and r_i; three Affine + LeakyReLU transformations to width 2048, then a final Affine and BatchNorm.
- Exact n_r and final Affine/BatchNorm dimensional interpretation are open questions; see decisions.md.
- No encoder or VAE is part of the reference architecture.

### Other components (later)
- G_Y: latent-to-64-step displacement block generator; also takes uniform and Gaussian noise.
- D_G: WGAN-GP critic on 128-step real/fake displacement sequences.

### Sources
- Kawada et al. (2022), supplementary material, Figure S2.
- Kawada et al. (2023), Sections MD-GAN Model and Method.
