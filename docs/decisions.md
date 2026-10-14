# Research decisions and unresolved questions

Status key: Confirmed / Open / Assumption

| Item | Status | Current decision / next check |
| --- | --- | --- |
| M = 64 | Confirmed | 2023 target |
| n_z = 32 | Confirmed | 2023 PE baseline |
| G_z noise input n_r | Open | Keep configurable; verify if given in any supplementary sources or software |
| G_z final affine dimension | Open | Fig. S2(B) has a dimension-label discrepancy; resolve before implementation |
| Mean operation placement | Open | Address with training computational graph |
| WGAN-GP optimizer and update schedule | Open | Address before training |
| Input scaling and weight initialization | Open | Document before training |
| Latent burn-in gradient handling | Open | Address at integration |

We will explicitly distinguish reproduced architecture from undocumented implementation assumptions.
