import torch
from torch import nn

from mdgan._util import positive_int
from mdgan.models.latent_transition import _draw_device


class ResidualAffineBlock(nn.Module):
    """x -> x + Affine(LReLU(Affine(x))). Width is preserved; the skip is an identity.
    Reference: Kawada et al. (2022) ESI Fig. S2(A), the bracketed pair of affines."""

    def __init__(self, width: int, negative_slope: float):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))

class TrajectoryGenerator(nn.Module):
    """Latent-to-trajectory generator (G_Y) for MD-GAN. Reference: Kawada et al. (2022) ESI Fig. S2(A)."""

    def __init__(self,
                 latent_dim: int,
                 noise_dim: int,
                 block_len: int = 64,
                 n_channels: int = 3,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2):
        super().__init__()

        positive_int("latent_dim", latent_dim)
        positive_int("noise_dim", noise_dim)
        positive_int("block_len", block_len)
        positive_int("n_channels", n_channels)
        positive_int("hidden_dim", hidden_dim)

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.block_len = block_len        # M
        self.n_channels = n_channels      # C: 3 for single-particle, 9 for three-particle
        self.hidden_dim = hidden_dim
        self.flat_dim = block_len * n_channels   # 192 at defaults
        self.noise_mix = nn.Conv1d(n_channels, n_channels, kernel_size=1, stride=1)

        self.input_layer = nn.Linear(noise_dim + latent_dim + self.flat_dim, hidden_dim)
        self.input_act = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = ResidualAffineBlock(hidden_dim, negative_slope)
        self.post_act = nn.LeakyReLU(negative_slope=negative_slope)
        self.output_layer = nn.Linear(hidden_dim, self.flat_dim)
    
    def _mix_noise(self, u: torch.Tensor) -> torch.Tensor:
        """(B, M, C) -> Conv1d over channels -> (B, M*C), flattened step-major."""
        batch = u.shape[0]
        mixed = self.noise_mix(u.transpose(1, 2)).transpose(1, 2)   # (B, M, C)
        return mixed.reshape(batch, self.flat_dim)
    
    def forward(self, z: torch.Tensor, r_prime: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Decode one M-step displacement block.

        z: (B, latent_dim); r_prime: (B, noise_dim) ~ U[0,1); u: (B, M, C) ~ N(0,1).
        Returns y: (B, M, C), per-step displacements.
        """
        if z.dim() != 2 or r_prime.dim() != 2 or u.dim() != 3:
            raise ValueError(
                "expected z (B, latent_dim), r_prime (B, noise_dim), u (B, M, C); "
                f"got z{tuple(z.shape)} r_prime{tuple(r_prime.shape)} u{tuple(u.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"z has {z.shape[1]} features; expected latent_dim={self.latent_dim}")
        if r_prime.shape[1] != self.noise_dim:
            raise ValueError(f"r_prime has {r_prime.shape[1]} features; expected noise_dim={self.noise_dim}")
        if u.shape[1:] != (self.block_len, self.n_channels):
            raise ValueError(
                f"u has per-sample shape {tuple(u.shape[1:])}; expected ({self.block_len}, {self.n_channels})"
            )
        if not (z.shape[0] == r_prime.shape[0] == u.shape[0]):
            raise ValueError(
                f"batch size mismatch: z {z.shape[0]}, r_prime {r_prime.shape[0]}, u {u.shape[0]}"
            )

        batch = z.shape[0]
        u_mixed = self._mix_noise(u)                              # (B, M*C)
        x = torch.cat((r_prime, z, u_mixed), dim=1)               # (B, n_r + n_z + M*C)
        h = self.input_act(self.input_layer(x))                   # (B, H)
        h = self.post_act(self.residual(h))                       # (B, H)
        y = self.output_layer(h)                                  # (B, M*C)
        return y.reshape(batch, self.block_len, self.n_channels)  # step-major, matches _mix_noise

    def sample_noise(self, batch: int, *, device=None, dtype=None,
                     generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(r_prime, u) for `batch` samples: r_prime ~ U[0,1), u ~ N(0,1).

        Drawn on the generator's device, then moved to `device` (see _draw_device)."""
        at = _draw_device(generator, device)
        r_prime = torch.rand(batch, self.noise_dim, device=at, dtype=dtype, generator=generator)
        u = torch.randn(batch, self.block_len, self.n_channels, device=at, dtype=dtype, generator=generator)
        if device is not None:
            r_prime, u = r_prime.to(device), u.to(device)
        return r_prime, u

    def generate(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Decode z with freshly sampled noise. Returns (B, M, C)."""
        r_prime, u = self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=generator)
        return self(z, r_prime, u)