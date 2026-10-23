import torch
from torch import nn

from mdgan._util import positive_int


def _draw_device(generator, device):
    """Noise is drawn on the GENERATOR's device and then moved to `device`. One CPU
    generator therefore gives the same random stream whether the model runs on
    CPU, CUDA or MPS (decisions.md, "Randomness")."""
    return generator.device if generator is not None else device


class LatentTransitionGenerator(nn.Module):
    """Stochastic latent transition generator (G_z) for MD-GAN."""

    def __init__(self,
                 latent_dim: int,
                 noise_dim: int,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2,
                 use_running_stats: bool = False):
        super().__init__()

        positive_int("latent_dim", latent_dim)
        positive_int("noise_dim", noise_dim)
        positive_int("hidden_dim", hidden_dim)

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.use_running_stats = use_running_stats

        # [z, r] -> Affine+LReLU -> Affine+LReLU -> Affine+LReLU -> Affine -> BatchNorm -> z'
        self.input_layer = nn.Linear(latent_dim + noise_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.BatchNorm1d(latent_dim, track_running_stats=use_running_stats)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)  # stateless; shared by all three uses

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute the next latent state from the current state and noise."""
        if z.dim() != 2 or r.dim() != 2:
            raise ValueError(
                f"z and r must be rank-2 (B, features); got z{tuple(z.shape)} r{tuple(r.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"z has {z.shape[1]} features; expected latent_dim={self.latent_dim}")
        if r.shape[1] != self.noise_dim:
            raise ValueError(f"r has {r.shape[1]} features; expected noise_dim={self.noise_dim}")
        if z.shape[0] != r.shape[0]:
            raise ValueError(f"batch size mismatch: z has {z.shape[0]}, r has {r.shape[0]}")
        if z.shape[0] < 2 and (self.training or not self.use_running_stats):
            raise ValueError(
                "batch size must be at least 2 when BatchNorm uses batch statistics"
            )

        x = torch.cat((z, r), dim=1)
        h = self.act(self.input_layer(x))
        h = self.act(self.hidden1(h))
        h = self.act(self.hidden2(h))
        return self.output_norm(self.output_layer(h))

    def sample_noise(self, batch: int, *, device=None, dtype=None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """r for `batch` samples: r ~ U[0,1), shape (B, noise_dim).

        Drawn on the generator's device, then moved to `device` (see _draw_device)."""
        r = torch.rand(batch, self.noise_dim, device=_draw_device(generator, device), dtype=dtype,
                       generator=generator)
        return r.to(device) if device is not None else r

    def unroll(self, z0: torch.Tensor, steps: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Evolve z0 for `steps` transitions with fresh r ~ U[0,1) at each step.

        Returns a tensor of shape (steps + 1, B, latent_dim) holding z_0 .. z_steps.
        Gradients flow through every step (full unroll); callers that want to
        cut the graph should detach the result themselves.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        states = [z0]
        z = z0
        for _ in range(steps):
            r = self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=generator)
            z = self(z, r)
            states.append(z)
        return torch.stack(states, dim=0)