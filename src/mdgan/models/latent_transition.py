import torch
from torch import nn


class LatentTransitionGenerator(nn.Module):
    """Stochastic latent transition generator (G_z) for MD-GAN."""
    
    def __init__(self, 
                 latent_dim: int, 
                 noise_dim: int,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2,
                 use_running_stats: bool = False,
                 validate_finite: bool = False):
        super().__init__()

        for name, value in {
            "latent_dim": latent_dim,
            "noise_dim": noise_dim,
            "hidden_dim": hidden_dim,
        }.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.validate_finite = validate_finite
        self.use_running_stats = use_running_stats

        self.network = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, track_running_stats=use_running_stats)
        )

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
        
        if self.validate_finite:
            for name, tensor in (("z", z), ("r", r)):
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f"Input {name} contains NaN or inf values")

        x = torch.cat((z, r), dim=1)
        return self.network(x)

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
            r = torch.rand(z.shape[0], self.noise_dim,
                           device=z.device, dtype=z.dtype, generator=generator)
            z = self(z, r)
            states.append(z)
        return torch.stack(states, dim=0)