import torch
from torch import nn


class LatentTransitionGenerator(nn.Module):
    """Stochastic latent transition generator (G_z) for MD-GAN."""
    
    def __init__(self, 
                 latent_dim: int, 
                 noise_dim: int,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2,
                 validate_finite=True):
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
        self.validate_finite = validate_finite

        self.network = nn.Sequential(
            nn.Linear(latent_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),

            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute the next latent state from the current state and noise."""
        if self.validate_finite:
            for name, tensor in (("z", z), ("r", r)):
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f"Input {name} contains NaN or inf values")

        x = torch.cat((z, r), dim=1)
        return self.network(x)
