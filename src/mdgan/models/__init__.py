"""The three MD-GAN networks (Kawada et al. 2022 ESI Fig. S2)."""

from mdgan.models.critic import Critic
from mdgan.models.init import apply_init
from mdgan.models.latent_transition import LatentTransitionGenerator
from mdgan.models.trajectory_generator import ResidualAffineBlock, TrajectoryGenerator

__all__ = ["Critic", "LatentTransitionGenerator", "ResidualAffineBlock", "TrajectoryGenerator", "build_models"]


def build_models(config) -> tuple[LatentTransitionGenerator, TrajectoryGenerator, Critic]:
    """(G_z, G_Y, D_G) from one Config, so the shared dimensions cannot disagree:
    G_z and G_Y share latent_dim and noise_dim; G_Y and D_G share block_len and
    n_channels. Weights are initialized from the current torch RNG (seed first), then by the
    scheme config.init ("pytorch" leaves PyTorch's defaults; see mdgan.models.init)."""
    gz = LatentTransitionGenerator(config.latent_dim, config.noise_dim, hidden_dim=config.hidden_dim,
                                   negative_slope=config.negative_slope,
                                   use_running_stats=config.use_running_stats)
    gy = TrajectoryGenerator(config.latent_dim, config.noise_dim, block_len=config.block_len,
                             n_channels=config.n_channels, hidden_dim=config.hidden_dim,
                             negative_slope=config.negative_slope)
    dg = Critic(block_len=config.block_len, n_channels=config.n_channels, hidden_dim=config.hidden_dim,
                negative_slope=config.negative_slope)
    for net in (gz, gy, dg):
        apply_init(net, config.init)
    return gz, gy, dg