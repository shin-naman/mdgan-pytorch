import torch
from torch import nn

from mdgan._util import positive_int


def _draw_device(generator, device):
    """Noise is drawn on the GENERATOR's device and then moved to `device`. One CPU
    generator therefore gives the same random stream whether the model runs on
    CPU, CUDA or MPS (decisions.md, "Randomness")."""
    return generator.device if generator is not None else device


NORM_POSITIONS = ("output", "hidden")


class LatentTransitionGenerator(nn.Module):
    """Stochastic latent transition generator (G_z) for MD-GAN.

    Paper default (all batch-1 options off):
        [z, r] -> Affine+LReLU -> Affine+LReLU -> Affine+LReLU -> Affine(H -> n_z) -> BatchNorm(n_z) -> z'
    with r ~ U[0,1)^noise_dim.

    Reproduction probes (Config fields of the same name; paper behaviour at their defaults):
      gz_additive_noise_std = s > 0   input z + eps, eps ~ N(0, s^2 I_{n_z}) fresh each transition;
                                      r leaves G_z (first Affine is n_z -> H); noise_dim then sizes
                                      only G_Y's r'.
      gz_dropout = p > 0              dropout after each of the three hidden LReLUs, training only,
                                      inverted scaling 1/(1-p). Masks are drawn from the explicit
                                      generator: in training mode sample_noise appends 3*H uniforms
                                      to the noise vector and forward keeps a unit where u >= p.
      gz_norm_position = "hidden"     tail Affine(H -> n_z) -> BatchNorm(n_z) replaced by
                                      Affine(H -> H) -> BatchNorm(H) -> Affine(H -> n_z); no activation
                                      after the new Affine or the BatchNorm, no output normalisation.

    The forward pass is trunk -> norm -> head, exposed as _trunk(z, r) (pre-normalisation tensor),
    norm (the BatchNorm module, wherever it sits) and _head(normalised, r), so analysis tools can
    replicate a step without assuming where the normalisation is.
    """

    def __init__(self,
                 latent_dim: int,
                 noise_dim: int,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2,
                 use_running_stats: bool = False,
                 gz_additive_noise_std: float = 0.0,
                 gz_dropout: float = 0.0,
                 gz_norm_position: str = "output"):
        super().__init__()

        positive_int("latent_dim", latent_dim)
        positive_int("noise_dim", noise_dim)
        positive_int("hidden_dim", hidden_dim)
        if not gz_additive_noise_std >= 0.0:
            raise ValueError("gz_additive_noise_std must be >= 0")
        if not 0.0 <= gz_dropout < 1.0:
            raise ValueError("gz_dropout must be in [0, 1)")
        if gz_norm_position not in NORM_POSITIONS:
            raise ValueError(f"gz_norm_position must be one of {NORM_POSITIONS}")

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.use_running_stats = use_running_stats
        self.gz_additive_noise_std = float(gz_additive_noise_std)
        self.gz_dropout = float(gz_dropout)
        self.gz_norm_position = gz_norm_position

        in_features = latent_dim if self.additive else latent_dim + noise_dim
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        if gz_norm_position == "output":
            self.output_layer = nn.Linear(hidden_dim, latent_dim)
            self.output_norm = nn.BatchNorm1d(latent_dim, track_running_stats=use_running_stats)
        else:
            self.tail_layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=use_running_stats)
            self.output_layer = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)  # stateless; shared by all three uses

    # --- structure ------------------------------------------------------------------------------
    @property
    def additive(self) -> bool:
        return self.gz_additive_noise_std > 0

    @property
    def norm(self) -> nn.BatchNorm1d:
        return self.output_norm if self.gz_norm_position == "output" else self.hidden_norm

    @property
    def _base_noise_width(self) -> int:
        return self.latent_dim if self.additive else self.noise_dim

    @property
    def _uses_masks(self) -> bool:
        return self.gz_dropout > 0 and self.training

    @property
    def noise_width(self) -> int:
        """Columns of the noise vector forward() expects in the current mode."""
        return self._base_noise_width + (3 * self.hidden_dim if self._uses_masks else 0)

    def _drop(self, h: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        if u is None:
            return h
        return h * (u >= self.gz_dropout).to(h.dtype) / (1.0 - self.gz_dropout)

    def _trunk(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Pre-normalisation tensor: (B, n_z) for norm position "output", (B, H) for "hidden"."""
        nb = self._base_noise_width
        x = z + r[:, :nb] if self.additive else torch.cat((z, r[:, :nb]), dim=1)
        masks = [None, None, None]
        if self._uses_masks:
            h_dim = self.hidden_dim
            masks = [r[:, nb + i * h_dim: nb + (i + 1) * h_dim] for i in range(3)]
        h = self._drop(self.act(self.input_layer(x)), masks[0])
        h = self._drop(self.act(self.hidden1(h)), masks[1])
        h = self._drop(self.act(self.hidden2(h)), masks[2])
        return self.output_layer(h) if self.gz_norm_position == "output" else self.tail_layer(h)

    def _head(self, normed: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """z' from the normalised tensor."""
        return normed if self.gz_norm_position == "output" else self.output_layer(normed)

    # --- interface ------------------------------------------------------------------------------
    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute the next latent state from the current state and noise."""
        if z.dim() != 2 or r.dim() != 2:
            raise ValueError(
                f"z and r must be rank-2 (B, features); got z{tuple(z.shape)} r{tuple(r.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"z has {z.shape[1]} features; expected latent_dim={self.latent_dim}")
        if r.shape[1] != self.noise_width:
            if self.additive or self.gz_dropout > 0:
                raise ValueError(f"r has {r.shape[1]} features; expected {self.noise_width} in "
                                 f"{'training' if self.training else 'eval'} mode (draw it with sample_noise)")
            raise ValueError(f"r has {r.shape[1]} features; expected noise_dim={self.noise_dim}")
        if z.shape[0] != r.shape[0]:
            raise ValueError(f"batch size mismatch: z has {z.shape[0]}, r has {r.shape[0]}")
        if z.shape[0] < 2 and (self.training or not self.use_running_stats):
            raise ValueError(
                "batch size must be at least 2 when BatchNorm uses batch statistics"
            )
        return self._head(self.norm(self._trunk(z, r)), r)

    def sample_noise(self, batch: int, *, device=None, dtype=None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        """Noise for `batch` samples, in the layout forward() expects in the current mode:
        r ~ U[0,1) of shape (B, noise_dim) (paper), or eps ~ N(0, s^2) of shape (B, latent_dim)
        (gz_additive_noise_std = s > 0); in training mode with gz_dropout > 0, followed by 3*H
        mask uniforms U[0,1). Drawn on the generator's device, then moved to `device`."""
        at = _draw_device(generator, device)
        if self.additive:
            r = torch.randn(batch, self.latent_dim, device=at, dtype=dtype, generator=generator)
            r = r * self.gz_additive_noise_std
        else:
            r = torch.rand(batch, self.noise_dim, device=at, dtype=dtype, generator=generator)
        if self._uses_masks:
            u = torch.rand(batch, 3 * self.hidden_dim, device=at, dtype=dtype, generator=generator)
            r = torch.cat((r, u), dim=1)
        return r.to(device) if device is not None else r

    def unroll(self, z0: torch.Tensor, steps: int,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """Evolve z0 for `steps` transitions with fresh noise (sample_noise) at each step.

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
