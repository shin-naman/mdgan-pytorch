# src/mdgan/models/critic.py
import torch
from torch import nn


class Critic(nn.Module):
    """WGAN-GP critic (D_G) for MD-GAN. Reference: Kawada et al. (2022) ESI Fig. S2(C).

    Scores one 2M-step displacement window (y_k ‖ y_{k+1}). No normalization
    layers anywhere: the gradient penalty is per-sample and BatchNorm would
    couple samples. Pure function of (window, parameters); never detaches.
    """

    def __init__(self,
                 block_len: int = 64,
                 n_channels: int = 3,
                 hidden_dim: int = 2048,
                 negative_slope: float = 0.2):
        super().__init__()

        for name, value in {
            "block_len": block_len,
            "n_channels": n_channels,
            "hidden_dim": hidden_dim,
        }.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        self.block_len = block_len            # M
        self.n_channels = n_channels          # C
        self.hidden_dim = hidden_dim
        self.window_len = 2 * block_len       # 2M = 128 at defaults

        # (B, 2M, C) -> Conv1d(C -> H, kernel 2M, stride 2M) -> (B, H, 1) -> (B, H)
        # With T == 2M exactly, this is one affine map from the flattened window.
        # Kept as Conv1d to match the reference.
        self.input_conv = nn.Conv1d(n_channels, hidden_dim,
                                    kernel_size=self.window_len, stride=self.window_len)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)  # stateless; shared by all four uses

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: (B, 2M, C) per-step displacements. Returns scores of shape (B,)."""
        if window.dim() != 3:
            raise ValueError(f"window must be rank-3 (B, 2M, C); got {tuple(window.shape)}")
        if window.shape[1:] != (self.window_len, self.n_channels):
            raise ValueError(
                f"window has per-sample shape {tuple(window.shape[1:])}; "
                f"expected ({self.window_len}, {self.n_channels})"
            )

        h = self.input_conv(window.transpose(1, 2))    # (B, H, 1)
        h = self.act(h.squeeze(-1))                    # (B, H)
        h = self.act(self.hidden1(h))
        h = self.act(self.hidden2(h))
        h = self.act(self.hidden3(h))
        return self.output_layer(h).squeeze(-1)        # (B,)