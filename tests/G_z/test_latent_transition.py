import torch

from mdgan.models.latent_transition import LatentTransitionGenerator


def test_gz_output_shape():
    batch_size = 64
    latent_dim = 32
    noise_dim = 32  

    model = LatentTransitionGenerator(
        latent_dim=latent_dim,
        noise_dim=noise_dim,
    )

    z = torch.rand(batch_size, latent_dim)
    r = torch.rand(batch_size, noise_dim)

    z_next = model(z, r)

    assert z_next.shape == (batch_size, latent_dim)
    assert torch.isfinite(z_next).all()


def test_gz_gradients():
    model = LatentTransitionGenerator(
        latent_dim=32,
        noise_dim=32,
    )

    z = torch.rand(64, 32)
    r = torch.rand(64, 32)

    z_next = model(z, r)

    # Use an arbitrary differentiable objective to test backpropagation.
    weights = torch.randn_like(z_next)
    loss = (z_next * weights).mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
