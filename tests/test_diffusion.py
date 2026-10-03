import torch

from diffusion_lab import EMA, GaussianDiffusion, UNet
from diffusion_lab.diffusion import make_beta_schedule


def test_schedules_are_valid():
    for name in ("linear", "cosine"):
        betas = make_beta_schedule(name, 100)
        assert betas.shape == (100,)
        assert (betas > 0).all() and (betas < 1).all()


def test_q_sample_matches_closed_form():
    d = GaussianDiffusion(timesteps=100)
    x0 = torch.randn(4, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 33, 66, 99])
    xt = d.q_sample(x0, t, noise)
    # Inverting the forward process with the true noise recovers x0 exactly.
    assert torch.allclose(d.predict_x0(xt, t, noise), x0, atol=1e-5)


def test_unet_shapes():
    for channels, size in ((1, 28), (3, 32)):
        model = UNet(in_channels=channels, base_channels=16, channel_mults=(1, 2), attn_levels=(1,),
                     num_res_blocks=1)
        x = torch.randn(2, channels, size, size)
        t = torch.randint(0, 10, (2,))
        assert model(x, t).shape == x.shape


def test_training_step_and_samplers_run():
    torch.manual_seed(0)
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2), attn_levels=(), num_res_blocks=1)
    d = GaussianDiffusion(timesteps=50)
    ema = EMA(model, decay=0.9)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand(4, 1, 8, 8) * 2 - 1

    loss = d.training_loss(model, x0)
    loss.backward()
    opt.step()
    ema.update(model)
    assert torch.isfinite(loss)

    ddpm = d.sample(ema.model, (2, 1, 8, 8))
    ddim = d.ddim_sample(ema.model, (2, 1, 8, 8), steps=4)
    assert ddpm.shape == ddim.shape == (2, 1, 8, 8)
    assert torch.isfinite(ddpm).all() and torch.isfinite(ddim).all()


def test_ddim_is_deterministic_at_eta_zero():
    model = UNet(in_channels=1, base_channels=16, channel_mults=(1, 2), attn_levels=(), num_res_blocks=1).eval()
    d = GaussianDiffusion(timesteps=50)
    torch.manual_seed(1)
    a = d.ddim_sample(model, (2, 1, 8, 8), steps=4)
    torch.manual_seed(1)
    b = d.ddim_sample(model, (2, 1, 8, 8), steps=4)
    assert torch.equal(a, b)
