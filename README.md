# diffusion-lab

A minimal, readable implementation of image diffusion models in plain PyTorch:
DDPM training ([Ho et al. 2020](https://arxiv.org/abs/2006.11239)) with DDPM and DDIM
([Song et al. 2020](https://arxiv.org/abs/2010.02502)) sampling. No framework, no
`diffusers`; the core is ~500 lines you can read in one sitting.

Works out of the box on MNIST, Fashion-MNIST and CIFAR-10. MNIST gives recognisable
digits in ~10 minutes on a consumer GPU.

```
train:    x_0 ──► add noise at random t ──► x_t ──► UNet ε_θ(x_t, t) ──► MSE(ε_θ, ε)

sample:   x_T ~ N(0, I) ──► for t = T-1 … 0:  x_{t-1} = f(x_t, ε_θ(x_t, t))  ──► x_0
```

## Layout

```
diffusion_lab/
  diffusion.py   forward process, ε-prediction loss, DDPM + DDIM samplers, β schedules
  unet.py        small UNet (ResBlocks + time embedding + self-attention)
  ema.py         exponential moving average of weights (used for sampling)
  data.py        dataset loaders, images scaled to [-1, 1]
train.py         config-driven training loop with AMP, EMA, warmup/cosine LR, resume
sample.py        generate a grid from a checkpoint
configs/         mnist.yaml, fashion_mnist.yaml, cifar10.yaml
tests/           shape checks, closed-form forward-process check, sampler determinism
scripts/         plot_loss.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # 5 quick CPU tests
```

## Train

```bash
python train.py --config configs/mnist.yaml
```

Writes to `outputs/mnist/`: a `samples/step_XXXXXX.png` grid every `sample_every`
steps, `ckpt_last.pt`, `log.jsonl`, and a copy of the resolved config. Any top-level
config key can be overridden from the command line:

```bash
python train.py --config configs/mnist.yaml --override max_steps=5000 batch_size=64
python train.py --config configs/mnist.yaml --resume outputs/mnist/ckpt_last.pt
python train.py --config configs/mnist.yaml --device cpu --override num_workers=0   # smoke test
```

Device is picked automatically (CUDA → MPS → CPU). Mixed precision is used on CUDA only.

## Sample

```bash
python sample.py --ckpt outputs/mnist/ckpt_last.pt --n 64 --out samples.png            # DDIM, 50 steps
python sample.py --ckpt outputs/mnist/ckpt_last.pt --sampler ddim --steps 20 --eta 0.0
python sample.py --ckpt outputs/mnist/ckpt_last.pt --sampler ddpm                       # full 1000 steps
```

By default samples come from the EMA weights, which are much cleaner than the raw ones
(pass `--no-ema` to compare).

## What the pieces do

- **β schedule** (`make_beta_schedule`): `linear` is the DDPM default; `cosine`
  (Nichol & Dhariwal) keeps more signal at high t and tends to work better on small images.
- **Forward process** (`q_sample`): `x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`, closed form,
  no iteration needed.
- **Loss** (`training_loss`): sample `t ~ U{0..T-1}`, `ε ~ N(0,I)`, predict ε from `x_t`.
  Plain MSE, no variational bound, no learned variance.
- **DDPM sampling** (`p_sample`): predict `x_0` from `ε_θ`, clip to [-1, 1], then take the
  mean of the posterior `q(x_{t-1} | x_t, x_0)` plus noise with the posterior variance.
- **DDIM sampling** (`ddim_sample`): same model, but skips timesteps. With `eta=0` it's
  deterministic, so a fixed seed gives a fixed image.
- **UNet**: channel multipliers per resolution level, two ResBlocks per level, attention
  at the levels listed in `attn_levels`, sinusoidal time embedding added inside every
  ResBlock.

## Config reference (`configs/mnist.yaml`)

| key | meaning |
|---|---|
| `model.base_channels`, `channel_mults` | width per level; `(1,2,2)` with 64 → 64/128/128 channels at 28/14/7 px |
| `model.attn_levels` | which levels get self-attention (index into `channel_mults`) |
| `diffusion.timesteps`, `schedule` | T and the β schedule (`linear` / `cosine`) |
| `lr`, `warmup_steps`, `max_steps` | linear warmup, then cosine decay to 10% of peak |
| `ema_decay` | 0.999 for short runs, 0.9999 for long ones |
| `amp` | float16 autocast + GradScaler on CUDA |
| `sample_steps` | DDIM steps for the periodic preview grids |

## Ideas to extend

Class-conditional generation with classifier-free guidance, learned variance (Improved
DDPM), v-prediction, latent diffusion with a small autoencoder, FID evaluation.

## License

MIT
