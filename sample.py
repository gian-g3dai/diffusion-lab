"""Generate an image grid from a trained checkpoint.

    python sample.py --ckpt outputs/mnist/ckpt_last.pt --n 64 --out samples.png
    python sample.py --ckpt outputs/mnist/ckpt_last.pt --sampler ddpm        # slow, full T steps
    python sample.py --ckpt outputs/mnist/ckpt_last.pt --sampler ddim --steps 20
"""
from __future__ import annotations

import argparse
import math

import torch
from torchvision.utils import save_image

from diffusion_lab.data import dataset_info
from train import build, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic)")
    p.add_argument("--no-ema", action="store_true", help="sample from raw weights instead of EMA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="samples.png")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["config"]
    model, diffusion = build(cfg, device)
    model.load_state_dict(ck["model" if args.no_ema else "ema"])
    model.eval()

    torch.manual_seed(args.seed)
    channels, _ = dataset_info(cfg["dataset"])
    shape = (args.n, channels, cfg["image_size"], cfg["image_size"])
    if args.sampler == "ddpm":
        x = diffusion.sample(model, shape, progress=True)
    else:
        x = diffusion.ddim_sample(model, shape, steps=args.steps, eta=args.eta, progress=True)
    save_image((x + 1) / 2, args.out, nrow=int(math.sqrt(args.n)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
