"""Train a DDPM.

    python train.py --config configs/mnist.yaml
    python train.py --config configs/mnist.yaml --resume outputs/mnist/ckpt_last.pt
    python train.py --config configs/mnist.yaml --override max_steps=500 batch_size=32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

from diffusion_lab import EMA, GaussianDiffusion, UNet
from diffusion_lab.data import dataset_info, get_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                   help="override top-level config keys, e.g. max_steps=500")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)  # parses ints/floats/bools/lists
    return cfg


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(cfg: dict, device: torch.device):
    channels, _ = dataset_info(cfg["dataset"])
    model = UNet(in_channels=channels, **{k: tuple(v) if isinstance(v, list) else v
                                           for k, v in cfg["model"].items()}).to(device)
    diffusion = GaussianDiffusion(**cfg["diffusion"]).to(device)
    return model, diffusion


def lr_at(step: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * step / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def save_samples(ema_model, diffusion, cfg, channels, out_path: Path):
    shape = (cfg["n_sample"], channels, cfg["image_size"], cfg["image_size"])
    x = diffusion.ddim_sample(ema_model, shape, steps=cfg["sample_steps"])
    save_image((x + 1) / 2, out_path, nrow=int(math.sqrt(cfg["n_sample"])))


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    channels, _ = dataset_info(cfg["dataset"])
    loader = get_dataloader(cfg["dataset"], cfg["data_root"], cfg["batch_size"],
                            cfg["num_workers"], cfg["image_size"])
    model, diffusion = build(cfg, device)
    ema = EMA(model, cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from {args.resume} at step {step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.2f}M  dataset={cfg['dataset']}  "
          f"steps={cfg['max_steps']}  amp={use_amp}")

    log = open(out_dir / "log.jsonl", "a")
    t0, running = time.time(), 0.0
    model.train()
    while step < cfg["max_steps"]:
        for x0, _ in loader:
            if step >= cfg["max_steps"]:
                break
            x0 = x0.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = diffusion.training_loss(model, x0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1
            running += loss.item()

            if step % cfg["log_every"] == 0:
                avg = running / cfg["log_every"]
                rate = cfg["log_every"] / (time.time() - t0)
                print(f"step {step:6d}  loss {avg:.4f}  lr {lr_at(step, cfg):.2e}  {rate:.1f} it/s")
                log.write(json.dumps({"step": step, "loss": avg, "lr": lr_at(step, cfg)}) + "\n")
                log.flush()
                t0, running = time.time(), 0.0

            if step % cfg["sample_every"] == 0 or step == cfg["max_steps"]:
                save_samples(ema.model, diffusion, cfg, channels, out_dir / "samples" / f"step_{step:06d}.png")

            if step % cfg["ckpt_every"] == 0 or step == cfg["max_steps"]:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                            "opt": opt.state_dict(), "step": step, "config": cfg},
                           out_dir / "ckpt_last.pt")

    print(f"done. samples in {out_dir/'samples'}, checkpoint at {out_dir/'ckpt_last.pt'}")


if __name__ == "__main__":
    main()
