"""A small UNet for epsilon-prediction, in the style of Ho et al. 2020 (DDPM).

    x_t, t  ──►  time embedding ──┐
                                  ▼
    conv ─► [ResBlock ×2 (+attn)] ─► down ─► ... ─► mid (Res, Attn, Res) ─► ... ─► up ─► conv ─► eps

Every ResBlock receives the time embedding (added after the first conv), and the
up path concatenates the matching down-path activations (the "U" skip connections).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps, as in the Transformer paper."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)  # (b, hw, hw)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 channel_mults: tuple[int, ...] = (1, 2, 2), attn_levels: tuple[int, ...] = (1,),
                 num_res_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        time_dim = base_channels * 4
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path. We record channel counts of every skip we emit so the up path can mirror them.
        self.down = nn.ModuleList()
        skip_chs = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                if level in attn_levels:
                    blocks.append(AttnBlock(out_ch))
                self.down.append(blocks)
                ch = out_ch
                skip_chs.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, time_dim, dropout), AttnBlock(ch), ResBlock(ch, ch, time_dim, dropout)])

        # Up path, mirror of the down path, consuming skips in reverse.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                blocks = nn.ModuleList([ResBlock(ch + skip_chs.pop(), out_ch, time_dim, dropout)])
                if level in attn_levels:
                    blocks.append(AttnBlock(out_ch))
                self.up.append(blocks)
                ch = out_ch
            if level != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(ch, ch, 3, padding=1)]))

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv2d(ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base_channels))

        h = self.in_conv(x)
        skips = [h]
        for blocks in self.down:
            if isinstance(blocks[0], ResBlock):
                h = blocks[0](h, temb)
                if len(blocks) > 1:
                    h = blocks[1](h)
            else:  # downsample conv
                h = blocks[0](h)
            skips.append(h)

        h = self.mid[0](h, temb)
        h = self.mid[1](h)
        h = self.mid[2](h, temb)

        for blocks in self.up:
            if isinstance(blocks[0], ResBlock):
                h = blocks[0](torch.cat([h, skips.pop()], dim=1), temb)
                if len(blocks) > 1:
                    h = blocks[1](h)
            else:  # upsample + conv
                h = blocks[1](blocks[0](h))

        return self.out(h)
