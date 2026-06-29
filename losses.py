"""Losses: L1, windowed D-SSIM, and silhouette (mask) loss (WRITEUP B.3).

SSIM is the standard Gaussian-windowed formulation implemented with depthwise
conv2d so it runs on the GPU and is fully differentiable.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def l1_loss(pred, gt):
    return (pred - gt).abs().mean()


def _gaussian_window(size: int, sigma: float, channels: int, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w2d = (g[:, None] * g[None, :])                       # (size,size)
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim(pred, gt, window_size: int = 11, sigma: float = 1.5):
    """SSIM between two (C,H,W) images in [0,1]. Returns a scalar in [-1,1]."""
    C = pred.shape[0]
    pred, gt = pred[None], gt[None]                       # (1,C,H,W)
    w = _gaussian_window(window_size, sigma, C, pred.device)
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, w, padding=pad, groups=C)

    mu1, mu2 = filt(pred), filt(gt)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = filt(pred * pred) - mu1_sq
    sig2 = filt(gt * gt) - mu2_sq
    sig12 = filt(pred * gt) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2))
    return s.mean()


def mask_loss(pred_alpha, gt_mask, kind="l1"):
    if kind == "bce":
        p = pred_alpha.clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(p, gt_mask)
    return (pred_alpha - gt_mask).abs().mean()


def total_loss(pred_img, gt_img, pred_alpha, gt_mask, cfg):
    l1 = l1_loss(pred_img, gt_img)
    dssim = 1.0 - ssim(pred_img, gt_img)
    lm = mask_loss(pred_alpha, gt_mask, cfg.mask_loss)
    loss = (1 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * dssim + cfg.lambda_mask * lm
    return loss, {"l1": l1.item(), "dssim": dssim.item(), "mask": lm.item()}
