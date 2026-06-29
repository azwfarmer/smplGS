"""Render a moving human from a trained checkpoint -> mp4/gif (WRITEUP B.7).

Animates the held-out ZJU pose sequence from a fixed camera, optionally adding a
global yaw to Rh each frame so the body turns ("orbit" effect without moving the
camera). Falls back to a GIF if ffmpeg is unavailable.

Run:  python demo.py --checkpoint checkpoints/ckpt_010000.pt   (from the human/ directory)
"""
from __future__ import annotations
import argparse
import math
import torch
import numpy as np
import imageio.v2 as imageio

from config import Config
from smpl_model import SMPLBody, rodrigues
from geometry import matrix_to_axis_angle
from gaussian_model import GaussianModel
from deform import Deformer
from rasterizer import render
from dataset_zju import ZJUDataset
from train import make_frame


def load_checkpoint(path, cfg):
    ck = torch.load(path, map_location=cfg.device)
    init = {k: v.to(cfg.device) for k, v in ck["gaussians"].items()}
    gaussians = GaussianModel(init, ck["sh_degree"], cfg.pose_feat_dim).to(cfg.device)
    deformer = Deformer(cfg).to(cfg.device)
    deformer.load_state_dict(ck["deformer"])
    deformer.eval()
    betas = ck["betas"].to(cfg.device)
    return gaussians, deformer, betas


def yaw_into_Rh(Rh, angle, device):
    """Compose a world-up yaw with the existing global orient Rh."""
    c, s = math.cos(angle), math.sin(angle)
    R_yaw = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]],
                         dtype=torch.float32, device=device)
    return matrix_to_axis_angle(R_yaw @ rodrigues(Rh.to(device)))


@torch.no_grad()
def run(cfg: Config, ckpt: str, out: str, orbit: bool, fps: int = 50,
        frames_split: str = "holdout", stride: int = 1):
    dev = cfg.device
    smpl = SMPLBody(cfg.smpl_model_path).to(dev)
    gaussians, deformer, betas = load_checkpoint(ckpt, cfg)
    # "holdout" = last 10% of frames (unseen poses); "all" = whole captured take.
    data = ZJUDataset(cfg, split=("all" if frames_split == "all" else "demo"))
    bg = torch.tensor(cfg.bg_color, device=dev)

    # one fixed camera for the whole clip
    cam = data[0]["camera"]

    # data.samples are (frame, camera) pairs, but the demo pins ONE camera and the
    # SMPL pose is camera-independent -> render each *pose* once. Otherwise every
    # pose repeats once per camera (~23x on ZJU) and the clip advances at ~1 fps.
    seen, idxs = set(), []
    for i, (fi, _) in enumerate(data.samples):
        if fi not in seen:
            seen.add(fi)
            idxs.append(i)
    idxs = idxs[::max(1, stride)]     # subsample poses to tune clip length

    frames = []
    n = len(idxs)
    for k, i in enumerate(idxs):
        sample = data[i]
        sample["camera"] = cam
        frame = make_frame(smpl, betas, sample, dev)
        if orbit:
            frame["Rh"] = yaw_into_Rh(sample["smpl"]["Rh"], 2 * math.pi * k / n, dev)
        means3D, cov3D, colors = deformer(gaussians, frame)
        out_r = render(means3D, cov3D, colors, gaussians.get_opacity, cam, bg, cfg)
        img = (out_r["image"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(img)
        print(f"\rrendered {k + 1}/{n}", end="")
    print()

    try:
        imageio.mimsave(out, frames, fps=fps)
        print(f"[demo] wrote {out}")
    except Exception as e:  # noqa  (ffmpeg missing -> GIF fallback)
        gif = out.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(gif, frames, duration=1.0 / fps)
        print(f"[demo] mp4 failed ({e}); wrote {gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--orbit", action="store_true", help="add a turning global yaw")
    ap.add_argument("--fps", type=int, default=50,
                    help="output frame rate; ZJU-MoCap is captured at 50 fps, so "
                         "50 plays the held-out motion back at real-time speed")
    ap.add_argument("--frames", choices=["holdout", "all"], default="holdout",
                    help="holdout=last 10%% unseen poses (default); all=whole take")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth pose (subsample to tune clip length)")
    args = ap.parse_args()
    run(Config(), args.checkpoint, args.out, args.orbit, args.fps,
        frames_split=args.frames, stride=args.stride)


if __name__ == "__main__":
    main()
