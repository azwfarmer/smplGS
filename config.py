"""Configuration for SMPL-sampled Gaussian Splatting.

One dataclass holds every knob: paths, device, resolution, the Gaussian set,
learning rates, the densification schedule, and feature flags. `train.py` and
`demo.py` both build a `Config` and pass it around explicitly so there is no
hidden global state.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ---- paths (user-supplied data) -------------------------------------
    data_root: str = "/bigdata/users/aaronzw/gsdata/zju-mocap/CoreView_377"   # ZJU subject folder
    smpl_model_path: str = "/bigdata/users/aaronzw/gsdata/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"  # .npz or .pkl
    smpl_gender: str = "neutral"
    out_dir: str = "/bigdata/users/aaronzw/gsdata/checkpoints"

    # ---- device / dtype --------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dataset ---------------------------------------------------------
    image_scale: float = 0.5          # resize factor applied to images + intrinsics
    train_views: tuple = ()           # () = all cameras; else a subset of cam indices
    train_frames: int = 100           # number of frames to train on (from the start)
    demo_frames_holdout: float = 0.1  # last 10% of frames are held out for the demo
    t_in_mm: bool = True              # ZJU camera T often in millimetres -> /1000
    bg_color: tuple = (0.0, 0.0, 0.0) # render/composite background

    # ---- Gaussian set ----------------------------------------------------
    n_gaussians: int = 50_000         # target count sampled on the rest mesh
    scale_normal_ratio: float = 0.35  # s_n = ratio * s_t  (out-of-plane thinness)
    init_opacity: float = 0.1         # initial alpha (sigmoid space set from this)
    sh_degree: int = 3                # spherical-harmonics degree for view-dep colour

    # ---- non-rigid + pose-colour MLPs -----------------------------------
    use_nonrigid: bool = True         # pose-conditioned per-Gaussian (dmu, ds, dq)
    use_pose_color: bool = True       # pose-conditioned per-Gaussian RGB offset
    posenc_freqs: int = 4             # positional-encoding bands for canonical xyz
    mlp_width: int = 128
    pose_feat_dim: int = 8            # per-Gaussian appearance latent (pose colour)

    # ---- rasterizer (CUDA diff-gaussian-rasterization) -------------------
    tile: int = 32                    # inert: CUDA kernel uses fixed 16x16 tiles
    cov2d_eps: float = 0.3            # inert: CUDA kernel adds its own 0.3 px^2 low-pass
    near: float = 0.05                # near plane for the projection matrix (metres)

    # ---- losses ----------------------------------------------------------
    lambda_dssim: float = 0.2
    lambda_mask: float = 0.1
    mask_loss: str = "l1"             # "l1" or "bce"

    # ---- optimisation ----------------------------------------------------
    iterations: int = 10_000
    lr_xyz_init: float = 1.6e-4
    lr_xyz_final: float = 1.6e-6      # exponential decay target for xyz
    lr_scale: float = 5e-3
    lr_quat: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20.0 # higher-order SH learns slower (3DGS)
    lr_mlp: float = 1e-3

    # ---- adaptive density control (3DGS) ---------------------------------
    densify: bool = True
    densify_from_iter: int = 500
    densify_until_iter: int = 7_000
    densify_interval: int = 100        # densify+prune every k iters
    opacity_reset_interval: int = 3_000
    densify_grad_threshold: float = 2e-4   # NDC view-space mean-grad threshold
    percent_dense: float = 0.01        # clone-vs-split scale cutoff (frac of extent)
    min_opacity: float = 5e-3          # prune below this alpha
    max_screen_size: int = 20          # prune if radius (px) exceeds this (0=off)

    # ---- logging ---------------------------------------------------------
    log_interval: int = 50
    preview_interval: int = 500
    ckpt_interval: int = 2_000

    seed: int = 0
