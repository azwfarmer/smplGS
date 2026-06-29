"""Training loop: init Gaussians from SMPL, then per-iter articulate + rasterize
+ loss + Adam, with 3DGS adaptive density control (WRITEUP B.4, B.7).

Run:  python train.py   (from the human/ directory; edit config.py for paths/subject)
"""
from __future__ import annotations
import os
import numpy as np
import torch
import imageio.v2 as imageio

from config import Config
from smpl_model import SMPLBody
from sampling import sample_gaussians
from gaussian_model import GaussianModel
from deform import Deformer
from rasterizer import render
from losses import total_loss
from dataset_zju import ZJUDataset


def build(cfg: Config):
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    smpl = SMPLBody(cfg.smpl_model_path).to(dev)
    data = ZJUDataset(cfg, split="train")
    betas = data.rest_betas().to(dev)
    v_rest, _ = smpl.rest_state(betas)

    init = sample_gaussians(v_rest, smpl.faces, smpl.lbs_weights,
                            cfg.n_gaussians, cfg.sh_degree,
                            cfg.scale_normal_ratio, cfg.init_opacity, cfg.seed)
    gaussians = GaussianModel(init, cfg.sh_degree, cfg.pose_feat_dim).to(dev)
    gaussians.setup_optimizer(cfg)
    deformer = Deformer(cfg).to(dev)
    mlp_opt = torch.optim.Adam(deformer.parameters(), lr=cfg.lr_mlp) \
        if list(deformer.parameters()) else None
    extent = (init["xyz"] - init["xyz"].mean(0)).norm(dim=1).max().item()
    return smpl, data, betas, gaussians, deformer, mlp_opt, extent


def make_frame(smpl, betas, sample, device):
    """Assemble the per-frame inputs for the deformer."""
    s = sample["smpl"]
    theta = s["poses"].to(device)                         # (72,)
    A, _ = smpl.bone_transforms(theta, betas)             # (J,4,4) with rest betas
    return {
        "A": A,
        "Rh": s["Rh"].to(device),
        "Th": s["Th"].to(device),
        "pose": theta[3:72].reshape(1, 69),               # body pose (no root)
        "campos": sample["camera"].campos,
    }


def save_preview(path, pred, gt):
    img = torch.cat([pred.clamp(0, 1), gt.clamp(0, 1)], dim=2)   # side by side
    img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, img)


def save_checkpoint(path, gaussians, deformer, betas, cfg, it):
    state = {n: getattr(gaussians, n).detach().cpu() for n in gaussians._names}
    state["skin"] = gaussians.skin.detach().cpu()
    torch.save({"gaussians": state, "deformer": deformer.state_dict(),
                "betas": betas.detach().cpu(), "iteration": it,
                "sh_degree": cfg.sh_degree}, path)


def train(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    dev = cfg.device
    smpl, data, betas, gaussians, deformer, mlp_opt, extent = build(cfg)
    bg = torch.tensor(cfg.bg_color, device=dev)
    print(f"[train] {len(data)} samples, {gaussians.num_points} Gaussians, "
          f"extent={extent:.3f} m")

    for it in range(1, cfg.iterations + 1):
        gaussians.update_xyz_lr(it)
        sample = data[np.random.randint(len(data))]
        cam = sample["camera"]
        frame = make_frame(smpl, betas, sample, dev)

        means3D, cov3D, colors = deformer(gaussians, frame)
        out = render(means3D, cov3D, colors, gaussians.get_opacity, cam, bg, cfg)
        loss, logs = total_loss(out["image"], sample["image"],
                                out["alpha"], sample["mask"], cfg)

        gaussians.optimizer.zero_grad(set_to_none=True)
        if mlp_opt:
            mlp_opt.zero_grad(set_to_none=True)
        loss.backward()

        # ---- densification stats (NDC view-space mean gradient) ----------
        # The CUDA rasterizer fills means2D.grad in NDC units already, so we use
        # it directly against the (NDC-scale) densify_grad_threshold.
        if cfg.densify and it < cfg.densify_until_iter:
            with torch.no_grad():
                g = out["means2D"].grad
                if g is not None:
                    gaussians.add_densification_stats(g[:, :2], out["visible"])
                    vis = out["visible"]
                    gaussians.max_radii2D[vis] = torch.maximum(
                        gaussians.max_radii2D[vis], out["max_radii2D"][vis])

        gaussians.optimizer.step()
        if mlp_opt:
            mlp_opt.step()

        # ---- adaptive density control ------------------------------------
        if cfg.densify and cfg.densify_from_iter <= it < cfg.densify_until_iter:
            if it % cfg.densify_interval == 0:
                max_screen = cfg.max_screen_size if it > cfg.opacity_reset_interval else 0
                gaussians.densify_and_prune(cfg.densify_grad_threshold,
                                            cfg.min_opacity, extent, max_screen,
                                            cfg.percent_dense)
            if it % cfg.opacity_reset_interval == 0:
                gaussians.reset_opacity(0.01)

        # ---- logging -----------------------------------------------------
        if it % cfg.log_interval == 0:
            print(f"it {it:6d} | loss {loss.item():.4f} | L1 {logs['l1']:.4f} "
                  f"D-SSIM {logs['dssim']:.4f} mask {logs['mask']:.4f} | "
                  f"N {gaussians.num_points}")
        if it % cfg.preview_interval == 0:
            save_preview(os.path.join(cfg.out_dir, f"preview_{it:06d}.png"),
                         out["image"], sample["image"])
        if it % cfg.ckpt_interval == 0 or it == cfg.iterations:
            save_checkpoint(os.path.join(cfg.out_dir, f"ckpt_{it:06d}.pt"),
                            gaussians, deformer, betas, cfg, it)
    print("[train] done.")


if __name__ == "__main__":
    train(Config())
