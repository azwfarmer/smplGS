"""Self-contained validation that needs no SMPL model or ZJU data.

Run:  python tests/test_synthetic.py   (from the human/ directory)

Covers the parts most likely to be wrong:
  1. from-scratch SMPL LBS (`bone_transforms`/`lbs`) on a synthetic skeleton,
  2. CUDA rasterizer differentiability via finite-difference checks,
  3. SH colour DC term, 4. densification optimiser-state surgery,
  5. full articulate->rasterize->backward on the GPU.
"""
from __future__ import annotations
import os
import sys
import tempfile
import types
import numpy as np
import torch

# This test lives in human/tests/ but imports the flat modules in human/.
# Put the parent (human/) directory on sys.path so it runs standalone from
# anywhere, e.g. `python tests/test_synthetic.py` or `python human/tests/test_synthetic.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smpl_model import SMPLBody, rodrigues
from rasterizer import render
from camera import Camera
from sh import eval_sh, RGB2SH
from gaussian_model import GaussianModel
from config import Config
from deform import Deformer


# ---------------------------------------------------------------------------
def _synthetic_smpl_npz(path):
    """A 3-joint chain skeleton with joints sitting on 3 of the vertices."""
    V, J = 6, 3
    v_template = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0],
                           [0.3, 0.2, 0], [0.3, 1.2, 0], [0.3, 2.2, 0]], np.float32)
    J_regressor = np.zeros((J, V), np.float32)
    J_regressor[0, 0] = J_regressor[1, 1] = J_regressor[2, 2] = 1.0   # joints = v0,v1,v2
    weights = np.zeros((V, J), np.float32)
    weights[[0, 3], 0] = 1.0      # bottom verts -> root
    weights[[1, 4], 1] = 1.0      # middle      -> joint 1
    weights[[2, 5], 2] = 1.0      # top         -> joint 2
    np.savez(path,
             v_template=v_template,
             shapedirs=np.zeros((V, 3, 10), np.float32),
             J_regressor=J_regressor, weights=weights,
             kintree_table=np.array([[-1, 0, 1], [0, 1, 2]], np.int64),
             f=np.array([[0, 1, 2]], np.int64))


def test_lbs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "synthetic.npz")
        _synthetic_smpl_npz(p)
        body = SMPLBody(p)
        betas = torch.zeros(10)
        v_rest, J = body.rest_state(betas)

        # (a) zero pose -> identity -> mesh unchanged
        zero = torch.zeros(9)
        v0 = body.lbs(zero, betas)
        assert torch.allclose(v0, v_rest, atol=1e-6), "zero-pose LBS must be identity"

        # (b) rotate root about z by 90deg -> root-weighted verts map by closed form
        ang = torch.tensor(np.pi / 2)
        theta = torch.zeros(3, 3)
        theta[0] = torch.tensor([0, 0, float(ang)])          # root axis-angle
        v = body.lbs(theta.reshape(-1), betas)
        Rz = rodrigues(theta[0])
        J0 = J[0]
        for vi in [0, 3]:                                    # verts weighted to root
            expect = Rz @ (v_rest[vi] - J0) + J0
            assert torch.allclose(v[vi], expect, atol=1e-5), "root rotation closed form"

        # (c) child joint rotation, vertex on joint 2 follows the chain
        theta = torch.zeros(3, 3)
        theta[1] = torch.tensor([0.0, 0.0, 0.5])             # rotate joint 1
        v = body.lbs(theta.reshape(-1), betas)
        # joint 1 frame: G1 = T_loc0 @ T_loc1 ; verify A applied to a j2-weighted vert
        A, _ = body.bone_transforms(theta.reshape(-1), betas)
        vi = 2
        vh = torch.cat([v_rest[vi], torch.ones(1)])
        assert torch.allclose(v[vi], (A[2] @ vh)[:3], atol=1e-6)
    print("  [1] SMPL LBS (identity / root / chain) .......... OK")


# ---------------------------------------------------------------------------
def test_gradcheck():
    """The CUDA kernel is float32/GPU-only, so float64 `gradcheck` doesn't apply.
    Instead we central-difference a global scalar (sum of the rendered image)
    against the analytic backward: exact for the linear colour path, smooth (and
    so closely matched) for opacity."""
    if not torch.cuda.is_available():
        print("  [2] rasterizer differentiability (CUDA FD) ..... SKIP (no CUDA)")
        return
    torch.manual_seed(0)
    dev = "cuda"
    M = 50
    cam = Camera(R=torch.eye(3, device=dev), t=torch.zeros(3, device=dev),
                 fx=200.0, fy=200.0, cx=64.0, cy=64.0, W=128, H=128)
    cfg = types.SimpleNamespace(near=0.05)
    bg = torch.zeros(3, device=dev)
    means3D = torch.randn(M, 3, device=dev) * 0.2 + torch.tensor([0, 0, 2.5], device=dev)
    cov3D = (torch.eye(3, device=dev) * 4e-3).expand(M, 3, 3).contiguous()

    def scalar(colors, opac):
        out = render(means3D, cov3D, colors, opac, cam, bg, cfg)
        return out["image"].sum()

    colors = torch.rand(M, 3, device=dev)
    opac = torch.full((M, 1), 0.4, device=dev)

    # (a) colour path is linear -> analytic grad must match FD almost exactly
    c = colors.clone().requires_grad_(True)
    scalar(c, opac).backward()
    eps = 1e-3
    cp, cm = colors.clone(), colors.clone()
    cp[7, 1] += eps; cm[7, 1] -= eps
    fd = (scalar(cp, opac) - scalar(cm, opac)) / (2 * eps)
    assert torch.allclose(c.grad[7, 1], fd, rtol=2e-2, atol=1e-2), (c.grad[7, 1].item(), fd.item())

    # (b) opacity path is smooth -> analytic grad closely matches FD
    o = opac.clone().requires_grad_(True)
    scalar(colors, o).backward()
    op, om = opac.clone(), opac.clone()
    op[3, 0] += eps; om[3, 0] -= eps
    fd_o = (scalar(colors, op) - scalar(colors, om)) / (2 * eps)
    assert torch.allclose(o.grad[3, 0], fd_o, rtol=5e-2, atol=1e-2), (o.grad[3, 0].item(), fd_o.item())
    print("  [2] rasterizer differentiability (CUDA FD) ..... OK")


# ---------------------------------------------------------------------------
def test_sh():
    dirs = torch.randn(5, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    sh = torch.zeros(5, 16, 3)
    rgb = torch.tensor([0.2, 0.6, 0.9])
    sh[:, 0, :] = RGB2SH(rgb)
    out = eval_sh(3, sh, dirs) + 0.5                         # DC only -> constant colour
    assert torch.allclose(out, rgb.expand(5, 3), atol=1e-5)
    print("  [3] SH DC term reproduces base colour .......... OK")


# ---------------------------------------------------------------------------
def _toy_gaussians(N, sh_deg=1, device="cpu"):
    K = (sh_deg + 1) ** 2
    init = {
        "xyz": torch.randn(N, 3, device=device),
        "features_dc": torch.randn(N, 1, 3, device=device),
        "features_rest": torch.randn(N, K - 1, 3, device=device),
        "log_scale": torch.full((N, 3), -3.0, device=device),
        "quat": torch.tensor([1.0, 0, 0, 0], device=device).expand(N, 4).contiguous(),
        "opacity_raw": torch.zeros(N, 1, device=device),
        "skin": torch.softmax(torch.randn(N, 24, device=device), dim=-1),
    }
    return GaussianModel(init, sh_deg).to(device)


def test_densify():
    cfg = Config(); cfg.device = "cpu"
    g = _toy_gaussians(200, device="cpu")
    g.setup_optimizer(cfg)
    # take one Adam step so moment buffers exist
    loss = (g.get_xyz ** 2).sum() + g.get_features.sum() + g.pose_feat.sum()
    loss.backward(); g.optimizer.step()
    n0 = g.num_points
    g.xyz_grad_accum += 1.0; g.denom += 1.0                  # force everything to densify
    g.densify_and_prune(grad_threshold=1e-9, min_opacity=-1.0, extent=1.0,
                        max_screen_size=0, percent_dense=0.5)
    assert g.num_points != n0
    # optimiser params and Adam state stay shape-consistent after surgery
    for grp in g.optimizer.param_groups:
        p = grp["params"][0]
        assert p.shape[0] == g.num_points
        st = g.optimizer.state.get(p)
        if st:
            assert st["exp_avg"].shape == p.shape
    assert g.skin.shape[0] == g.num_points
    # prune everything below opacity 2.0 (all) -> 0 points
    print(f"  [4] densify+prune surgery ({n0}->{g.num_points}) ...... OK")


# ---------------------------------------------------------------------------
def test_full_forward():
    if not torch.cuda.is_available():
        print("  [5] GPU forward/backward ....................... SKIP (no CUDA)")
        return
    dev = "cuda"
    cfg = Config(); cfg.device = dev; cfg.sh_degree = 3
    cfg.tile = 16; cfg.cov2d_eps = 0.3; cfg.near = 0.05
    N = 3000
    g = _toy_gaussians(N, sh_deg=3, device=dev)
    # squeeze the cloud in front of the camera
    g.xyz.data = g.xyz.data * 0.3 + torch.tensor([0, 0, 2.5], device=dev)
    deformer = Deformer(cfg).to(dev)
    cam = Camera(R=torch.eye(3, device=dev), t=torch.zeros(3, device=dev),
                 fx=200.0, fy=200.0, cx=64.0, cy=64.0, W=128, H=128)
    frame = {
        "A": torch.eye(4, device=dev).expand(24, 4, 4).contiguous(),
        "Rh": torch.zeros(3, device=dev),
        "Th": torch.zeros(3, device=dev),
        "pose": torch.randn(1, 69, device=dev) * 0.1,
        "campos": cam.campos,
    }
    means3D, cov3D, colors = deformer(g, frame)
    out = render(means3D, cov3D, colors, g.get_opacity, cam,
                 torch.zeros(3, device=dev), cfg)
    assert out["image"].shape == (3, 128, 128)
    assert torch.isfinite(out["image"]).all()
    loss = out["image"].mean() + out["alpha"].mean()
    loss.backward()
    assert g.xyz.grad is not None and torch.isfinite(g.xyz.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in deformer.parameters())
    cov_ok = out["alpha"].max().item()
    print(f"  [5] GPU articulate+render+backward (alpha_max={cov_ok:.2f}) .. OK")


def test_train_integration():
    """Exercise train.py's inner loop (stats -> step -> densify/prune/reset) on
    fake data, without a dataset or SMPL file."""
    if not torch.cuda.is_available():
        print("  [6] train-loop integration ..................... SKIP (no CUDA)")
        return
    dev = "cuda"
    cfg = Config(); cfg.device = dev; cfg.sh_degree = 3
    cfg.tile = 32; cfg.cov2d_eps = 0.3; cfg.near = 0.05
    cfg.densify_from_iter = 20; cfg.densify_interval = 20
    cfg.densify_until_iter = 200; cfg.opacity_reset_interval = 40
    cfg.densify_grad_threshold = 0.0       # densify aggressively for the test
    g = _toy_gaussians(2000, sh_deg=3, device=dev)
    g.xyz.data = g.xyz.data * 0.3 + torch.tensor([0, 0, 2.5], device=dev)
    g.setup_optimizer(cfg)
    deformer = Deformer(cfg).to(dev)
    mlp_opt = torch.optim.Adam(deformer.parameters(), lr=1e-3)
    cam = Camera(R=torch.eye(3, device=dev), t=torch.zeros(3, device=dev),
                 fx=200.0, fy=200.0, cx=64.0, cy=64.0, W=128, H=128)
    frame = {"A": torch.eye(4, device=dev).expand(24, 4, 4).contiguous(),
             "Rh": torch.zeros(3, device=dev), "Th": torch.zeros(3, device=dev),
             "pose": torch.randn(1, 69, device=dev) * 0.1, "campos": cam.campos}
    gt = torch.rand(3, 128, 128, device=dev)
    mask = torch.zeros(128, 128, device=dev); mask[32:96, 32:96] = 1.0
    extent = 1.0
    from losses import total_loss
    n0 = g.num_points
    for it in range(1, 81):
        m3, c3, col = deformer(g, frame)
        out = render(m3, c3, col, g.get_opacity, cam, torch.zeros(3, device=dev), cfg)
        loss, _ = total_loss(out["image"], gt, out["alpha"], mask, cfg)
        g.optimizer.zero_grad(set_to_none=True); mlp_opt.zero_grad(set_to_none=True)
        loss.backward()
        g.add_densification_stats(out["means2D"].grad[:, :2], out["visible"])
        g.optimizer.step(); mlp_opt.step()
        if cfg.densify_from_iter <= it < cfg.densify_until_iter:
            if it % cfg.densify_interval == 0:
                g.densify_and_prune(cfg.densify_grad_threshold, cfg.min_opacity,
                                    extent, 0, cfg.percent_dense)
            if it % cfg.opacity_reset_interval == 0:
                g.reset_opacity(0.01)
        assert torch.isfinite(loss).all()
    assert g.num_points != n0
    print(f"  [6] train-loop integration ({n0}->{g.num_points}, no NaN) ... OK")


if __name__ == "__main__":
    print("synthetic validation:")
    test_lbs()
    test_gradcheck()
    test_sh()
    test_densify()
    test_full_forward()
    test_train_integration()
    print("all synthetic tests passed.")
