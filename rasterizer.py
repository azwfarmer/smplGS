"""CUDA-backed differentiable rasterizer (the standard 3DGS splatter).

This is a thin adapter over `diff_gaussian_rasterization` (graphdeco-inria), the
canonical EWA-splatting CUDA kernel, replacing the earlier pure-PyTorch tiled
splatter. It keeps the *exact* `render(...)` signature the rest of the project
expects (`train.py`, `demo.py`) so nothing downstream changes:

  render(means3D, cov3D, colors, opacities, cam, bg, cfg)
      -> dict(image (3,H,W), alpha (H,W), means2D, radii, max_radii2D, visible)

We feed the kernel the *precomputed* world covariance (from `deform.py`) and the
*precomputed* RGB (already SH-evaluated + pose-offset there), so the CUDA side
does no SH/covariance work — it just projects, tiles, sorts and alpha-composites.

Two details worth knowing:
  * Off-centre principal point. ZJU-MoCap cameras have (cx,cy) != image centre.
    The stock kernel assumes a centred frustum, so we bake (fx,fy,cx,cy) into an
    asymmetric OpenGL projection matrix (`_gs_camera`) — the projected screen
    means then land at the true pixels. (The EWA covariance Jacobian still uses a
    symmetric tan-fov, a negligible approximation away from the frustum edge.)
  * Alpha (silhouette) for the mask loss. The stock kernel returns only RGB +
    radii, so we run a second pass with unit colours on a black background; the
    rendered value is then exactly sum_i alpha_i T_i — the accumulated alpha.
    Both passes share one screen-space tensor, so its `.grad` accumulates the
    photometric *and* mask gradients for densification (as the old single-pass
    splatter did).
"""
from __future__ import annotations
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

_ZFAR = 100.0   # far plane (only affects depth NDC, not x/y screen position)


def _gs_camera(cam, near: float, device, dtype):
    """Build the kernel's (viewmatrix, full_projmatrix, tanfovx, tanfovy, campos).

    Follows the 3DGS row-vector / glm convention: matrices are transposed so the
    CUDA `transformPoint4x4` (which treats them as column-major) yields
    x_cam = R x_world + t and the off-centre clip transform below.
    """
    R, t = cam.R.to(device=device, dtype=dtype), cam.t.to(device=device, dtype=dtype)
    # world->cam as a 4x4, then transpose for the kernel's convention
    Rt = torch.zeros(4, 4, device=device, dtype=dtype)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    viewmatrix = Rt.transpose(0, 1)

    # Asymmetric perspective from (fx,fy,cx,cy): x_ndc = (2fx/W)(X/Z) + (2cx/W-1)
    W, H = cam.W, cam.H
    P = torch.zeros(4, 4, device=device, dtype=dtype)
    P[0, 0] = 2.0 * cam.fx / W
    P[1, 1] = 2.0 * cam.fy / H
    P[0, 2] = 2.0 * cam.cx / W - 1.0
    P[1, 2] = 2.0 * cam.cy / H - 1.0
    P[2, 2] = _ZFAR / (_ZFAR - near)
    P[2, 3] = -(_ZFAR * near) / (_ZFAR - near)
    P[3, 2] = 1.0
    full_proj = viewmatrix.unsqueeze(0).bmm(P.transpose(0, 1).unsqueeze(0)).squeeze(0)

    # tan(fov/2) chosen so the kernel recovers focal_x = W/(2 tanfovx) = fx
    tanfovx = W / (2.0 * cam.fx)
    tanfovy = H / (2.0 * cam.fy)
    campos = cam.campos.to(device=device, dtype=dtype)
    return viewmatrix, full_proj, tanfovx, tanfovy, campos


def render(means3D, cov3D, colors, opacities, cam, bg, cfg):
    """Rasterize world Gaussians with the CUDA kernel.

    Args mirror the old pure-torch splatter:
        means3D  (N,3)   posed world means
        cov3D    (N,3,3) posed world covariance
        colors   (N,3)   precomputed RGB in [0,1]
        opacities(N,1)   activated alpha
        cam      Camera, bg (3,) background, cfg Config
    Returns dict(image (3,H,W), alpha (H,W), means2D, radii, max_radii2D, visible).
    """
    device, dtype = means3D.device, means3D.dtype
    means3D = means3D.contiguous()
    colors = colors.contiguous()
    opacities = opacities.contiguous()
    bg = bg.to(device=device, dtype=dtype)

    # symmetric 3x3 covariance -> upper-triangular 6-vector [00,01,02,11,12,22]
    cov6 = torch.stack([cov3D[:, 0, 0], cov3D[:, 0, 1], cov3D[:, 0, 2],
                        cov3D[:, 1, 1], cov3D[:, 1, 2], cov3D[:, 2, 2]],
                       dim=-1).contiguous()

    viewmatrix, full_proj, tanfovx, tanfovy, campos = _gs_camera(cam, cfg.near, device, dtype)

    # screen-space means: zeros tensor the kernel fills with the NDC mean gradient
    # (used by densification). Shared across both passes so mask-loss grads count.
    screen = torch.zeros_like(means3D, requires_grad=True)
    try:
        screen.retain_grad()
    except Exception:
        pass

    def _settings(background):
        return GaussianRasterizationSettings(
            image_height=int(cam.H), image_width=int(cam.W),
            tanfovx=float(tanfovx), tanfovy=float(tanfovy),
            bg=background, scale_modifier=1.0,
            viewmatrix=viewmatrix, projmatrix=full_proj,
            sh_degree=0, campos=campos, prefiltered=False, debug=False)

    # --- pass 1: RGB, real background ---
    rasterizer = GaussianRasterizer(_settings(bg))
    image, radii = rasterizer(
        means3D=means3D, means2D=screen, opacities=opacities,
        shs=None, colors_precomp=colors, scales=None, rotations=None,
        cov3D_precomp=cov6)

    # --- pass 2: silhouette alpha = render unit colour on black -> sum of weights ---
    ones = torch.ones_like(colors)
    black = torch.zeros(3, device=device, dtype=dtype)
    alpha_rasterizer = GaussianRasterizer(_settings(black))
    alpha_img, _ = alpha_rasterizer(
        means3D=means3D, means2D=screen, opacities=opacities,
        shs=None, colors_precomp=ones, scales=None, rotations=None,
        cov3D_precomp=cov6)
    alpha = alpha_img[0]                              # all 3 channels identical

    radii = radii.to(dtype)
    visible = radii > 0
    return {"image": image, "alpha": alpha, "means2D": screen,
            "radii": radii, "max_radii2D": radii, "visible": visible}
