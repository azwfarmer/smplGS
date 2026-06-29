"""Sample Gaussians on the SMPL rest surface and initialise their parameters.

Given the rest mesh we place one Gaussian per vertex plus area-weighted
barycentric face samples to reach the target count N (WRITEUP A.2). Each gets:
  * a tangent-aligned rotation so it starts as a flat disk on the surface (A.3),
  * an anisotropic scale from the local sample spacing (A.3),
  * SMPL skinning weights, interpolated and renormalised (A.4),
  * an initial grey colour and low opacity.
Everything here is plain geometry on the *rest* mesh; no autograd needed.
"""
from __future__ import annotations
import torch
from sh import RGB2SH


def inverse_sigmoid(x: float) -> float:
    import math
    return math.log(x / (1.0 - x))


def vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Area-weighted per-vertex normals."""
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)  # (F,3)
    n = torch.zeros_like(v)
    n.index_add_(0, f[:, 0], fn)
    n.index_add_(0, f[:, 1], fn)
    n.index_add_(0, f[:, 2], fn)
    return torch.nn.functional.normalize(n, dim=-1)


def _knn_mean_dist(pts: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance to the k nearest *other* points (chunked, memory-safe)."""
    M = pts.shape[0]
    out = torch.empty(M, device=pts.device, dtype=pts.dtype)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        d = torch.cdist(pts[s:e], pts)               # (chunk, M)
        knn = d.topk(k + 1, largest=False).values    # includes self (dist 0)
        out[s:e] = knn[:, 1:].mean(dim=-1)           # drop self
    return out


def rotation_from_normal(n: torch.Tensor) -> torch.Tensor:
    """Build R0=[t,b,n] with an arbitrary in-plane tangent -> quaternion (w,x,y,z).

    In-plane orientation is irrelevant at init (s_t=s_b, so the disk is
    rotationally symmetric in plane); we only need t,b spanning the tangent
    plane. Pick a reference axis least aligned with n, Gram-Schmidt for t.
    """
    ref = torch.zeros_like(n)
    ref[:, 0] = 1.0
    flip = n[:, 0].abs() > 0.9                       # n ~ x-axis -> use y instead
    ref[flip] = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    t = torch.nn.functional.normalize(ref - (ref * n).sum(-1, keepdim=True) * n, dim=-1)
    b = torch.cross(n, t, dim=-1)
    R = torch.stack([t, b, n], dim=-1)               # columns = axes -> (M,3,3)
    return matrix_to_quaternion(R)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Rotation (...,3,3) -> unit quaternion (...,4) in (w,x,y,z) order."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    w = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    x = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0))
    y = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0))
    z = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0))
    x = torch.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = torch.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = torch.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    q = torch.stack([w, x, y, z], dim=-1)
    return torch.nn.functional.normalize(q, dim=-1)


@torch.no_grad()
def sample_gaussians(v_rest, faces, lbs_weights, n_target, sh_degree,
                     scale_normal_ratio=0.35, init_opacity=0.1, seed=0):
    """Return a dict of init tensors for the canonical Gaussians.

    Keys: xyz (N,3), quat (N,4), log_scale (N,3), opacity_raw (N,1),
          features_dc (N,1,3), features_rest (N,K-1,3), skin (N,J).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = v_rest.device
    V = v_rest.shape[0]
    vnorm = vertex_normals(v_rest, faces)

    # --- vertex samples (up to N) ---
    if n_target <= V:
        idx = torch.randperm(V, generator=g)[:n_target]
        pos = v_rest[idx]; nrm = vnorm[idx]; skin = lbs_weights[idx]
    else:
        pos, nrm, skin = v_rest, vnorm, lbs_weights
        # --- area-weighted face samples for the remainder ---
        n_face = n_target - V
        v0, v1, v2 = v_rest[faces[:, 0]], v_rest[faces[:, 1]], v_rest[faces[:, 2]]
        area = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
        prob = (area / area.sum()).cpu()                       # multinomial on CPU
        fsel = torch.multinomial(prob, n_face, replacement=True, generator=g).to(device)
        u1 = torch.rand(n_face, 1, generator=g).to(device)
        u2 = torch.rand(n_face, 1, generator=g).to(device)
        b0 = 1 - torch.sqrt(u1); b1 = torch.sqrt(u1) * (1 - u2); b2 = torch.sqrt(u1) * u2
        fv = faces[fsel]
        fp = b0 * v_rest[fv[:, 0]] + b1 * v_rest[fv[:, 1]] + b2 * v_rest[fv[:, 2]]
        fn = b0 * vnorm[fv[:, 0]] + b1 * vnorm[fv[:, 1]] + b2 * vnorm[fv[:, 2]]
        fn = torch.nn.functional.normalize(fn, dim=-1)
        fw = b0 * lbs_weights[fv[:, 0]] + b1 * lbs_weights[fv[:, 1]] + b2 * lbs_weights[fv[:, 2]]
        fw = fw / fw.sum(-1, keepdim=True).clamp_min(1e-8)
        pos = torch.cat([pos, fp], 0)
        nrm = torch.nn.functional.normalize(torch.cat([nrm, fn], 0), dim=-1)
        skin = torch.cat([skin, fw], 0)

    N = pos.shape[0]
    quat = rotation_from_normal(nrm)

    # anisotropic surface-hugging scale from neighbour spacing (A.3)
    d = _knn_mean_dist(pos, k=3).clamp_min(1e-4)                # (N,)
    s_t = d
    s_n = scale_normal_ratio * d
    scale = torch.stack([s_t, s_t, s_n], dim=-1)               # tangent,tangent,normal
    log_scale = torch.log(scale.clamp_min(1e-6))

    opacity_raw = torch.full((N, 1), inverse_sigmoid(init_opacity), device=device)

    # SH colour: grey DC, zero higher orders
    K = (sh_degree + 1) ** 2
    features = torch.zeros(N, K, 3, device=device)
    features[:, 0, :] = RGB2SH(torch.full((N, 3), 0.5, device=device))
    features_dc = features[:, :1, :].contiguous()
    features_rest = features[:, 1:, :].contiguous()

    return {
        "xyz": pos.contiguous(),
        "quat": quat.contiguous(),
        "log_scale": log_scale.contiguous(),
        "opacity_raw": opacity_raw,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "skin": skin.contiguous(),
    }
