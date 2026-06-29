"""Motion + appearance: non-rigid deformation, skeletal articulation (LBS),
and view/pose-dependent colour (WRITEUP A.5, B colour).

`Deformer` owns two small pose-conditioned MLPs (both zero-initialised at the
head so training starts as pure-LBS, pure-SH):
  * non-rigid:  F(gamma(mu), pose) -> (d_mu, d_logscale, d_quat)
  * pose-colour: g(feat, pose)     -> RGB offset added to the SH colour.
Its `forward` maps the canonical Gaussians to posed-world means and covariances
and evaluates their colour for a given camera.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from geometry import quaternion_to_matrix
from smpl_model import rodrigues
from sh import eval_sh


class PositionalEncoding:
    """gamma(x) = [x, sin(2^k pi x), cos(2^k pi x)]_{k<freqs}."""
    def __init__(self, freqs: int):
        self.freqs = freqs
        self.bands = (2.0 ** torch.arange(freqs)) * math.pi

    @property
    def out_dim_per_in(self):
        return 1 + 2 * self.freqs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.bands.to(x)
        xb = x[..., None] * bands                       # (...,D,F)
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # (...,D,2F)
        return torch.cat([x, enc.flatten(-2)], dim=-1)  # (...,D*(1+2F))


def _zero_init(layer: nn.Linear):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class NonRigidMLP(nn.Module):
    def __init__(self, xyz_dim, pose_dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xyz_dim + pose_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.head = nn.Linear(width, 3 + 3 + 4)         # d_mu, d_logscale, d_quat
        _zero_init(self.head)

    def forward(self, enc_xyz, pose):
        pose = pose.expand(enc_xyz.shape[0], -1)
        h = self.net(torch.cat([enc_xyz, pose], dim=-1))
        out = self.head(h)
        return out[:, :3], out[:, 3:6], out[:, 6:10]    # d_mu, d_logscale, d_quat


class PoseColorMLP(nn.Module):
    def __init__(self, feat_dim, pose_dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + pose_dim, width), nn.ReLU(),
            nn.Linear(width, width // 2), nn.ReLU(),
        )
        self.head = nn.Linear(width // 2, 3)            # RGB offset (SH/linear space)
        _zero_init(self.head)

    def forward(self, feat, pose):
        pose = pose.expand(feat.shape[0], -1)
        return self.head(self.net(torch.cat([feat, pose], dim=-1)))


class Deformer(nn.Module):
    def __init__(self, cfg, pose_dim=69):
        super().__init__()
        self.cfg = cfg
        self.sh_degree = cfg.sh_degree
        self.posenc = PositionalEncoding(cfg.posenc_freqs)
        xyz_dim = 3 * self.posenc.out_dim_per_in
        self.nonrigid = NonRigidMLP(xyz_dim, pose_dim, cfg.mlp_width) \
            if cfg.use_nonrigid else None
        self.pose_color = PoseColorMLP(cfg.pose_feat_dim, pose_dim, cfg.mlp_width) \
            if cfg.use_pose_color else None
        self._ident = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # ---- non-rigid corrections in canonical space -----------------------
    def _nonrigid(self, xyz, pose):
        N = xyz.shape[0]
        if self.nonrigid is None:
            z = xyz.new_zeros(N, 3)
            return z, z, self._ident.to(xyz).expand(N, 4)
        d_mu, d_logs, d_quat = self.nonrigid(self.posenc(xyz), pose)
        q = self._ident.to(xyz) + d_quat                 # residual to identity
        return d_mu, d_logs, torch.nn.functional.normalize(q, dim=-1)

    def forward(self, gaussians, frame):
        """Map canonical Gaussians to posed world. `frame` provides A (J,4,4),
        Rh (3,), Th (3,), pose (69,), campos (3,)."""
        mu, log_s, quat = gaussians.xyz, gaussians.log_scale, gaussians.get_rotation
        A, Rh, Th, pose = frame["A"], frame["Rh"], frame["Th"], frame["pose"]

        # 1) non-rigid deformation (canonical)
        d_mu, d_logs, dq = self._nonrigid(mu, pose)
        mu_def = mu + d_mu
        R_eff = quaternion_to_matrix(quat) @ quaternion_to_matrix(dq)
        S_eff = torch.exp(log_s + d_logs)
        Sigma_def = R_eff @ torch.diag_embed(S_eff * S_eff) @ R_eff.transpose(1, 2)

        # 2) LBS blend T = sum_b w_b A_b
        T = torch.einsum("nj,jik->nik", gaussians.skin, A)   # (N,4,4)
        T_lin, T_t = T[:, :3, :3], T[:, :3, 3]
        x = (T_lin @ mu_def.unsqueeze(-1)).squeeze(-1) + T_t

        # 3) ZJU global placement; covariance via linear part M = R(Rh) T_lin
        R_Rh = rodrigues(Rh)
        x_world = x @ R_Rh.transpose(0, 1) + Th
        M = R_Rh.unsqueeze(0) @ T_lin
        Sigma_world = M @ Sigma_def @ M.transpose(1, 2)

        colors = self._colors(gaussians, x_world, frame["campos"], pose)
        return x_world, Sigma_world, colors

    # ---- view/pose-dependent colour -------------------------------------
    def _colors(self, gaussians, x_world, campos, pose):
        view_dir = torch.nn.functional.normalize(x_world - campos, dim=-1)
        col = eval_sh(self.sh_degree, gaussians.get_features, view_dir)  # (N,3) linear
        if self.pose_color is not None:
            col = col + self.pose_color(gaussians.pose_feat, pose)
        return (col + 0.5).clamp(0.0, 1.0)
