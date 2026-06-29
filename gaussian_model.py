"""The canonical Gaussian set + 3DGS adaptive density control.

Holds the optimisable per-Gaussian parameters (position, SH colour, scale,
rotation, opacity, and a pose-colour latent) plus a fixed skinning-weight
*buffer*. It owns its own Adam optimiser over those parameters so that
densification can perform the in-place optimiser-state surgery 3DGS needs
(extend/prune the Adam moment buffers when the Gaussian count changes).

Activations (WRITEUP A.1): scale=exp(log_scale), rotation=normalize(quat),
opacity=sigmoid(opacity_raw). Colour is SH, evaluated elsewhere with the view
direction. Densification (WRITEUP B.4): clone/split by view-space gradient and
scale, prune by opacity/size, periodic opacity reset.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from geometry import quaternion_to_matrix


class GaussianModel(nn.Module):
    def __init__(self, init: dict, sh_degree: int, pose_feat_dim: int = 8):
        super().__init__()
        self.sh_degree = sh_degree
        N = init["xyz"].shape[0]
        # optimisable parameters
        self.xyz = nn.Parameter(init["xyz"])
        self.features_dc = nn.Parameter(init["features_dc"])      # (N,1,3)
        self.features_rest = nn.Parameter(init["features_rest"])  # (N,K-1,3)
        self.log_scale = nn.Parameter(init["log_scale"])
        self.quat = nn.Parameter(init["quat"])
        self.opacity_raw = nn.Parameter(init["opacity_raw"])
        pf = init.get("pose_feat", torch.zeros(N, pose_feat_dim))
        self.pose_feat = nn.Parameter(pf)
        # fixed skinning weights (not trained)
        self.register_buffer("skin", init["skin"])               # (N,J)
        # densification bookkeeping
        self.register_buffer("xyz_grad_accum", torch.zeros(N, 1))
        self.register_buffer("denom", torch.zeros(N, 1))
        self.register_buffer("max_radii2D", torch.zeros(N))
        self.optimizer = None
        # names <-> attributes for optimiser-state surgery (densified tensors only)
        self._names = ["xyz", "features_dc", "features_rest",
                       "log_scale", "quat", "opacity_raw", "pose_feat"]

    # ---- activated getters ----------------------------------------------
    @property
    def get_xyz(self):       return self.xyz
    @property
    def get_scaling(self):   return torch.exp(self.log_scale)
    @property
    def get_rotation(self):  return torch.nn.functional.normalize(self.quat, dim=-1)
    @property
    def get_opacity(self):   return torch.sigmoid(self.opacity_raw)
    @property
    def get_features(self):  return torch.cat([self.features_dc, self.features_rest], dim=1)
    @property
    def num_points(self):    return self.xyz.shape[0]

    def covariance(self):
        """Canonical 3D covariance Sigma = R S^2 R^T, (N,3,3)."""
        R = quaternion_to_matrix(self.get_rotation)
        S = self.get_scaling
        return R @ torch.diag_embed(S * S) @ R.transpose(-1, -2)

    # ---- optimiser ------------------------------------------------------
    def setup_optimizer(self, cfg):
        groups = [
            {"params": [self.xyz],           "lr": cfg.lr_xyz_init,  "name": "xyz"},
            {"params": [self.features_dc],   "lr": cfg.lr_sh_dc,     "name": "features_dc"},
            {"params": [self.features_rest], "lr": cfg.lr_sh_rest,   "name": "features_rest"},
            {"params": [self.log_scale],     "lr": cfg.lr_scale,     "name": "log_scale"},
            {"params": [self.quat],          "lr": cfg.lr_quat,      "name": "quat"},
            {"params": [self.opacity_raw],   "lr": cfg.lr_opacity,   "name": "opacity_raw"},
            {"params": [self.pose_feat],     "lr": cfg.lr_mlp,       "name": "pose_feat"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self._xyz_lr = (cfg.lr_xyz_init, cfg.lr_xyz_final, cfg.iterations)

    def update_xyz_lr(self, it: int):
        """Log-linear exponential decay of the position LR (3DGS schedule)."""
        lr0, lr1, T = self._xyz_lr
        t = min(it / max(T, 1), 1.0)
        lr = lr0 * (lr1 / lr0) ** t
        for grp in self.optimizer.param_groups:
            if grp["name"] == "xyz":
                grp["lr"] = lr
        return lr

    # ---- densification stats (called each iter, before optimiser.step) ---
    def add_densification_stats(self, viewspace_grad_ndc: torch.Tensor,
                                visible: torch.Tensor):
        """Accumulate ||grad of NDC mean|| for visible Gaussians (3DGS)."""
        self.xyz_grad_accum[visible] += viewspace_grad_ndc[visible].norm(dim=-1, keepdim=True)
        self.denom[visible] += 1

    # ---- optimiser-state surgery ----------------------------------------
    def _cat_to_optimizer(self, new: dict):
        """Append new rows to each param + zero Adam moments; reassign params."""
        for grp in self.optimizer.param_groups:
            name = grp["name"]
            ext = new[name]
            old = grp["params"][0]
            state = self.optimizer.state.get(old, None)
            new_p = nn.Parameter(torch.cat([old.data, ext], dim=0).requires_grad_(True))
            if state is not None:
                state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros_like(ext)], 0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(ext)], 0)
                del self.optimizer.state[old]
                self.optimizer.state[new_p] = state
            grp["params"][0] = new_p
            setattr(self, name, new_p)

    def _prune_optimizer(self, keep: torch.Tensor):
        for grp in self.optimizer.param_groups:
            name = grp["name"]
            old = grp["params"][0]
            state = self.optimizer.state.get(old, None)
            new_p = nn.Parameter(old.data[keep].requires_grad_(True))
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
                del self.optimizer.state[old]
                self.optimizer.state[new_p] = state
            grp["params"][0] = new_p
            setattr(self, name, new_p)

    def _reset_stats(self, N):
        dev = self.xyz.device
        self.xyz_grad_accum = torch.zeros(N, 1, device=dev)
        self.denom = torch.zeros(N, 1, device=dev)
        self.max_radii2D = torch.zeros(N, device=dev)

    def prune_points(self, prune_mask: torch.Tensor):
        keep = ~prune_mask
        self._prune_optimizer(keep)
        self.skin = self.skin[keep]
        self.xyz_grad_accum = self.xyz_grad_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def _append(self, new: dict, new_skin: torch.Tensor):
        self._cat_to_optimizer(new)
        self.skin = torch.cat([self.skin, new_skin], dim=0)
        self._reset_stats(self.xyz.shape[0])

    # ---- the three density operations -----------------------------------
    @torch.no_grad()
    def densify_and_prune(self, grad_threshold, min_opacity, extent,
                          max_screen_size, percent_dense):
        grads = (self.xyz_grad_accum / self.denom.clamp_min(1)).squeeze(-1)
        grads[self.denom.squeeze(-1) == 0] = 0.0

        self._densify_and_clone(grads, grad_threshold, extent, percent_dense)
        self._densify_and_split(grads, grad_threshold, extent, percent_dense)

        # prune: transparent, or (after warmup) too big on screen / in world
        prune = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            prune |= self.max_radii2D > max_screen_size
            prune |= self.get_scaling.max(dim=1).values > 0.1 * extent
        self.prune_points(prune)
        torch.cuda.empty_cache() if self.xyz.is_cuda else None

    def _select_new(self, idx, fields):
        """Gather rows `idx` from every densified tensor + skin."""
        new = {n: getattr(self, n).data[idx] for n in self._names}
        new.update(fields)                          # allow overrides (e.g. xyz, scale)
        return new, self.skin[idx]

    def _densify_and_clone(self, grads, thr, extent, percent_dense):
        sel = (grads >= thr) & (self.get_scaling.max(dim=1).values <= percent_dense * extent)
        if sel.sum() == 0:
            return
        idx = sel.nonzero(as_tuple=True)[0]
        new, new_skin = self._select_new(idx, {})   # exact duplicates
        self._append(new, new_skin)

    def _densify_and_split(self, grads, thr, extent, percent_dense, n_split=2):
        # grads/scaling may have grown if clone ran first; re-pad to current N
        if grads.shape[0] < self.num_points:
            grads = torch.cat([grads, torch.zeros(self.num_points - grads.shape[0],
                                                  device=grads.device)])
        sel = (grads >= thr) & (self.get_scaling.max(dim=1).values > percent_dense * extent)
        if sel.sum() == 0:
            return
        idx = sel.nonzero(as_tuple=True)[0]
        idx_rep = idx.repeat(n_split)
        # sample child means inside the parent ellipsoid
        stds = self.get_scaling[idx_rep]
        samples = torch.randn_like(stds) * stds
        R = quaternion_to_matrix(self.get_rotation[idx_rep])
        offset = (R @ samples.unsqueeze(-1)).squeeze(-1)
        new_xyz = self.xyz.data[idx_rep] + offset
        new_log_scale = torch.log(self.get_scaling[idx_rep] / (0.8 * n_split))
        new, new_skin = self._select_new(idx_rep,
                                         {"xyz": new_xyz, "log_scale": new_log_scale})
        self._append(new, new_skin)
        # remove the parents that were split
        prune = torch.zeros(self.num_points, dtype=torch.bool, device=self.xyz.device)
        prune[idx] = True
        self.prune_points(prune)

    @torch.no_grad()
    def reset_opacity(self, value=0.01):
        """Clamp opacity down to `value` (3DGS opacity reset)."""
        from sampling import inverse_sigmoid
        new = torch.full_like(self.opacity_raw, inverse_sigmoid(value))
        new = torch.minimum(self.opacity_raw.data, new)
        self._replace_param("opacity_raw", new)

    def _replace_param(self, name, new_data):
        """Replace a single param's data + zero its Adam moments."""
        for grp in self.optimizer.param_groups:
            if grp["name"] != name:
                continue
            old = grp["params"][0]
            state = self.optimizer.state.get(old, None)
            new_p = nn.Parameter(new_data.requires_grad_(True))
            if state is not None:
                state["exp_avg"] = torch.zeros_like(new_p)
                state["exp_avg_sq"] = torch.zeros_like(new_p)
                del self.optimizer.state[old]
                self.optimizer.state[new_p] = state
            grp["params"][0] = new_p
            setattr(self, name, new_p)
