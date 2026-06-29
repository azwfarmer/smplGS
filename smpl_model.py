"""SMPL body model: rest mesh, joints, skinning weights, and *from-scratch*
per-bone skinning matrices A_b(theta, beta).

We re-implement `smplx.lbs.batch_rigid_transform` ourselves (smplx does not
expose A_b) because A_b is what articulates our Gaussians. The geometry buffers
(v_template, shapedirs, J_regressor, lbs_weights, kintree, faces) are read
directly from a SMPL `.npz` (preferred) or `.pkl`, so we do not depend on
constructing an smplx model — though `validate_against_smplx` will use smplx if
installed, as a correctness gate.

Conventions (see WRITEUP A.5/A.6):
  * pose theta: (24,3) axis-angle, joint 0 is the (usually zero) root.
  * A_b [v_canon; 1] gives the LBS-articulated point; blend by lbs_weights.
  * ZJU global placement: x_world = R(Rh) x + Th.
We deliberately skip SMPL pose blendshapes (`posedirs`); the canonical mesh is
shaped-only and the non-rigid MLP absorbs pose-dependent shape.
"""
from __future__ import annotations
import pickle
import numpy as np
import torch
import torch.nn as nn


def rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3). Differentiable."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)            # (...,1)
    k = aa / (theta + 1e-8)                                        # unit axis
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1)
    K = K.reshape(aa.shape[:-1] + (3, 3))                          # [k]_x
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta)[..., None]                                # (...,1,1)
    c = torch.cos(theta)[..., None]
    return eye + s * K + (1.0 - c) * (K @ K)


def _to_dense(x):
    """Densify a possibly-sparse (scipy) J_regressor."""
    if hasattr(x, "todense"):
        return np.asarray(x.todense())
    return np.asarray(x)


def load_smpl_arrays(path: str) -> dict:
    """Read SMPL geometry from a `.npz` or `.pkl` into numpy arrays."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        data = {k: d[k] for k in d.files}
    else:  # .pkl (original SMPL) -- arrays are chumpy.Ch, so unpickling imports
        # chumpy. chumpy 0.70 does `from numpy import bool, int, float, ...` at
        # import time; numpy>=1.24 removed those aliases, so restore them first.
        for _name, _val in {
            "bool": np.bool_, "int": np.int_, "float": np.float64,
            "complex": np.complex128, "object": np.object_,
            "str": np.str_, "unicode": np.str_,
        }.items():
            if not hasattr(np, _name):
                setattr(np, _name, _val)
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    out = {
        "v_template": np.asarray(data["v_template"], np.float32),          # (V,3)
        "shapedirs": np.asarray(data["shapedirs"], np.float32),            # (V,3,B)
        "J_regressor": _to_dense(data["J_regressor"]).astype(np.float32),  # (J,V)
        "weights": np.asarray(data["weights"], np.float32),                # (V,J)
        "kintree_table": np.asarray(data["kintree_table"]).astype(np.int64),  # (2,J)
        "faces": np.asarray(data["f"] if "f" in data else data["faces"], np.int64),
    }
    # shapedirs sometimes ships extra (e.g. SMPL+H 300) betas; keep first 10.
    if out["shapedirs"].shape[-1] > 10:
        out["shapedirs"] = out["shapedirs"][..., :10]
    return out


class SMPLBody(nn.Module):
    """Holds SMPL buffers and produces rest geometry + bone skinning matrices."""

    def __init__(self, model_path: str):
        super().__init__()
        a = load_smpl_arrays(model_path)
        self.register_buffer("v_template", torch.tensor(a["v_template"]))      # (V,3)
        self.register_buffer("shapedirs", torch.tensor(a["shapedirs"]))        # (V,3,B)
        self.register_buffer("J_regressor", torch.tensor(a["J_regressor"]))    # (J,V)
        self.register_buffer("lbs_weights", torch.tensor(a["weights"]))        # (V,J)
        self.register_buffer("faces", torch.tensor(a["faces"]))                # (F,3)
        parents = a["kintree_table"][0].copy()
        parents[0] = -1
        self.register_buffer("parents", torch.tensor(parents))                 # (J,)
        self.num_joints = int(self.lbs_weights.shape[1])
        self.num_betas = int(self.shapedirs.shape[-1])

    # ---- rest (shaped, zero-pose) geometry ------------------------------
    def rest_state(self, betas: torch.Tensor):
        """betas (B,) -> rest vertices (V,3) and rest joints J (J,3)."""
        betas = betas[: self.num_betas].to(self.v_template)
        v = self.v_template + torch.einsum("vcb,b->vc", self.shapedirs, betas)
        J = self.J_regressor @ v
        return v, J

    # ---- per-bone skinning matrices A_b (from scratch) ------------------
    def bone_transforms(self, theta: torch.Tensor, betas: torch.Tensor):
        """theta (24,3) axis-angle, betas (B,) -> A (J,4,4), rest joints (J,3).

        Replicates smplx.lbs.batch_rigid_transform for a single frame:
        compose local joint transforms down the kinematic tree, then subtract
        the rest joint so A_b acts on canonical points (WRITEUP A.5).
        """
        _, J = self.rest_state(betas)                       # (J,3) rest joints
        theta = theta.reshape(self.num_joints, 3).to(J)
        R = rodrigues(theta)                                # (J,3,3) local rots

        # local transform of each joint relative to its parent
        rel_J = J.clone()
        rel_J[1:] = J[1:] - J[self.parents[1:]]
        T_loc = torch.zeros(self.num_joints, 4, 4, device=J.device, dtype=J.dtype)
        T_loc[:, :3, :3] = R
        T_loc[:, :3, 3] = rel_J
        T_loc[:, 3, 3] = 1.0

        # accumulate global transforms G_b = G_parent @ T_loc_b
        G = [T_loc[0]]
        for i in range(1, self.num_joints):
            G.append(G[self.parents[i]] @ T_loc[i])
        G = torch.stack(G, dim=0)                           # (J,4,4)

        # A_b = G_b - [0 | G_b @ [J_b; 0]]  (subtract rest joint, A.5)
        A = G.clone()
        A[:, :3, 3] = A[:, :3, 3] - (G[:, :3, :3] @ J.unsqueeze(-1)).squeeze(-1)
        return A, J

    # ---- forward LBS (used only for the validation gate) ----------------
    def lbs(self, theta: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """Articulate the rest mesh: v_posed = sum_b w_b A_b [v;1]."""
        v_rest, _ = self.rest_state(betas)
        A, _ = self.bone_transforms(theta, betas)
        T = torch.einsum("vj,jik->vik", self.lbs_weights, A)   # (V,4,4)
        v_h = torch.cat([v_rest, torch.ones_like(v_rest[:, :1])], dim=-1)
        return torch.einsum("vik,vk->vi", T, v_h)[:, :3]


def validate_against_smplx(model_path: str, gender: str = "neutral",
                           device: str = "cpu") -> float:
    """Max vertex error (m) of our LBS vs smplx on a random pose.

    Expect a few mm because we skip `posedirs`; a large error means a bug in
    `bone_transforms` (parent order, -J_b subtraction, axis-angle, etc.).
    Returns the max error, or -1.0 if smplx / the model cannot be loaded.
    """
    try:
        import smplx
        import os
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        sm = smplx.create(model_dir, model_type="smpl", gender=gender,
                          use_pca=False).to(device)
    except Exception as e:  # noqa
        print(f"[validate] smplx unavailable ({e}); skipping.")
        return -1.0

    body = SMPLBody(model_path).to(device)
    torch.manual_seed(0)
    betas = torch.randn(10, device=device) * 0.0          # zero shape for a clean test
    body_pose = torch.randn(69, device=device) * 0.2      # small random pose
    theta = torch.cat([torch.zeros(3, device=device), body_pose])

    ours = body.lbs(theta, betas)
    out = sm(betas=betas[None], body_pose=body_pose[None],
             global_orient=torch.zeros(1, 3, device=device), pose2rot=True)
    ref = out.vertices[0]
    # NOTE: smplx applies pose blendshapes (posedirs) which we intentionally skip,
    # so a few-mm residual is expected and correct; a large error signals a bug.
    err = (ours - ref).norm(dim=-1).max().item()
    print(f"[validate] max |ours - smplx| = {err*1000:.3f} mm "
          f"(few mm expected from skipped posedirs)")
    return err
