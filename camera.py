"""Pinhole camera. Because our rasterizer is hand-written we use the plain
OpenCV/SMPL convention directly (no OpenGL projection matrix), which natively
supports an off-centre principal point (c_x, c_y) — important for ZJU-MoCap.

R, t map world -> camera (x_cam = R x_world + t); the camera looks down +z with
y pointing down. The rasterizer (B.1) consumes R, t and (fx,fy,cx,cy) directly.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class Camera:
    R: torch.Tensor      # (3,3) world->camera rotation
    t: torch.Tensor      # (3,)  world->camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int

    @property
    def campos(self) -> torch.Tensor:
        """Camera centre in world coordinates: c = -R^T t."""
        return -self.R.transpose(-1, -2) @ self.t

    def to(self, device):
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        return self


def make_camera(K, R, T, scale: float = 1.0, t_in_mm: bool = False,
                W: int = None, H: int = None, device: str = "cpu") -> Camera:
    """Build a Camera from ZJU intrinsics/extrinsics, applying a resize `scale`.

    Args:
        K: (3,3) intrinsics at full resolution.
        R: (3,3) world->camera. T: (3,) or (3,1) world->camera translation.
        scale: image resize factor (intrinsics scale with it; principal point too).
        t_in_mm: divide T by 1000 if ZJU stored it in millimetres.
        W, H: full-resolution image size (the returned camera uses scaled size).
    """
    K = torch.as_tensor(K, dtype=torch.float32)
    R = torch.as_tensor(R, dtype=torch.float32)
    T = torch.as_tensor(T, dtype=torch.float32).reshape(3)
    if t_in_mm:
        T = T / 1000.0
    fx, fy = K[0, 0].item() * scale, K[1, 1].item() * scale
    cx, cy = K[0, 2].item() * scale, K[1, 2].item() * scale
    Ws = int(round(W * scale)) if W else None
    Hs = int(round(H * scale)) if H else None
    return Camera(R=R, t=T, fx=fx, fy=fy, cx=cx, cy=cy, W=Ws, H=Hs).to(device)
