"""Tiny shared geometry helpers (quaternion <-> rotation), (w,x,y,z) convention."""
from __future__ import annotations
import torch


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit (or unnormalised) quaternion (...,4) in (w,x,y,z) -> rotation (...,3,3)."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation (3,3) -> axis-angle (3,). Used to compose a yaw into Rh for the demo."""
    angle = torch.acos(((R.trace() - 1.0) * 0.5).clamp(-1.0, 1.0))
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = axis.norm()
    if n < 1e-8:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    return axis / n * angle

