import numpy as np
import json
import open3d as o3d
import math
import random
import torch
import numpy as np
from typing import Optional
import torch_geometric.transforms as T
from torch_geometric.data import Data


def subsample(
    data: Data,
    upper_bound: float = 1.0,
    lower_bound: float = 0.5,
    max_points: Optional[int] = None,
    extra_fields: list[str] = [],
) -> Data:
    r"""Subsamples the point cloud to a random number of points within the
    range :obj:`[lower_bound, upper_bound]` (functional name: :obj:`subsample`).
    """
    if data.pos.size(0) == 0:
        return data
    num_points = int(random.uniform(lower_bound, upper_bound) * data.pos.size(0))
    if max_points is not None:
        num_points = min(num_points, max_points)
    idx = torch.randperm(data.pos.size(0))[:num_points]
    data.pos = data.pos[idx]
    for field in extra_fields:
        if hasattr(data, field):
            setattr(data, field, getattr(data, field)[idx])
    return data


def numpy_normalize_and_scale(xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
    r"""Normalizes the point cloud in such a way that the points are centered
    around the origin and are within the interval :math:`[-1, 1]` (functional
    name: :obj:`normalize`).
    """
    center = xyz.mean(0)
    scale = (1 / np.max(np.abs(xyz - center))) * 0.999999
    xyz = numpy_normalize_and_scale_with_params(xyz, center, scale)
    return xyz, center, scale


def numpy_normalize_and_scale_with_params(
    xyz: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    r"""Normalizes the point cloud in such a way that the points are centered
    around the origin and are within the interval :math:`[-1, 1]` (functional
    name: :obj:`normalize`).
    """
    if xyz.size == 0:
        return xyz
    shape = xyz.shape
    return ((xyz.reshape(-1, shape[-1]) - center) * scale).reshape(shape)


def normalize_and_scale(data: Data, extra_fields: list[str] = []) -> Data:
    r"""Centers and normalizes the given fields to the interval :math:`[-1, 1]`
    (functional name: :obj:`normalize_scale`).
    """
    if data.pos.size(0) == 0:
        data.center = torch.empty(0)
        data.scale = torch.empty(0)
        return data
    # center the pos points
    center = data.pos.mean(dim=-2, keepdim=True)
    # scale the pos points
    scale = (1 / (data.pos - center).abs().max()) * 0.999999

    return normalize_and_scale_with_params(data, center, scale, extra_fields)


def reverse_normalize_and_scale(data: Data, extra_fields: list[str] = []) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    assert hasattr(data, "center") and hasattr(
        data, "scale"
    ), "Data object does not contain the center and scale attributes."
    return reverse_normalize_and_scale_with_params(
        data, data.center, data.scale, extra_fields
    )


def normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    if data.pos.size(0) == 0:
        data.center = torch.empty(0)
        data.scale = torch.empty(0)
        return data
    data.pos = (data.pos - center) * scale
    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) - center) * scale,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = center
    data.scale = scale
    return data


def reverse_normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    # Reverse the scaling and centering of the pos points
    data.pos = data.pos / scale + center

    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) / scale) + center,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = torch.empty(0)
    data.scale = torch.empty(0)
    return data


def reverse_normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    # Reverse the scaling and centering of the pos points
    data.pos = data.pos / scale + center

    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) / scale) + center,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = torch.empty(0)
    data.scale = torch.empty(0)
    return data


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to quaternions.
    Convention: XYZ intrinsic rotations.
    roll  = rotation about x
    pitch = rotation about y
    yaw   = rotation about z
    """
    if not isinstance(roll, torch.Tensor):
        roll = torch.tensor(roll, dtype=torch.float32)
    if not isinstance(pitch, torch.Tensor):
        pitch = torch.tensor(pitch, dtype=torch.float32)
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.tensor(yaw, dtype=torch.float32)

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack((w, x, y, z), dim=-1)


def quat_normalize(q, eps=1e-8):
    """Normalize quaternion(s) q (...,4) to unit length."""
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_mul(q1, q2):
    """Hamilton product: q = q1 ⊗ q2. Shapes (...,4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def rotate_local_180(q: torch.Tensor, axis_idx: str) -> torch.Tensor:
    """
    Rotate quaternion q by +180 degrees around its local axis (x, y, or z).
    q is in (w, x, y, z) format.
    """
    if axis_idx == "x":
        # 180° about X axis => (w=0, x=1, y=0, z=0)
        qx180 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
        return quat_mul(q, qx180)  # post-multiply = local rotation
    elif axis_idx == "y":
        # 180° about Y axis => (w=0, x=0, y=1, z=0)
        qy180 = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=q.dtype, device=q.device)
        return quat_mul(q, qy180)  # post-multiply = local rotation
    elif axis_idx == "z":
        # 180° about Z axis => (w=0, x=0, y=0, z=1)
        qz180 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
        return quat_mul(q, qz180)  # post-multiply = local rotation
    else:
        raise ValueError("axis_idx must be 'x', 'y', or 'z'")


def quat_rotate_points(q, pts, normalize=True):
    """
    Rotate 3D point(s) with quaternion(s).
    q:   (..., 4)  quaternions (w, x, y, z)
    pts: (..., 3)  3D vectors
    returns rotated points with shape broadcast of q/pts over last dims.
    """
    if normalize:
        q = quat_normalize(q)

    w = q[..., :1]  # shape (...,1)
    v = q[..., 1:]  # shape (...,3)
    p = pts

    # Efficient formula (avoids constructing pure quaternion and two ⊗):
    # p' = p + 2*w*(v × p) + 2*(v × (v × p))
    # where × is cross product.
    cross1 = torch.linalg.cross(v, p)
    term1 = 2.0 * w * cross1
    cross2 = torch.linalg.cross(v, cross1)
    term2 = 2.0 * cross2
    return p + term1 + term2


def quat_compose(r1, r2, normalize=True):
    """
    Compose rotations as quaternions.
    Returns q = r2 ⊗ r1, meaning: apply r1, then r2.
    (Active rotations with column vectors.)
    """
    if normalize:
        r1 = quat_normalize(r1)
        r2 = quat_normalize(r2)
    q = quat_mul(r2, r1)
    if normalize:
        q = quat_normalize(q)
    return q
