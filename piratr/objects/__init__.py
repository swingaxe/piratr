import torch
import trimesh
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from .utils import (
    quat_to_rotmat_batch,
    sample_gripper_points,
    GRIPPER_MESH_MAIN,
    GRIPPER_MESH_SIDE1,
    GRIPPER_MESH_SIDE2,
    LOCAL_T1,
    LOCAL_T1_TORCH,
    LOCAL_T2,
    LOCAL_T2_TORCH,
    CLOSED_RANGE,
    OPEN_RANGE,
    LOADING_PLATFORM_MESH,
    sample_loading_platform_points,
    PALLET_MESH,
    sample_pallet_points,
)


@dataclass(frozen=True)
class ObjectInfo:
    num_points: int
    gripper_main_points: torch.Tensor
    gripper_s1_points: torch.Tensor
    gripper_s2_points: torch.Tensor
    loading_platform_points: torch.Tensor
    pallet_points: torch.Tensor


OBJECT_INFO = ObjectInfo(
    64,
    *sample_gripper_points([16, 24, 24]),
    sample_loading_platform_points(64),
    sample_pallet_points(64)
)


def build_gripper(
    pos,
    quat,
    jaw_opening,
    scale=1.0,
) -> trimesh.Trimesh:
    """
    Build a posed gripper mesh given position, orientation (quaternion),
    jaw opening, and scale.

    Args:
        pos (array-like): 3D position of gripper base (shape [3]).
        quat (array-like): Quaternion [w, x, y, z].
        jaw_opening (float): Normalized jaw opening in [0,1].
        scale (float): Global scale factor (default = 1.0).

    Returns:
        trimesh.Trimesh: Combined posed mesh.
    """

    # --- rotation from quaternion (SciPy expects [x,y,z,w]) ---
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    rot_mat = r.as_matrix()

    # --- copy meshes ---
    mesh_main = GRIPPER_MESH_MAIN.copy()
    mesh_s1 = GRIPPER_MESH_SIDE1.copy()
    mesh_s2 = GRIPPER_MESH_SIDE2.copy()

    # --- apply scale about origin before anything else ---
    S = np.eye(4)
    S[:3, :3] *= scale
    for m in (mesh_main, mesh_s1, mesh_s2):
        m.apply_transform(S)

    # also scale local translations
    local_t1 = LOCAL_T1 * scale
    local_t2 = LOCAL_T2 * scale

    # --- jaw opening angle ---
    angle = CLOSED_RANGE + (OPEN_RANGE - CLOSED_RANGE) * jaw_opening
    jaw_angle_rad = np.deg2rad(angle)

    # --- helper transforms ---
    def apply_child_transform(child_mesh, local_translation, parent_rot, parent_pos):
        transform = np.eye(4)
        transform[:3, :3] = parent_rot
        transform[:3, 3] = parent_pos
        child_t = np.eye(4)
        child_t[:3, 3] = local_translation
        child_mesh.apply_transform(transform @ child_t)

    def rotate_child_mesh(child_mesh, angle_rad, axis=(1, 0, 0)):
        rot = R.from_rotvec(np.array(axis) * angle_rad).as_matrix()
        R_mat = np.eye(4)
        R_mat[:3, :3] = rot
        child_mesh.apply_transform(R_mat)

    # --- apply transforms ---
    mesh_main.apply_transform(
        np.vstack([np.column_stack((rot_mat, pos)), [0, 0, 0, 1]])
    )

    rotate_child_mesh(mesh_s1, -jaw_angle_rad)
    rotate_child_mesh(mesh_s2, jaw_angle_rad)

    apply_child_transform(mesh_s1, local_t1, rot_mat, pos)
    apply_child_transform(mesh_s2, local_t2, rot_mat, pos)

    # --- combine meshes ---
    return trimesh.util.concatenate([mesh_main, mesh_s1, mesh_s2])


def build_loading_platform(
    pos,
    quat,
    scale=1.0,
) -> trimesh.Trimesh:
    """
    Build a posed loading platform mesh given position, orientation (quaternion), and scale.

    Args:
        pos (array-like): 3D position of platform base (shape [3]).
        quat (array-like): Quaternion [w, x, y, z].
        scale (float): Global scale factor (default = 1.0).

    Returns:
        trimesh.Trimesh: Posed mesh.
    """
    return build_mesh(LOADING_PLATFORM_MESH, pos, quat, scale)


def build_pallet(
    pos,
    quat,
    scale=1.0,
) -> trimesh.Trimesh:
    """
    Build a posed pallet mesh given position, orientation (quaternion), and scale.

    Args:
        pos (array-like): 3D position of pallet base (shape [3]).
        quat (array-like): Quaternion [w, x, y, z].
        scale (float): Global scale factor (default = 1.0).

    Returns:
        trimesh.Trimesh: Posed mesh.
    """
    return build_mesh(PALLET_MESH, pos, quat, scale)


def build_mesh(
    mesh_template: trimesh.Trimesh,
    pos,
    quat,
    scale=1.0,
) -> trimesh.Trimesh:
    """
    Build a posed mesh (platform or pallet) given position, orientation (quaternion), and scale.

    Args:
        mesh_template (trimesh.Trimesh): The mesh to copy and transform.
        pos (array-like): 3D position of base (shape [3]).
        quat (array-like): Quaternion [w, x, y, z].
        scale (float): Global scale factor (default = 1.0).

    Returns:
        trimesh.Trimesh: Posed mesh.
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    rot_mat = r.as_matrix()

    mesh = mesh_template.copy()

    # Apply scale about origin
    S = np.eye(4)
    S[:3, :3] *= scale
    mesh.apply_transform(S)

    # Apply rotation and translation
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    mesh.apply_transform(T)

    return mesh


def transform_gripper_points(
    gripper_pos: torch.Tensor,  # [B, 3]
    gripper_quat: torch.Tensor,  # [B, 4]
    gripper_opening: torch.Tensor,  # [B]
    points_main: torch.Tensor,  # [Nm, 3] or [B, Nm, 3]
    points_s1: torch.Tensor,  # [Ns, 3] or [B, Ns, 3]
    points_s2: torch.Tensor,  # [Ns, 3] or [B, Ns, 3]
    scale: torch.Tensor,  # [B]
) -> torch.Tensor:
    device = gripper_pos.device
    B = gripper_pos.shape[0]

    # --- opening angle (in radians) ---
    angle = CLOSED_RANGE + (OPEN_RANGE - CLOSED_RANGE) * gripper_opening
    angle_rad = angle * torch.pi / 180.0
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)

    # --- global rotation from quaternion ---
    R = quat_to_rotmat_batch(gripper_quat)  # [B,3,3]

    # --- ensure batched point clouds ---
    def ensure_batch(pts, B):
        if pts.dim() == 2:  # [N,3]
            pts = pts.unsqueeze(0).expand(B, -1, -1)
        return pts.to(device)

    points_main = ensure_batch(points_main, B)
    points_s1 = ensure_batch(points_s1, B)
    points_s2 = ensure_batch(points_s2, B)

    # --- scale in local space ---
    points_main = points_main * scale[:, None, None]
    points_s1 = points_s1 * scale[:, None, None]
    points_s2 = points_s2 * scale[:, None, None]

    # --- jaw local rotation (rotation about x-axis) ---
    R_jaw_s1 = torch.stack(
        [
            torch.stack(
                [torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1
            ),
            torch.stack([torch.zeros_like(c), c, -s], dim=-1),
            torch.stack([torch.zeros_like(c), s, c], dim=-1),
        ],
        dim=-2,
    )  # [B,3,3]
    R_jaw_s2 = R_jaw_s1.transpose(1, 2)  # symmetric opposite

    # --- apply local jaw rotations ---
    points_s1 = points_s1 @ R_jaw_s1  # .transpose(1, 2)
    points_s2 = points_s2 @ R_jaw_s2  # .transpose(1, 2)

    # --- scaled + rotated local translations for jaws ---
    local_t1 = LOCAL_T1_TORCH.to(device)[None, :] * scale[:, None]  # [B,3]
    local_t2 = LOCAL_T2_TORCH.to(device)[None, :] * scale[:, None]  # [B,3]
    local_t1_world = (R @ local_t1[:, :, None]).squeeze(-1)  # [B,3]
    local_t2_world = (R @ local_t2[:, :, None]).squeeze(-1)  # [B,3]

    # --- transform everything into world ---
    points_main_final = (points_main @ R.transpose(1, 2)) + gripper_pos[:, None, :]
    points_s1_final = (
        (points_s1 @ R.transpose(1, 2))
        + gripper_pos[:, None, :]
        + local_t1_world[:, None, :]
    )
    points_s2_final = (
        (points_s2 @ R.transpose(1, 2))
        + gripper_pos[:, None, :]
        + local_t2_world[:, None, :]
    )

    return torch.cat([points_main_final, points_s1_final, points_s2_final], dim=1)


def transform_loading_or_pallet_points(
    pos: torch.Tensor,  # [B, 3]
    quat: torch.Tensor,  # [B, 4]
    points: torch.Tensor,  # [Np, 3] or [B, Np, 3]
    scale: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Transform points for platform or pallet: scale, rotate, translate.

    Args:
        pos: [B, 3] batch of positions
        quat: [B, 4] batch of quaternions
        points: [Np, 3] or [B, Np, 3] points
        scale: [B] batch of scales

    Returns:
        [B, Np, 3] transformed points
    """
    device = pos.device
    B = pos.shape[0]

    R = quat_to_rotmat_batch(quat)  # [B,3,3]

    def ensure_batch(pts, B):
        if pts.dim() == 2:
            pts = pts.unsqueeze(0).expand(B, -1, -1)
        return pts.to(device)

    points = ensure_batch(points, B)
    points = points * scale[:, None, None]
    points_final = (points @ R.transpose(1, 2)) + pos[:, None, :]
    return points_final
