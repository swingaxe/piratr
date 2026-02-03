import torch
import trimesh
import numpy as np

##  GRIPPER START
GRIPPER_MESH_MAIN = trimesh.load("piratr/objects/gripper_main.obj")
GRIPPER_MESH_SIDE1 = trimesh.load("piratr/objects/gripper_side1.obj")
GRIPPER_MESH_SIDE2 = trimesh.load("piratr/objects/gripper_side2.obj")
# Local translation of side 1 jaw.
LOCAL_T1 = np.array([-0.00573, -0.031728, -0.326143])
LOCAL_T1_TORCH = torch.tensor(LOCAL_T1, dtype=torch.float32)
# Local translation of side 2 jaw.
LOCAL_T2 = np.array([0.008859, -0.031725, 0.325861])
LOCAL_T2_TORCH = torch.tensor(LOCAL_T2, dtype=torch.float32)
# Jaw closed angle in degrees.
CLOSED_RANGE = 163 - 20
# Jaw open angle in degrees.
OPEN_RANGE = 0
## GRIPPER END

LOADING_PLATFORM_MESH = trimesh.load("piratr/objects/loading_platform.obj")
PALLET_MESH = trimesh.load("piratr/objects/pallet.obj")


def sample_gripper_points(num_points: list[int] = [16, 24, 24]) -> tuple[torch.Tensor]:
    points_main_full = torch.from_numpy(GRIPPER_MESH_MAIN.sample(2048)).to(
        torch.float32
    )  # (2048, 3)
    points_s1_full = torch.from_numpy(GRIPPER_MESH_SIDE1.sample(2048)).to(
        torch.float32
    )  # (2048, 3)
    points_s2_full = torch.from_numpy(GRIPPER_MESH_SIDE2.sample(2048)).to(
        torch.float32
    )  # (2048, 3)

    assert len(num_points) == 3, "num_points should be a list of 3 integers"
    points_main, _ = farthest_point_sample(points_main_full, num_points[0])
    points_s1, _ = farthest_point_sample(points_s1_full, num_points[1])
    points_s2, _ = farthest_point_sample(points_s2_full, num_points[2])

    return points_main, points_s1, points_s2


def sample_loading_platform_points(num_points: int = 64) -> torch.Tensor:
    loading_points_full = torch.from_numpy(LOADING_PLATFORM_MESH.sample(2048)).to(
        torch.float32
    )  # (2048, 3)

    loading_points, _ = farthest_point_sample(loading_points_full, num_points)

    return loading_points


def sample_pallet_points(num_points: int = 64) -> torch.Tensor:
    pallet_points_full = torch.from_numpy(PALLET_MESH.sample(2048)).to(
        torch.float32
    )  # (2048, 3)

    pallet_points, _ = farthest_point_sample(pallet_points_full, num_points)

    return pallet_points


def farthest_point_sample(points, npoint):
    """
    Input:
        points: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, D = points.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=points.device)
    distance = torch.ones(N, device=points.device) * 1e10
    farthest = torch.randint(0, N, (1,), device=points.device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return points[centroids], centroids


def quat_to_rotmat_batch(q: torch.Tensor, assumes="wxyz") -> torch.Tensor:
    """
    q: [B,4]
    returns R: [B,3,3] (active, column-vector convention)
    """
    if assumes == "xyzw":
        q = q[..., [3, 0, 1, 2]]

    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)

    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
