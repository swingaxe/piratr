"""
Prediction script for PIRATR model.

This script loads a trained PIRATR model and performs inference on 3D point cloud data,
visualizing the detected curves and polylines.
"""

import os
import argparse
import numpy as np
import polyscope as ps
import pytorch_lightning as pl
import torch
import torch_geometric
import trimesh
from torch_geometric.data.data import Data
import fpsample
from piratr import (
    build_model,
    build_model_config,
    load_args,
    load_weights,
)


from piratr.dataset import normalize_and_scale
from piratr.objects import build_gripper, build_loading_platform, build_pallet


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict curves on point cloud data using PIRATR"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to input file (.ply, .obj, .pt, .xyz) or input folder.",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=32768,
        help="Number of points to sample from input point cloud (default: 32768)",
    )
    parser.add_argument(
        "--sample_mode",
        "-s",
        type=str,
        default="all",
        choices=["fps", "random", "uniform", "all"],
        help="Sampling method for point cloud subsampling",
    )
    parser.add_argument(
        "--reduction",
        "-r",
        type=int,
        default=0,
        help="Reduce points by random sampling before sample_mode sampling (default: 0, no reduction). Usable if you are using fps sampling to reduce runtime.",
    )
    parser.add_argument(
        "--distance_filter",
        "-d",
        type=float,
        default=0.0,
        help="Distance threshold for filtering points (default: 0.0, no filtering)",
    )

    return parser.parse_args()


def file_loader(file_path: str) -> Data:
    """
    Load point cloud data from various file formats.

    Args:
        file_path: Path to the input file

    Returns:
        Data object containing point positions

    Raises:
        ValueError: If file format is not supported
    """
    if file_path.endswith(".pt"):
        pos = torch.load(file_path, weights_only=False)["pos"]
        return Data(pos=pos)

    elif file_path.endswith(".ply"):
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.PointCloud):
            pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        else:
            # Sample points from mesh surface
            pos = torch.tensor(mesh.sample(100000), dtype=torch.float32)
        return Data(pos=pos)

    elif file_path.endswith(".obj"):
        mesh = trimesh.load(file_path, force="mesh")
        if isinstance(mesh, trimesh.PointCloud):
            pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        else:
            # Sample points from mesh surface
            pos = torch.tensor(mesh.sample(100000), dtype=torch.float32)
        return Data(pos=pos)

    elif file_path.endswith(".xyz"):
        points = np.loadtxt(file_path, dtype=np.float32)
        pos = torch.tensor(points[:, :3], dtype=torch.float32)
        return Data(pos=pos)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def process_data(
    data: Data,
    sample: int = 32768,
    sample_mode: str = "fps",
    reduction: int = 0,
    distance: float = 30.0,
) -> Data:
    """
    Process and subsample point cloud data.

    Args:
        data: Input point cloud data
        sample: Number of points to sample
        sample_mode: Sampling method ("fps", "random", "uniform", "all")
        reduction: number of points to reduce before sampling

    Returns:
        Processed and normalized data
    """
    if not hasattr(data, "pos"):
        return data

    if distance > 0:
        distances = torch.norm(data.pos, dim=1)
        mask = distances < distance
        data.pos = data.pos[mask]
        # Filter out points that are too close to the origin
        distances = distances[mask]
        data.pos = data.pos[distances > 1]

    # Apply reduction if specified
    if reduction > 0 and data.pos.size(0) > reduction:
        indices = torch.randperm(data.pos.size(0))[:reduction]
        data.pos = data.pos[indices]

    # Apply sampling strategy
    if sample_mode == "random":
        indices = torch.randperm(data.pos.size(0))[:sample]
        data.pos = data.pos[indices]

    elif sample_mode == "fps":
        if data.pos.size(0) > sample:
            idx = fpsample.bucket_fps_kdline_sampling(data.pos, sample, h=6)
            data.pos = data.pos[idx]

    elif sample_mode == "uniform":
        step = max(1, data.pos.size(0) // sample)
        data.pos = data.pos[::step][:sample]

    elif sample_mode == "all":
        pass  # Keep all points

    # Add batch information for single point cloud BEFORE normalization
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.pos.size(0), dtype=torch.long)
        data.batch_size = 1

    # Normalize and scale the points
    data = normalize_and_scale(data)

    # Ensure scale and center are proper batch tensors
    if hasattr(data, "scale") and data.scale.dim() == 0:
        data.scale = data.scale.unsqueeze(0)
    if hasattr(data, "center") and data.center.dim() == 1:
        data.center = data.center.unsqueeze(0)

    return data


def visualize(data) -> None:
    """
    Visualize detected objects and input point cloud using Polyscope.

    Args:
        data: Data object with fields:
              - pos: (N_pts, 3) tensor
              - object_class: (N_obj,) tensor[int] in {0,1,2,3}
              - object_score: (N_obj,) tensor[float]
              - active: (N_obj,) tensor[bool]
              - object_points: list of (M_i, 3) tensors
              - gripper_params / loading_platform_params / pallet_params
              - scale: scalar tensor/float
    """
    # ---- Polyscope base setup ----
    ps.init()
    ps.remove_all_structures()
    ps.remove_all_groups()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        return np.asarray(x)

    scale = float(to_np(data.scale))

    # Colors indexed by class id; class 0 is unused
    colors = [
        (0.0, 0.0, 0.0),  # dummy
        (0.76, 0.83, 0.21),  # class 1: gripper
        (0.21, 0.76, 0.83),  # class 2: loading platform
        (0.83, 0.21, 0.76),  # class 3: pallet
    ]

    # Class-specific configuration
    class_cfg = {
        1: {
            "name": "Gripper",
            "params_attr": "gripper_params",
            "build": lambda p: build_gripper(
                pos=p[:3], quat=p[3:7], jaw_opening=(p[7] + 1) / 2, scale=scale
            ),
        },
        2: {
            "name": "Loading_Platform",
            "params_attr": "loading_platform_params",
            "build": lambda p: build_loading_platform(
                pos=p[:3], quat=p[3:7], scale=scale
            ),
        },
        3: {
            "name": "Pallet",
            "params_attr": "pallet_params",
            "build": lambda p: build_pallet(pos=p[:3], quat=p[3:7], scale=scale),
        },
    }

    grp_active = ps.create_group("Objects")
    grp_inactive = ps.create_group("Inactive Objects")

    # ---- Helper to register a mesh + its points ----
    def register_object(cls_id: int, idx: int):
        name = class_cfg[cls_id]["name"]
        is_active = bool(to_np(data.active[idx]))
        base_color = colors[cls_id]
        color = base_color if is_active else (1.0, 0.0, 0.0)
        alpha = 1.0 if is_active else 0.5
        group = grp_active if is_active else grp_inactive

        # Build mesh
        params = getattr(data, class_cfg[cls_id]["params_attr"])[idx]
        mesh = class_cfg[cls_id]["build"](to_np(params))

        mesh_handle = ps.register_surface_mesh(
            f"{name}_{idx}_{float(to_np(data.object_score[idx])):.4f}",
            to_np(mesh.vertices),
            to_np(mesh.faces),
            color=color,
            transparency=alpha,
            material="wax",
        )
        group.add_child_structure(mesh_handle)

        # # Per-object point cloud
        # ps.register_point_cloud(
        #     f"{name}_{idx}_Points_{float(to_np(data.object_score[idx])):.4f}",
        #     to_np(data.object_points[idx]),
        #     radius=0.002,
        #     color=color,
        #     transparency=alpha,
        #     material="wax",
        #     enabled=False,
        # )

    # ---- Iterate objects ----
    for i, cls in enumerate(to_np(data.object_class)):
        cls = int(cls)
        if cls in class_cfg:
            register_object(cls, i)

    # ---- Input point cloud ----
    ps.register_point_cloud(
        "Input Points",
        to_np(data.pos),
        radius=0.001,
        color=(0.5, 0.5, 0.5),
        material="wax",
    )

    ps.reset_camera_to_home_view()
    ps.show()


def predict(model: pl.LightningModule, data: Data) -> Data:
    """Run prediction on the input data."""
    # Run inference
    data = data.to("cuda")
    output = model.predict_step(
        data, score_thresholds=[0.95, 0.98, 0.95], reverse_norm=True
    )

    return output[0]


@torch.no_grad()
def main():
    """Main prediction pipeline."""
    # Set seeds for reproducibility
    pl.seed_everything(42)
    torch_geometric.seed_everything(42)

    # Parse arguments
    parsed_args = parse_args()
    args = load_args(parsed_args.config)

    if not parsed_args.path:
        raise ValueError("Please provide a path using --path argument.")

    # Build and load model
    model_config = build_model_config(args)
    model = build_model(model_config)
    load_weights(model, parsed_args.checkpoint)
    model.eval()
    model.to("cuda")

    # check if input is a file or directory
    if os.path.isdir(parsed_args.path):
        # Load all files in the directory

        for filename in sorted(os.listdir(parsed_args.path)):
            if not filename.endswith((".ply", ".obj", ".pt", ".xyz")):
                print("Skipping unsupported file:", filename)
                continue
            print(filename)
            file_data = file_loader(os.path.join(parsed_args.path, filename))
            processed_data = process_data(
                file_data,
                sample=parsed_args.samples,
                sample_mode=parsed_args.sample_mode,
                reduction=parsed_args.reduction,
                distance=parsed_args.distance_filter,
            )
            output_data = predict(model, processed_data)

            visualize(output_data)
    else:
        # Load single file
        file_data = file_loader(parsed_args.path)
        processed_data = process_data(
            file_data,
            sample=parsed_args.samples,
            sample_mode=parsed_args.sample_mode,
            reduction=parsed_args.reduction,
            distance=parsed_args.distance_filter,
        )
        output_data = predict(model, processed_data)

        visualize(output_data)


if __name__ == "__main__":
    main()
