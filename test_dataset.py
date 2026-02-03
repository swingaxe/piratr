import torch
import argparse
import numpy as np
import polyscope as ps
from piratr import (
    build_dataset_config,
    build_dataset,
    load_args,
)

from piratr.objects import build_gripper, build_loading_platform, build_pallet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="path to config"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="dataset to visualize",
    )
    return parser.parse_args()


def visualize_gt_data(data):
    ps.init()
    ps.remove_all_structures()
    points = data.pos.cpu().numpy()

    pc = ps.register_point_cloud(
        "Point Cloud", points, color=(0.5, 0.5, 0.5), radius=0.001, material="wax"
    )
    scale = 1.0
    if hasattr(data, "scale"):
        scale = data.scale.cpu().item()

    for i in range(data.y_params.shape[0]):
        if data.y_cls[i] == 1:  # gripper class
            mesh_gripper = build_gripper(
                pos=data.y_params[i, :3].cpu().numpy(),
                quat=data.y_params[i, 3:7].cpu().numpy(),
                jaw_opening=(data.y_params[i, 7].cpu().numpy() + 1) / 2,
                scale=scale,
            )
            ps.register_surface_mesh(
                f"Gripper_{i}",
                mesh_gripper.vertices,
                mesh_gripper.faces,
                color=(0.76, 0.83, 0.21),
                material="wax",
            )

            ps.register_point_cloud(
                f"Gripper Points {i}",
                data.object_points[i].cpu().numpy(),
                radius=0.002,
                color=(0.76, 0.83, 0.21),
                material="wax",
                enabled=False,
            )
        elif data.y_cls[i] == 2:  # loading platform class
            mesh_loading_platform = build_loading_platform(
                pos=data.y_params[i, :3].cpu().numpy(),
                quat=data.y_params[i, 3:7].cpu().numpy(),
                scale=scale,
            )
            ps.register_surface_mesh(
                f"Loading Platform_{i}",
                mesh_loading_platform.vertices,
                mesh_loading_platform.faces,
                color=(0.21, 0.76, 0.83),
                material="wax",
            )

            ps.register_point_cloud(
                f"Loading Platform Points {i}",
                data.object_points[i].cpu().numpy(),
                radius=0.002,
                color=(0.21, 0.76, 0.83),
                material="wax",
                enabled=False,
            )
        elif data.y_cls[i] == 3:  # pallet class
            mesh_pallet = build_pallet(
                pos=data.y_params[i, :3].cpu().numpy(),
                quat=data.y_params[i, 3:7].cpu().numpy(),
                scale=scale,
            )
            ps.register_surface_mesh(
                f"Pallet_{i}",
                mesh_pallet.vertices,
                mesh_pallet.faces,
                color=(0.83, 0.21, 0.76),
                material="wax",
            )

            ps.register_point_cloud(
                f"Pallet Points {i}",
                data.object_points[i].cpu().numpy(),
                radius=0.002,
                color=(0.83, 0.21, 0.76),
                material="wax",
                enabled=False,
            )

    # up direction
    ps.set_up_dir("z_up")
    # ground plane
    ps.set_ground_plane_mode("none")

    ps.show()


def main():
    parsed_args = parse_args()
    args = load_args(parsed_args.config)
    if parsed_args.dataset == "train":
        split = args.data_root
    elif parsed_args.dataset == "val":
        split = args.data_val_root
    else:
        split = args.data_test_root
    dataset_config = build_dataset_config(args, split, augment=False)
    dataset = build_dataset(dataset_config)

    visualize_fn = visualize_gt_data

    for data in dataset:
        visualize_fn(data)


if __name__ == "__main__":
    main()
