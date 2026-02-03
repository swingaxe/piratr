import json
import random
import torch
import math
import numpy as np
from typing import Union
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData, Data
from dataclasses import dataclass
import torch_geometric.transforms as T
import fpsample

from piratr.objects import (
    OBJECT_INFO,
    transform_gripper_points,
    transform_loading_or_pallet_points,
)

from .utils import (
    normalize_and_scale,
    subsample,
    euler_to_quaternion,
    quat_normalize,
    quat_rotate_points,
    quat_compose,
)


@dataclass
class DatasetConfig:
    dataset: str
    root: str
    augment: bool = False
    random_rotate_prob: float = 1.0
    random_rotate_angles: tuple[float, float, float] = (5, 5, 180)
    random_sample_prob: float = 0.5
    random_sample_bounds: tuple[float, float] = (1, 0.5)
    noise_prob: float = 0
    noise_scale_range: tuple[float, float] = (0, 0.04)
    min_gripper_points: int = 20
    min_loading_platform_points: int = 50
    min_pallet_points: int = 50
    sample_points: int = 0
    max_distance: float = 0.0  # in meters, 0 means no


class SynthDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
    ) -> None:
        self.file_names = self._read_file_names(config.root)
        self.config = config
        self.gripper_main_points = OBJECT_INFO.gripper_main_points
        self.gripper_s1_points = OBJECT_INFO.gripper_s1_points
        self.gripper_s2_points = OBJECT_INFO.gripper_s2_points
        self.loading_platform_points = OBJECT_INFO.loading_platform_points
        self.pallet_points = OBJECT_INFO.pallet_points

        super().__init__(
            config.root,
            None,
            None,
            None,
        )

    def get(self, idx: int) -> BaseData:

        # Read Input Data
        pc = np.loadtxt(Path(self.processed_dir) / f"{self.raw_file_names[idx]}.xyz")
        pc = torch.from_numpy(pc).to(torch.float32)

        # Preprocessing and handling both synth and real data
        pc = self._distance_filter(pc, self.config.max_distance)
        pc = self._align_rotation(pc)
        pc = self._fps_sample(pc, self.config.sample_points)

        with open(
            Path(self.processed_dir) / f"{self.raw_file_names[idx]}_meta.json", "r"
        ) as f:
            annos = json.load(f)

            if annos.get("gripper") is None:
                enable_gripper = False
            else:
                gripper_pos = torch.tensor(
                    annos["gripper"]["pos"], dtype=torch.float32
                ).unsqueeze(0)
                gripper_rot = torch.tensor(
                    annos["gripper"]["quat"], dtype=torch.float32
                ).unsqueeze(0)
                gripper_opening = torch.tensor(
                    annos["gripper"]["jaw_opening"], dtype=torch.float32
                ).unsqueeze(0)

                if annos["gripper"].get("gripper_points") is None:
                    enable_gripper = True
                else:
                    num_gripper_points = annos["gripper"]["gripper_points"]
                    enable_gripper = (
                        num_gripper_points >= self.config.min_gripper_points
                    )
            if not enable_gripper:
                gripper_pos = torch.zeros((1, 3), dtype=torch.float32)
                gripper_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(
                    0
                )
                gripper_opening = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)

            if annos.get("loading_platform") is None:
                enable_loading_platform = False
            else:
                loading_platform_pos = torch.tensor(
                    annos["loading_platform"]["pos"], dtype=torch.float32
                ).unsqueeze(0)
                loading_platform_rot = torch.tensor(
                    annos["loading_platform"]["quat"], dtype=torch.float32
                ).unsqueeze(0)

                if annos["loading_platform"].get("loading_platform_points") is None:
                    enable_loading_platform = True
                else:
                    num_loading_platform_points = annos["loading_platform"][
                        "loading_platform_points"
                    ]
                    enable_loading_platform = (
                        num_loading_platform_points
                        >= self.config.min_loading_platform_points
                    )

            if not enable_loading_platform:
                loading_platform_pos = torch.zeros((1, 3), dtype=torch.float32)
                loading_platform_rot = torch.tensor(
                    [1, 0, 0, 0], dtype=torch.float32
                ).unsqueeze(0)
            # TODO: visibility can be computed inside the list comprehensions

            pallet_pos = [
                torch.tensor(pallet["pos"], dtype=torch.float32).unsqueeze(0)
                for pallet in annos.get("pallets", [])
                if (
                    pallet.get("pallet_points") is None
                    or pallet["pallet_points"] > self.config.min_pallet_points
                )
            ]
            pallet_rot = [
                torch.tensor(pallet["quat"], dtype=torch.float32).unsqueeze(0)
                for pallet in annos.get("pallets", [])
                if (
                    pallet.get("pallet_points") is None
                    or pallet["pallet_points"] > self.config.min_pallet_points
                )
            ]
            num_pallets = len(pallet_pos)
            if num_pallets > 0:
                pallet_pos = torch.cat(pallet_pos, dim=0)
                pallet_rot = torch.cat(pallet_rot, dim=0)
            else:
                pallet_pos = torch.zeros((1, 3), dtype=torch.float32)
                pallet_rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(
                    0
                )

        #  Noise Augmentation
        augment = self.config.augment
        if augment and random.random() < self.config.noise_prob:
            low, high = self.config.noise_scale_range
            std = torch.empty(1).uniform_(low, high).item()
            noise = torch.normal(
                mean=0,
                std=std,
                size=pc.shape,
                dtype=pc.dtype,
                device=pc.device,
            )
            pc += noise

        # Rotation Augmentation
        if augment and random.random() < self.config.random_rotate_prob:
            random_quat = self._random_quat(self.config.random_rotate_angles)
            pc = quat_rotate_points(random_quat, pc)
            gripper_pos = quat_rotate_points(random_quat, gripper_pos)
            gripper_rot = quat_compose(gripper_rot, random_quat)
            loading_platform_pos = quat_rotate_points(random_quat, loading_platform_pos)
            loading_platform_rot = quat_compose(loading_platform_rot, random_quat)
            pallet_pos = quat_rotate_points(random_quat, pallet_pos)
            pallet_rot = quat_compose(pallet_rot, random_quat)

        # Create Data object
        labels = []
        if enable_gripper:
            labels.append(1)
        if enable_loading_platform:
            labels.append(2)
        for i in range(num_pallets):
            labels.append(3)
        data = Data(
            pos=pc,
            gripper_pos=gripper_pos,
            gripper_rot=gripper_rot,
            gripper_opening=gripper_opening,
            loading_platform_pos=loading_platform_pos,
            loading_platform_rot=loading_platform_rot,
            pallet_pos=pallet_pos,
            pallet_rot=pallet_rot,
            object_points=torch.zeros(
                len(labels),
                OBJECT_INFO.num_points,
                3,
            ),
            num_objects=len(labels),
            y_cls=torch.tensor(labels, dtype=torch.long),
        )

        if enable_gripper:
            data.object_points[data.y_cls == 1] = transform_gripper_points(
                data.gripper_pos,
                data.gripper_rot,
                data.gripper_opening,
                self.gripper_main_points,
                self.gripper_s1_points,
                self.gripper_s2_points,
                torch.tensor([1]),
            )
        if enable_loading_platform:
            data.object_points[data.y_cls == 2] = transform_loading_or_pallet_points(
                data.loading_platform_pos,
                data.loading_platform_rot,
                self.loading_platform_points,
                torch.tensor([1]),
            )
        if num_pallets > 0:
            data.object_points[data.y_cls == 3] = transform_loading_or_pallet_points(
                data.pallet_pos,
                data.pallet_rot,
                self.pallet_points,
                torch.tensor(1).repeat(num_pallets),
            )

        # Normalization
        data = self._normalize(data)

        # Build Training Targets
        data.y_params = torch.zeros(data.num_objects, 8, dtype=torch.float32)
        pallet_index_adjust = sum([enable_gripper, enable_loading_platform])
        for i in range(data.num_objects):
            if data.y_cls[i] == 1:
                # gripper: pos(3), quat(4), opening(1)
                data.y_params[i][:3] = data.gripper_pos[0]
                data.y_params[i][3:7] = data.gripper_rot[0]
                data.y_params[i][7] = data.gripper_opening[0]
            if data.y_cls[i] == 2:
                # loading platform: pos(3), quat(4)
                data.y_params[i][:3] = data.loading_platform_pos[0]
                data.y_params[i][3:7] = data.loading_platform_rot[0]
            if data.y_cls[i] == 3:
                # pallet: pos(3), quat(4)
                data.y_params[i][:3] = data.pallet_pos[i - pallet_index_adjust]
                data.y_params[i][3:7] = data.pallet_rot[i - pallet_index_adjust]

        data.filename = self.raw_file_names[idx]

        return data

    def _normalize(self, data: Data) -> Data:
        data = normalize_and_scale(
            data,
            extra_fields=[
                "gripper_pos",
                "loading_platform_pos",
                "pallet_pos",
                "object_points",
            ],
        )
        data["gripper_rot"] = quat_normalize(data["gripper_rot"])
        data["gripper_opening"] = (
            data["gripper_opening"] * 2.0 - 1.0
        )  # scale to [-1, 1]
        data["loading_platform_rot"] = quat_normalize(data["loading_platform_rot"])
        data["pallet_rot"] = quat_normalize(data["pallet_rot"])
        return data

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self) -> Union[str, list[str], tuple]:
        return self.file_names

    @property
    def processed_dir(self) -> str:
        return self.root

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple]:
        return [f"{file_name}.xyz" for file_name in self.file_names]

    def process(self) -> None:
        print("Should already be processed.")

    def _read_file_names(self, root: str) -> list[Path]:
        return sorted(
            [fp.stem for fp in Path(root).glob(f"*.xyz") if "pre_" not in fp.stem]
        )

    def len(self) -> int:
        return len(self.processed_file_names)

    def _random_quat(self, angle_bounds: tuple[float]) -> torch.Tensor:
        """
        Generate random quaternion q (1, 4) from angle bounds (in degrees)
        around each axis (x, y, z).
        """
        max_rot_angle_x, max_rot_angle_y, max_rot_angle_z = angle_bounds
        rot_angle_x = math.radians(random.uniform(-max_rot_angle_x, max_rot_angle_x))
        rot_angle_y = math.radians(random.uniform(-max_rot_angle_y, max_rot_angle_y))
        rot_angle_z = math.radians(random.uniform(-max_rot_angle_z, max_rot_angle_z))
        random_quat = euler_to_quaternion(
            rot_angle_x, rot_angle_y, rot_angle_z
        ).unsqueeze(0)
        random_quat = quat_normalize(random_quat)
        return random_quat

    def _align_rotation(self, pc: torch.Tensor) -> torch.Tensor:
        if (pc[:, 0] >= 0).all().item():
            # rotate on z 180
            rotation = torch.tensor(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32
            )
            pc = pc @ rotation.T
        return pc

    def _fps_sample(self, pc: torch.Tensor, num_points: int) -> Data:
        if num_points > 0 and pc.size(0) > num_points:
            idx = fpsample.bucket_fps_kdline_sampling(pc, num_points, h=6)
            pc = pc[idx]
        return pc

    def _distance_filter(self, pc: torch.Tensor, max_distance: float) -> torch.Tensor:
        if max_distance > 0:
            distance = torch.norm(pc, dim=1)
            pc = pc[distance < max_distance]
        return pc
