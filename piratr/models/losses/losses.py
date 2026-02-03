from abc import abstractmethod
import torch
from torch import nn
from dataclasses import dataclass, field
import torch.nn.functional as F
from torch_geometric.data.data import Data
from kornia.losses import focal_loss

from .matcher import ParametricMatcher
from ...objects import (
    sample_gripper_points,
    transform_gripper_points,
    transform_loading_or_pallet_points,
)
from ...dataset import rotate_local_180


def chamfer_distance_batch(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    assert len(pts1.shape) == 3 and len(pts2.shape) == 3
    if pts1.nelement() == 0 or pts2.nelement() == 0:
        return torch.tensor(0.0, device=pts1.device, requires_grad=True)
    dist_matrix = torch.cdist(
        pts1, pts2, p=2
    )  # shape: (batch_size, num_points, num_points)
    dist1 = dist_matrix.min(dim=2).values.mean(dim=1)  # min over pts2, mean over pts1
    dist2 = dist_matrix.min(dim=1).values.mean(dim=1)  # min over pts1, mean over pts2
    return (dist1 + dist2) / 2


def best_l1_with_symmetries(
    src_params: torch.Tensor,
    tgt_params: torch.Tensor,
    quat_slice: slice,
    pos_slice: slice = slice(0, 3),
    extra_slice: slice | None = None,
    rot_axis: str | None = None,
) -> torch.Tensor:
    """
    Compute L1 over param vectors under quaternion symmetries
    (q, -q, optional 180° local rotation, and their sign flips), then take min.

    Shapes:
      src_params, tgt_params: [N, D]
    Returns:
      per-sample loss: [N]
    """
    q_src = src_params[:, quat_slice]  # [N, 4]
    variants = [q_src, -q_src]
    if rot_axis is not None:
        q_rot = rotate_local_180(q_src, rot_axis)
        variants += [q_rot, -q_rot]

    assembled = []
    for qv in variants:
        parts = [src_params[:, pos_slice], qv]
        if extra_slice is not None:
            parts.append(src_params[:, extra_slice])
        assembled.append(torch.cat(parts, dim=-1))  # [N, D or D'] matching target

    stacked = torch.stack(
        [F.l1_loss(v, tgt_params, reduction="none").mean(-1) for v in assembled], dim=0
    )  # [num_variants, N]
    return stacked.min(dim=0).values  # [N]


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def f1_score(output, target, threshold=0.5):
    output = output > threshold
    target = target > threshold
    tp = (output & target).sum()
    tn = (~output & ~target).sum()
    fp = (output & ~target).sum()
    fn = (~output & target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * (precision * recall) / (precision + recall + 1e-8), precision, recall


@dataclass
class LossParams:
    num_classes: int
    cost_class: int = 1
    cost_params: int = 1
    class_loss_type: str = "cross_entropy"  # or "focal"
    class_loss_weights: list[float] = field(
        default_factory=lambda: [
            0.04834912,
            0.40329467,
            0.09588135,
            0.23071379,
            0.22176106,
        ]
    )


class Loss(nn.Module):

    def __init__(self, params: LossParams) -> None:
        super().__init__()
        self.matcher = ParametricMatcher(params.cost_class, params.cost_params)
        self.num_classes = params.num_classes
        self.class_loss_type = params.class_loss_type
        class_weights = torch.tensor(
            params.class_loss_weights,
        )

        self.register_buffer("class_weights", class_weights)

    def forward(
        self, outputs: dict[str, torch.Tensor], data: Data
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, data)
        losses = {}
        losses.update(self._loss_class(outputs, data, indices))
        losses.update(self._loss_objects(outputs, data, indices))
        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, data)
                l_dict = self._loss_class(aux_outputs, data, indices, False)
                l_dict.update(self._loss_objects(aux_outputs, data, indices))
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    @abstractmethod
    def _loss_objects(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute the object loss."""
        pass

    def _loss_class(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        log: bool = True,
    ) -> torch.Tensor:
        num_targets = (
            data.num_polylines.tolist()
            if hasattr(data, "num_polylines")
            else data.num_objects.tolist()
        )
        src_logits = outputs["pred_class"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [
                target[J]
                for target, (_, J) in zip(
                    data.y_cls.split_with_sizes(num_targets), indices
                )
            ]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )  # 0: empty class
        target_classes[idx] = target_classes_o
        losses = {}

        if self.class_loss_type == "cross_entropy":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                weight=self.class_weights.to(src_logits.device),
                reduction="mean",
            )
        else:
            loss_class = focal_loss(
                src_logits.transpose(1, 2),
                target_classes,
                alpha=0.25,
                gamma=2.0,
                weight=self.class_weights.to(src_logits.device),
                reduction="mean",
            )
        losses["loss_class"] = loss_class
        if log:
            losses["class_error"] = (
                100
                - accuracy(
                    src_logits.reshape(-1, src_logits.size(-1)),
                    target_classes.flatten(),
                )[0]
            )
            f1, _, _ = f1_score(
                src_logits.reshape(-1, src_logits.size(-1)).softmax(-1).argmax(-1),
                target_classes.flatten(),
                threshold=0.5,
            )
            losses["class_f1_score"] = f1

        return losses

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class ParametricLoss(Loss):
    def __init__(
        self,
        params: LossParams,
        gripper_main_points: torch.Tensor,
        gripper_s1_points: torch.Tensor,
        gripper_s2_points: torch.Tensor,
        loading_platform_points: torch.Tensor,
        pallet_points: torch.Tensor,
    ) -> None:
        super().__init__(params)
        self.gripper_main_points = gripper_main_points
        self.gripper_s1_points = gripper_s1_points
        self.gripper_s2_points = gripper_s2_points
        self.loading_platform_points = loading_platform_points
        self.pallet_points = pallet_points

    def _loss_objects(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        batch_idx = idx[0]

        # Slice predicted params for matched predictions
        src_gripper_params = outputs["pred_gripper_params"][idx]  # [..., Dg]
        src_loading_platform_params = outputs["pred_loading_platform_params"][
            idx
        ]  # [..., 7]
        src_pallet_params = outputs["pred_pallet_params"][idx]  # [..., 7]

        # Precompute transformed point sets (for Chamfer)
        src_gripper_points = transform_gripper_points(
            src_gripper_params[:, :3],
            src_gripper_params[:, 3:7],
            (src_gripper_params[:, 7] + 1) / 2,
            self.gripper_main_points,
            self.gripper_s1_points,
            self.gripper_s2_points,
            data.scale[batch_idx],
        )
        src_loading_platform_points = transform_loading_or_pallet_points(
            src_loading_platform_params[:, :3],
            src_loading_platform_params[:, 3:7],
            self.loading_platform_points,
            data.scale[batch_idx],
        )
        src_pallet_points = transform_loading_or_pallet_points(
            src_pallet_params[:, :3],
            src_pallet_params[:, 3:7],
            self.pallet_points,
            data.scale[batch_idx],
        )

        # Gather matched targets
        target_params = torch.cat(
            [
                t[J]
                for t, (_, J) in zip(
                    data.y_params.split_with_sizes(data.num_objects.tolist()), indices
                )
            ]
        )
        target_classes = torch.cat(
            [
                t[J]
                for t, (_, J) in zip(
                    data.y_cls.split_with_sizes(data.num_objects.tolist()), indices
                )
            ]
        )
        target_point_sets = torch.cat(
            [
                t[J]
                for t, (_, J) in zip(
                    data.object_points.split_with_sizes(data.num_objects.tolist()),
                    indices,
                )
            ]
        )

        # Masks
        gripper_mask = target_classes == 1
        loading_platform_mask = target_classes == 2
        pallet_mask = target_classes == 3

        losses = {}

        # Class config table: how to compare params + which point tensors to use
        class_configs = [
            dict(
                name="gripper",
                class_mask=gripper_mask,
                src_params=src_gripper_params,
                tgt_params=lambda tgt: tgt,  # full vector
                quat_slice=slice(3, 7),
                extra_slice=slice(7, None),  # keep width param as-is
                rot_axis="y",  # 180° about local y
                src_points=src_gripper_points,
            ),
            dict(
                name="loading_platform",
                class_mask=loading_platform_mask,
                src_params=src_loading_platform_params,
                tgt_params=lambda tgt: tgt[:, :7],  # compare first 7 (pos+quat)
                quat_slice=slice(3, 7),
                extra_slice=None,
                rot_axis=None,  # only sign symmetry (q, -q)
                src_points=src_loading_platform_points,
            ),
            dict(
                name="pallet",
                class_mask=pallet_mask,
                src_params=src_pallet_params,
                tgt_params=lambda tgt: tgt[:, :7],  # compare first 7 (pos+quat)
                quat_slice=slice(3, 7),
                extra_slice=None,
                rot_axis="z",  # 180° about local z
                src_points=src_pallet_points,
            ),
        ]

        total = 0.0
        for cfg in class_configs:
            class_mask = cfg["class_mask"]
            if class_mask.any():
                sp = cfg["src_params"][class_mask]
                tp = cfg["tgt_params"](target_params[class_mask])

                param_losses = best_l1_with_symmetries(
                    sp,
                    tp,
                    quat_slice=cfg["quat_slice"],
                    extra_slice=cfg["extra_slice"],
                    rot_axis=cfg["rot_axis"],
                )
                losses[f"loss_{cfg['name']}_params"] = param_losses.mean()

                chamfer = chamfer_distance_batch(
                    cfg["src_points"][class_mask], target_point_sets[class_mask]
                )
                losses[f"loss_{cfg['name']}_chamfer"] = chamfer.mean()
            else:
                device = cfg["src_params"].device
                losses[f"loss_{cfg['name']}_params"] = torch.tensor(0.0, device=device)
                losses[f"loss_{cfg['name']}_chamfer"] = torch.tensor(0.0, device=device)

            total = (
                total
                + losses[f"loss_{cfg['name']}_params"]
                + losses[f"loss_{cfg['name']}_chamfer"]
            )

        losses["total_object"] = total
        return losses
