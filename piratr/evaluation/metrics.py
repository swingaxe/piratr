import torch
import numpy as np
from math import pi
from torchmetrics import Metric
from torchmetrics.functional import average_precision

from ..dataset import quat_normalize, quat_mul, rotate_local_180
from ..objects import CLOSED_RANGE, OPEN_RANGE


class ChamferMAP(Metric):
    def __init__(
        self,
        chamfer_thresh=0.00125,
        class_names={1: "mAP_gripper", 2: "mAP_loading_platform", 3: "mAP_pallet"},
        activate_param_metrics=False,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.chamfer_thresh = chamfer_thresh
        self.class_names = class_names
        self.activate_param_metrics = activate_param_metrics

        self.add_state("all_scores", default=[], dist_reduce_fx="cat")
        self.add_state("all_matches", default=[], dist_reduce_fx="cat")
        self.add_state("all_classes", default=[], dist_reduce_fx="cat")

        self.add_state("total_l2_gripper", default=[], dist_reduce_fx="cat")
        self.add_state("total_l2_loading_platform", default=[], dist_reduce_fx="cat")
        self.add_state("total_l2_pallet", default=[], dist_reduce_fx="cat")
        self.add_state("total_opening_gripper", default=[], dist_reduce_fx="cat")
        self.add_state(
            "total_opening_loading_platform",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state("total_opening_pallet", default=[], dist_reduce_fx="cat")
        self.add_state("total_geodesic_gripper", default=[], dist_reduce_fx="cat")
        self.add_state(
            "total_geodesic_loading_platform",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state("total_geodesic_pallet", default=[], dist_reduce_fx="cat")
        self.add_state("total_yaw_gripper", default=[], dist_reduce_fx="cat")
        self.add_state(
            "total_yaw_loading_platform",
            default=[],
            dist_reduce_fx="cat",
        )
        self.add_state("total_yaw_pallet", default=[], dist_reduce_fx="cat")
        self.add_state("gripper_filenames", default=[], dist_reduce_fx="cat")
        self.add_state("loading_platform_filenames", default=[], dist_reduce_fx="cat")
        self.add_state("pallet_filenames", default=[], dist_reduce_fx="cat")

    def pairwise_chamfer_distance_batch(self, pred, gt):
        """
        pred: [P, 64, 3]
        gt:   [G, 64, 3]
        returns: [P, G] chamfer distances
        """
        P, G = pred.size(0), gt.size(0)

        # Reshape for pairwise comparison
        pred_exp = pred.unsqueeze(1)  # [P, 1, 64, 3]
        gt_exp = gt.unsqueeze(0)  # [1, G, 64, 3]

        # Compute pairwise distances between points
        dists = torch.cdist(pred_exp, gt_exp, p=2)  # [P, G, 64, 64]
        dists = dists**2

        a2b = dists.min(dim=3).values.mean(dim=2)  # [P, G]
        b2a = dists.min(dim=2).values.mean(dim=2)  # [P, G]

        return a2b + b2a  # [P, G]

    def update(self, outputs, batch):
        gt_points_batch = batch.object_points.split_with_sizes(
            batch.num_objects.tolist()
        )
        gt_params_batch = batch.y_params.split_with_sizes(batch.num_objects.tolist())
        gt_classes_batch = batch.y_cls.split_with_sizes(batch.num_objects.tolist())
        for batch_idx, output_data in enumerate(outputs):
            pred_points = output_data.object_points
            pred_params = [
                None,
                output_data.gripper_params,
                output_data.loading_platform_params,
                output_data.pallet_params,
            ]
            pred_classes = output_data.object_class
            pred_scores = output_data.object_score
            gt_points = gt_points_batch[batch_idx]
            gt_params = gt_params_batch[batch_idx]
            y_classes = gt_classes_batch[batch_idx]
            scale = batch.scale[batch_idx]
            gt_filename = batch.filename[batch_idx]

            for cls in self.class_names.keys():
                pred_mask = pred_classes == cls
                gt_mask = y_classes == cls

                pred_cls_points = pred_points[pred_mask]
                pred_cls_scores = pred_scores[pred_mask]
                pred_cls_params = pred_params[cls][pred_mask]
                gt_cls_params = gt_params[gt_mask]
                gt_cls_points = gt_points[gt_mask]

                if len(pred_cls_points) == 0 and len(gt_cls_points) != 0:
                    scores = torch.zeros(len(gt_cls_points)).cpu()
                    self.all_scores.append(scores)
                    self.all_matches.append(torch.zeros_like(scores).cpu())
                    self.all_classes.append(torch.full_like(scores, cls).cpu())
                    continue
                elif len(gt_cls_points) == 0 and len(pred_cls_points) == 0:
                    self.all_scores.append(pred_cls_scores.cpu())
                    self.all_matches.append(torch.zeros_like(pred_cls_scores).cpu())
                    self.all_classes.append(torch.full_like(pred_cls_scores, cls).cpu())
                    continue
                elif len(pred_cls_points) != 0 and len(gt_cls_points) == 0:
                    self.all_scores.append(pred_cls_scores.cpu())
                    self.all_matches.append(torch.zeros_like(pred_cls_scores).cpu())
                    self.all_classes.append(torch.full_like(pred_cls_scores, cls).cpu())
                    continue

                chamfer = self.pairwise_chamfer_distance_batch(
                    pred_cls_points, gt_cls_points
                )
                used_gt = torch.zeros(gt_cls_points.size(0), dtype=torch.bool).cpu()
                matches = torch.zeros(pred_cls_points.size(0)).cpu()
                pred_params_matched = []
                gt_params_matched = []

                for i in range(pred_cls_points.size(0)):
                    dists = chamfer[i]
                    min_dist, min_idx = dists.min(0)
                    if min_dist < self.chamfer_thresh and not used_gt[min_idx]:
                        matches[i] = 1.0
                        pred_params_matched.append(pred_cls_params[i].cpu())
                        gt_params_matched.append(gt_cls_params[min_idx].cpu())
                        used_gt[min_idx] = True

                self.all_scores.append(pred_cls_scores.cpu())
                self.all_matches.append(matches.cpu())
                self.all_classes.append(torch.full_like(matches, cls).cpu())
                if sum(matches) > 0 and self.activate_param_metrics:
                    self._update_param_metrics(
                        cls,
                        torch.stack(pred_params_matched),
                        torch.stack(gt_params_matched),
                        scale.cpu(),
                        gt_filename,
                    )

    def _update_param_metrics(self, cls, pred_params, gt_params, scale, gt_filename):
        l2_error = torch.norm(pred_params[:, :3] - gt_params[:, :3], dim=1)
        consider_local_180 = True if cls != 2 else False
        local_axis = "y" if cls == 1 else "z"
        geo_error, yaw_error = self._geodesic_and_yaw_error(
            pred_params[:, 3:7],
            gt_params[:, 3:7],
            consider_local_180=consider_local_180,
            local_axis=local_axis,
        )
        if cls == 1:
            self.total_l2_gripper.append((l2_error / scale).cpu())
            self.total_geodesic_gripper.append(geo_error.cpu())
            self.total_yaw_gripper.append(yaw_error.cpu())
            gt_opening_angles = CLOSED_RANGE + (OPEN_RANGE - CLOSED_RANGE) * (
                gt_params[:, 7] / 2 + 0.5
            )
            pred_opening_angles = CLOSED_RANGE + (OPEN_RANGE - CLOSED_RANGE) * (
                pred_params[:, 7] / 2 + 0.5
            )
            opening_error = (gt_opening_angles - pred_opening_angles).abs()
            self.total_opening_gripper.append(opening_error.cpu())
            self.gripper_filenames.extend([gt_filename] * len(opening_error))
        elif cls == 2:
            self.total_l2_loading_platform.append((l2_error / scale).cpu())
            self.total_geodesic_loading_platform.append(geo_error.cpu())
            self.total_yaw_loading_platform.append(yaw_error.cpu())
            self.loading_platform_filenames.extend([gt_filename] * len(l2_error))
        elif cls == 3:
            self.total_l2_pallet.append((l2_error / scale).cpu())
            self.total_geodesic_pallet.append(geo_error.cpu())
            self.total_yaw_pallet.append(yaw_error.cpu())
            self.pallet_filenames.extend([gt_filename] * len(l2_error))

    @staticmethod
    def _geodesic_and_yaw_error(
        q_pred: torch.Tensor,
        q_gt: torch.Tensor,
        w_last: bool = False,  # True if inputs are [x,y,z,w]; else [w,x,y,z]
        degrees: bool = True,  # return degrees
        abs_yaw: bool = True,  # take |yaw|
        consider_local_180: bool = False,
        local_axis: str = "z",  # 'x' | 'y' | 'z' (used if consider_local_180=True)
    ):
        """
        Errors between q_gt and q_pred:
        - geodesic angle (sequence-independent)
        - yaw (rotation about Z) from yaw-first ZYX (yaw–pitch–roll) decomposition.

        If consider_local_180=True, also evaluate errors for q_pred rotated by +180°
        around its *local* `local_axis` (using rotate_local_180), and take the
        elementwise minimum (|yaw| used if abs_yaw=True).

        q_pred, q_gt: [..., 4] quaternions
        Returns: geodesic_err, yaw_err with shape [...]
        """
        assert q_pred.shape[-1] == 4 and q_gt.shape[-1] == 4

        def to_wxyz(q):
            return (
                torch.stack((q[..., 3], q[..., 0], q[..., 1], q[..., 2]), dim=-1)
                if w_last
                else q
            )

        def q_conj(q):  # wxyz
            w, x, y, z = q.unbind(-1)
            return torch.stack((w, -x, -y, -z), dim=-1)

        def geodesic_from_qerr(q_err_wxyz):
            w = q_err_wxyz[..., 0].abs()
            v = q_err_wxyz[..., 1:]
            return 2.0 * torch.atan2(v.norm(dim=-1), w.clamp_min(1e-12))

        def yaw_from_qerr_zyx(q_err_wxyz):
            # yaw-first ZYX (yaw–pitch–roll); yaw ψ about Z:
            w, x, y, z = q_err_wxyz.unbind(-1)
            num = 2.0 * (w * z + x * y)
            den = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(num, den)

        def pitch_from_qerr_yxz(q_err_wxyz: torch.Tensor) -> torch.Tensor:
            # Pitch-first YXZ (pitch–roll–yaw); pitch θ about Y.
            w, x, y, z = q_err_wxyz.unbind(-1)
            num = 2.0 * (x * z + w * y)
            den = 1.0 - 2.0 * (x * x + y * y)
            return torch.atan2(num, den)

        # normalize to wxyz
        qp = quat_normalize(to_wxyz(q_pred))
        qg = quat_normalize(to_wxyz(q_gt))

        def errors_for(qp_variant, local_axis):
            # relative rotation: gt -> pred
            q_err = quat_mul(qp_variant, q_conj(qg))
            geo = geodesic_from_qerr(q_err)  # ∈ [0, π]
            if local_axis == "z":
                yaw = yaw_from_qerr_zyx(q_err)  # ∈ (−π, π]
            elif local_axis == "y":
                yaw = pitch_from_qerr_yxz(q_err)  # ∈ (−π, π]
            else:
                raise ValueError(f"local_axis={local_axis} not supported")
            return geo, yaw

        geo1, yaw1 = errors_for(qp, local_axis)

        if consider_local_180:
            qp2 = rotate_local_180(qp, local_axis)  # expects wxyz; local post-rotation
            geo2, yaw2 = errors_for(qp2, local_axis)

            geo = torch.minimum(geo1, geo2)

            # choose smaller |yaw|; keep sign unless abs_yaw=True
            pick_second = yaw2.abs() < yaw1.abs()
            yaw = torch.where(pick_second, yaw2, yaw1)
        else:
            geo, yaw = geo1, yaw1

        if degrees:
            geo = torch.rad2deg(geo)
            yaw = torch.rad2deg(yaw)

        if abs_yaw:
            yaw = yaw.abs()

        return geo, yaw

    def compute(self):
        if not self.all_scores:
            return {cls_name: 0.0 for cls_name in self.class_names.values()}

        scores = torch.cat(self.all_scores)
        matches = torch.cat(self.all_matches)
        classes = torch.cat(self.all_classes)

        result = {}
        ap_values = []

        for cls in self.class_names.keys():
            mask = classes == cls
            if mask.sum() == 0 or torch.sum(matches[mask]) == 0:
                ap = torch.tensor(0.0).cpu()
            else:
                ap = average_precision(
                    scores[mask], matches[mask].to(torch.int32), task="binary"
                ).cpu()
            result[self.class_names[cls]] = ap.item()
            ap_values.append(ap)

        result["mAP"] = torch.stack(ap_values).mean().item()
        if not self.activate_param_metrics:
            return result
        result["l2_gripper[m]"] = (
            torch.cat(self.total_l2_gripper).mean() if self.total_l2_gripper else 0.0
        )
        result["l2_gripper[std]"] = (
            torch.cat(self.total_l2_gripper).std() if self.total_l2_gripper else 0.0
        )
        result["l2_loading_platform[m]"] = (
            torch.cat(self.total_l2_loading_platform).mean()
            if self.total_l2_loading_platform
            else 0.0
        )
        result["l2_loading_platform[std]"] = (
            torch.cat(self.total_l2_loading_platform).std()
            if self.total_l2_loading_platform
            else 0.0
        )
        result["l2_pallet[m]"] = (
            torch.cat(self.total_l2_pallet).mean() if self.total_l2_pallet else 0.0
        )
        result["l2_pallet[std]"] = (
            torch.cat(self.total_l2_pallet).std() if self.total_l2_pallet else 0.0
        )
        result["geodesic_gripper[deg]"] = (
            torch.cat(self.total_geodesic_gripper).mean()
            if self.total_geodesic_gripper
            else 0.0
        )
        result["geodesic_gripper[std]"] = (
            torch.cat(self.total_geodesic_gripper).std()
            if self.total_geodesic_gripper
            else 0.0
        )
        result["geodesic_loading_platform[deg]"] = (
            torch.cat(self.total_geodesic_loading_platform).mean()
            if self.total_geodesic_loading_platform
            else 0.0
        )
        result["geodesic_loading_platform[std]"] = (
            torch.cat(self.total_geodesic_loading_platform).std()
            if self.total_geodesic_loading_platform
            else 0.0
        )
        result["geodesic_pallet[deg]"] = (
            torch.cat(self.total_geodesic_pallet).mean()
            if self.total_geodesic_pallet
            else 0.0
        )
        result["geodesic_pallet[std]"] = (
            torch.cat(self.total_geodesic_pallet).std()
            if self.total_geodesic_pallet
            else 0.0
        )
        result["yaw_gripper[deg]"] = (
            torch.cat(self.total_yaw_gripper).mean() if self.total_yaw_gripper else 0.0
        )
        result["yaw_gripper[std]"] = (
            torch.cat(self.total_yaw_gripper).std() if self.total_yaw_gripper else 0.0
        )
        result["yaw_loading_platform[deg]"] = (
            torch.cat(self.total_yaw_loading_platform).mean()
            if self.total_yaw_loading_platform
            else 0.0
        )
        result["yaw_loading_platform[std]"] = (
            torch.cat(self.total_yaw_loading_platform).std()
            if self.total_yaw_loading_platform
            else 0.0
        )
        result["yaw_pallet[deg]"] = (
            torch.cat(self.total_yaw_pallet).mean() if self.total_yaw_pallet else 0.0
        )
        result["yaw_pallet[std]"] = (
            torch.cat(self.total_yaw_pallet).std() if self.total_yaw_pallet else 0.0
        )
        result["opening_gripper[deg]"] = (
            torch.cat(self.total_opening_gripper).mean()
            if self.total_opening_gripper
            else 0.0
        )
        result["opening_gripper[std]"] = (
            torch.cat(self.total_opening_gripper).std()
            if self.total_opening_gripper
            else 0.0
        )
        return result

    def get_all_state_values_dict(self):
        """
        Returns a dictionary of all state's values in the metric.
        """

        def to_list(lst):
            if not lst:
                return []
            if isinstance(lst[0], torch.Tensor):
                return torch.cat(lst).cpu().tolist()
            return list(lst)

        return {
            "scores": to_list(self.all_scores),
            "matches": to_list(self.all_matches),
            "classes": to_list(self.all_classes),
            "l2_gripper": to_list(self.total_l2_gripper),
            "l2_loading_platform": to_list(self.total_l2_loading_platform),
            "l2_pallet": to_list(self.total_l2_pallet),
            "opening_gripper": to_list(self.total_opening_gripper),
            "opening_loading_platform": to_list(self.total_opening_loading_platform),
            "opening_pallet": to_list(self.total_opening_pallet),
            "geodesic_gripper": to_list(self.total_geodesic_gripper),
            "geodesic_loading_platform": to_list(self.total_geodesic_loading_platform),
            "geodesic_pallet": to_list(self.total_geodesic_pallet),
            "yaw_gripper": to_list(self.total_yaw_gripper),
            "yaw_loading_platform": to_list(self.total_yaw_loading_platform),
            "yaw_pallet": to_list(self.total_yaw_pallet),
            "gripper_filenames": to_list(self.gripper_filenames),
            "loading_platform_filenames": to_list(self.loading_platform_filenames),
            "pallet_filenames": to_list(self.pallet_filenames),
        }
