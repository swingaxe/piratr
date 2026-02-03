import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch_geometric.data.data import Data
from piratr.dataset import rotate_local_180


class ParametricMatcher(nn.Module):
    def __init__(self, cost_class: int = 1, cost_params: int = 1) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_params = cost_params

    @staticmethod
    def _param_cost_with_symmetries(
        preds: torch.Tensor,
        targets: torch.Tensor,
        quat_slice: slice,
        pos_slice: slice = slice(0, 3),
        extra_slice: slice | None = None,
        rot_axis: str | None = None,
        target_slice: slice | None = None,
        p: int = 1,
    ) -> torch.Tensor:
        """
        Compute pairwise parameter costs with quaternion symmetries.

        preds      : [Nq, Dp]
        targets    : [No, Dt]  (Dt may be Dp or a prefix, e.g. first 7 dims)
        quat_slice : slice selecting quaternion in preds (e.g., slice(3,7))
        pos_slice  : slice selecting position in preds (default first 3)
        extra_slice: optional slice for extra params (e.g., width for gripper)
        rot_axis   : None | "x" | "y" | "z" (for 180° local rotation)
        target_slice: None or slice applied to targets (e.g., slice(None, 7))
        p          : Lp norm for cdist (1 == L1)

        Returns: [Nq, No] pairwise min cost across symmetry variants.
        """
        if target_slice is not None:
            targets = targets[:, target_slice]  # e.g., first 7 dims

        q = preds[:, quat_slice]
        variants = [q, -q]
        if rot_axis is not None:
            q_rot = rotate_local_180(q, rot_axis)
            variants += [q_rot, -q_rot]

        assembled = []
        for qv in variants:
            parts = [preds[:, pos_slice], qv]
            if extra_slice is not None:
                parts.append(preds[:, extra_slice])
            assembled.append(torch.cat(parts, dim=-1))  # [Nq, Dt]

        dists = torch.stack([torch.cdist(v, targets, p=p) for v in assembled], dim=0)
        return dists.min(dim=0).values  # [Nq, No]

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], data: Data
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the matching indices based on class costs and parameter costs.
        """
        bs, num_queries = outputs["pred_class"].shape[:2]

        # Classification cost: for each (query, object), take -P(class_of_object | query)
        out_prob = outputs["pred_class"].flatten(0, 1).softmax(-1)  # [B*Q, C]
        # Advanced indexing: column k picks the class of object k (works like your version)
        cost_class = -out_prob[:, data.y_cls]  # [B*Q, B*O]

        # Flatten predictions per-head for param costs
        pred_gripper = outputs["pred_gripper_params"].flatten(0, 1)  # [B*Q, Dg]
        pred_load = outputs["pred_loading_platform_params"].flatten(0, 1)  # [B*Q, 7]
        pred_pallet = outputs["pred_pallet_params"].flatten(0, 1)  # [B*Q, 7]

        # Targets (params shared; we'll slice per class)
        tgt_params_all = data.y_params  # [B*O, D*]

        # Per-class symmetry/config
        quat_slice = slice(3, 7)  # common across heads

        cost_gripper = self._param_cost_with_symmetries(
            pred_gripper,
            tgt_params_all,
            quat_slice=quat_slice,
            extra_slice=slice(7, None),  # width/etc stays as-is
            rot_axis="y",
            target_slice=slice(None, None),  # full vector
            p=1,
        )

        cost_loading = self._param_cost_with_symmetries(
            pred_load,
            tgt_params_all,
            quat_slice=quat_slice,
            extra_slice=None,
            rot_axis=None,  # only sign symmetry (q, -q)
            target_slice=slice(None, 7),  # compare first 7 (pos+quat)
            p=1,
        )

        cost_pallet = self._param_cost_with_symmetries(
            pred_pallet,
            tgt_params_all,
            quat_slice=quat_slice,
            extra_slice=None,
            rot_axis="z",
            target_slice=slice(None, 7),  # compare first 7 (pos+quat)
            p=1,
        )

        # Select the appropriate param cost per object class (class 0 -> zeros)
        zeros = torch.zeros_like(cost_gripper)
        cost_params = torch.stack(
            [zeros, cost_gripper, cost_loading, cost_pallet], dim=-1
        )  # [B*Q, B*O, 4]
        cost_params = cost_params[
            torch.arange(cost_params.size(0))[:, None],
            torch.arange(cost_params.size(1)),
            data.y_cls,
        ]  # [B*Q, B*O]

        # Combine costs and split per batch for Hungarian
        C = self.cost_class * cost_class + self.cost_params * cost_params
        C = C.view(bs, num_queries, -1).cpu()

        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(data.num_objects.cpu().tolist(), dim=-1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
