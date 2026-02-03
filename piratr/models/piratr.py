import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch_geometric.data.data import Data
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch

import json

from piratr.objects import (
    OBJECT_INFO,
    transform_gripper_points,
    transform_loading_or_pallet_points,
)

from .model_config import ModelConfig
from .losses import LossParams, ParametricLoss
from .transformer import Transformer
from .positional_embedding import PositionEmbeddingCoordsSine
from .query_engine import build_query_engine
from piratr.dataset import reverse_normalize_and_scale, quat_normalize

from piratr.evaluation.metrics import ChamferMAP


class PIRATR(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.register_buffer("gripper_main_points", OBJECT_INFO.gripper_main_points)
        self.register_buffer("gripper_s1_points", OBJECT_INFO.gripper_s1_points)
        self.register_buffer("gripper_s2_points", OBJECT_INFO.gripper_s2_points)
        self.register_buffer(
            "loading_platform_points", OBJECT_INFO.loading_platform_points
        )
        self.register_buffer("pallet_points", OBJECT_INFO.pallet_points)

        self.config = config
        self.pc_preencoder = config.get_preencoder()
        self.enc_dim = config.encoder_dim
        self.dec_dim = config.decoder_dim
        self.num_preds = config.num_preds
        self.num_curve_points = config.num_curve_points
        self.num_curve_points_val = config.num_curve_points_val
        self.num_classes = config.num_classes
        self.max_points_in_param = config.max_points_in_param
        self.preenc_to_enc_proj = MLP(
            [self.pc_preencoder.out_channels, self.enc_dim, self.enc_dim],
            act="relu",
            norm="layer_norm",
        )
        num_decoder_layers = config.num_decoder_layers
        self.transformer = Transformer(
            self.enc_dim,
            self.dec_dim,
            nhead=config.num_attn_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            enc_dim_feedforward=config.enc_dim_feedforward,
            dec_dim_feedforward=config.dec_dim_feedforward,
            enc_dropout=config.encoder_dropout,
            dec_dropout=config.decoder_dropout,
            return_intermediate_dec=True,
        )
        self.positional_embedding = PositionEmbeddingCoordsSine(
            d_pos=self.dec_dim, pos_type=self.config.pos_embed_type
        )
        self.pos_embed_proj = MLP(
            [self.dec_dim, self.dec_dim, self.dec_dim], act="relu", norm="layer_norm"
        )
        self.query_engine = build_query_engine(
            config.query_type,
            self.positional_embedding,
            self.dec_dim,
            self.max_points_in_param,
            self.num_preds,
        )

        def make_mlp(out_dim, layers=4, base_dim=None, bias_last=True):
            base_dim = base_dim or self.dec_dim
            n_layers = layers - 1
            return MLP(
                channel_list=[base_dim] * n_layers + [out_dim],
                bias=[False] * (n_layers - 1) + [bias_last],
                dropout=self.config.mlp_dropout,
                act="relu",
                norm="layer_norm",
            )

        self.class_head = make_mlp(self.num_classes)
        self.gripper_param_head = make_mlp(
            3 + 4 + 1
        )  # position + orientation + opening
        self.loading_platform_param_head = make_mlp(3 + 4)  # position + orientation
        self.pallet_param_head = make_mlp(3 + 4)  # position + orientation

        self.loss = ParametricLoss(
            LossParams(
                num_classes=self.num_classes - 1,  # -1 for the EOS token
                cost_class=config.cost_weights["cost_class"],
                cost_params=config.cost_weights["cost_params"],
                class_loss_type=config.class_loss_type,
                class_loss_weights=config.class_loss_weights,
            ),
            self.gripper_main_points,
            self.gripper_s1_points,
            self.gripper_s2_points,
            self.loading_platform_points,
            self.pallet_points,
        )
        self.auxiliary_loss = self.config.auxiliary_loss
        self.weight_dict = {
            "loss_class": config.loss_weights["loss_class"],
            "loss_gripper_params": config.loss_weights["loss_gripper_params"],
            "loss_gripper_chamfer": config.loss_weights["loss_gripper_chamfer"],
            "loss_loading_platform_params": config.loss_weights[
                "loss_loading_platform_params"
            ],
            "loss_loading_platform_chamfer": config.loss_weights[
                "loss_loading_platform_chamfer"
            ],
            "loss_pallet_params": config.loss_weights["loss_pallet_params"],
            "loss_pallet_chamfer": config.loss_weights["loss_pallet_chamfer"],
        }
        # TODO this is a hack
        self.aux_weight_dict = {}
        if self.auxiliary_loss:
            for i in range(num_decoder_layers - 1):
                self.aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(self.aux_weight_dict)

        self.chamfer_map = ChamferMAP(chamfer_thresh=config.mAP_chamfer_threshold)

    def forward(self, data: Data) -> dict[str, Tensor]:
        x, pos, batch = self.pc_preencoder(data)[-1]
        x = self.preenc_to_enc_proj(x)
        x, mask = to_dense_batch(x, batch)
        pos_dense_batch, _ = to_dense_batch(pos, batch)
        pos_embed = self.pos_embed_proj(
            self.positional_embedding(
                pos_dense_batch, num_channels=self.dec_dim
            ).permute(0, 2, 1)
        ).permute(0, 2, 1)
        query_xyz, query_embed = self.query_engine(Data(pos=pos, batch=batch))
        x = self.transformer(
            x,  # [batch_size, num_points, enc_dim]
            # transformer expects 1s to be masked
            ~mask if not torch.all(mask) else None,  # [batch_size, num_points]
            query_embed,  # [batch_size, dec_dim, num_queries]
            pos_embed,  # [batch_size, dec_dim, num_points]
        )
        output_class = self.class_head(x)
        output_gripper_params = self.gripper_param_head(x)
        pred_gripper_params = output_gripper_params[-1].reshape(
            data.batch_size, self.num_preds, 8
        )
        pred_gripper_params[:, :, :3] = (
            pred_gripper_params[:, :, :3] + query_xyz[:, :, 0, :]
        )
        pred_gripper_params[:, :, 3:7] = quat_normalize(
            pred_gripper_params[:, :, 3:7].clone()
        )
        output_loading_platform_params = self.loading_platform_param_head(x)
        pred_loading_platform_params = output_loading_platform_params[-1].reshape(
            data.batch_size, self.num_preds, 7
        )
        pred_loading_platform_params[:, :, :3] = (
            pred_loading_platform_params[:, :, :3] + query_xyz[:, :, 0, :]
        )
        pred_loading_platform_params[:, :, 3:7] = quat_normalize(
            pred_loading_platform_params[:, :, 3:7].clone()
        )
        output_pallet_params = self.pallet_param_head(x)
        pred_pallet_params = output_pallet_params[-1].reshape(
            data.batch_size, self.num_preds, 7
        )
        pred_pallet_params[:, :, :3] = (
            pred_pallet_params[:, :, :3] + query_xyz[:, :, 0, :]
        )
        pred_pallet_params[:, :, 3:7] = quat_normalize(
            pred_pallet_params[:, :, 3:7].clone()
        )

        out = {
            "pred_class": output_class[-1],
            "pred_gripper_params": pred_gripper_params,
            "pred_loading_platform_params": pred_loading_platform_params,
            "pred_pallet_params": pred_pallet_params,
            "query_xyz": query_xyz,
        }
        if self.auxiliary_loss and self.training:
            out["aux_outputs"] = self._set_aux_loss(
                output_gripper_params,
                output_loading_platform_params,
                output_pallet_params,
                query_xyz,
                output_class,
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(
        self,
        output_gripper_params: torch.Tensor,
        output_loading_platform_params: torch.Tensor,
        output_pallet_params: torch.Tensor,
        query_xyz: torch.Tensor,
        output_class: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        out_aux = []
        for g, lp, p, cl in zip(
            output_gripper_params[:-1],
            output_loading_platform_params[:-1],
            output_pallet_params[:-1],
            output_class[:-1],
        ):
            pred_gripper_params = g.reshape(*g.shape[:2], 8)
            pred_gripper_params_adjusted = pred_gripper_params.clone()
            pred_gripper_params_adjusted[:, :, :3] = (
                pred_gripper_params[:, :, :3] + query_xyz[:, :, 0, :]
            )
            pred_gripper_params_adjusted[:, :, 3:7] = quat_normalize(
                pred_gripper_params_adjusted[:, :, 3:7].clone()
            )
            pred_loading_platform_params = lp.reshape(*lp.shape[:2], 7)
            pred_loading_platform_params_adjusted = pred_loading_platform_params.clone()
            pred_loading_platform_params_adjusted[:, :, :3] = (
                pred_loading_platform_params[:, :, :3] + query_xyz[:, :, 0, :]
            )
            pred_loading_platform_params_adjusted[:, :, 3:7] = quat_normalize(
                pred_loading_platform_params_adjusted[:, :, 3:7].clone()
            )
            pred_pallet_params = p.reshape(*p.shape[:2], 7)
            pred_pallet_params_adjusted = pred_pallet_params.clone()
            pred_pallet_params_adjusted[:, :, :3] = (
                pred_pallet_params[:, :, :3] + query_xyz[:, :, 0, :]
            )
            pred_pallet_params_adjusted[:, :, 3:7] = quat_normalize(
                pred_pallet_params_adjusted[:, :, 3:7].clone()
            )

            layer_out = {
                "pred_gripper_params": pred_gripper_params_adjusted,
                "pred_loading_platform_params": pred_loading_platform_params_adjusted,
                "pred_pallet_params": pred_pallet_params_adjusted,
                "pred_class": cl,
            }

            out_aux.append(layer_out)

        return out_aux

    def predict_step(
        self,
        batch: Data,
        score_thresholds: list[float] = None,
        reverse_norm: bool = True,
    ) -> list[Data]:
        preds = self(batch)
        outputs = self.decode_predictions(batch, preds, reverse_norm)
        for output in outputs:
            output.active = output.object_class != 0

        if score_thresholds:
            outputs = [
                self._filter_predictions(output, score_thresholds) for output in outputs
            ]

        return outputs

    @staticmethod
    def _filter_predictions(output, score_thresholds: list[float]):
        """
        Filter predictions based on per-class score thresholds.

        Behavior (unchanged):
        - Class 1 (gripper): keep exactly one — the single highest-score instance,
            only if its score >= threshold for class 1.
        - Class 2 (loading platform): same as class 1.
        - Class 3 (pallet): keep all with score >= threshold for class 3.
        - All other classes are ignored.
        - The final boolean mask is written to `output.active` and the same `output`
            object is returned.

        Args:
            output: An object with tensor fields:
                - output.object_class: (N,) int/long tensor of class ids.
                - output.object_score: (N,) float tensor of scores.
                - output.active      : (N,) bool tensor will be set by this function.
            score_thresholds: Iterable of three floats [thr_class1, thr_class2, thr_class3].

        Returns:
            The same `output` object with `output.active` set.
        """
        # Validate thresholds and align types/devices.
        if len(score_thresholds) != 3:
            raise ValueError(
                "score_thresholds must be a sequence of three floats: "
                "[gripper_thr, loading_platform_thr, pallet_thr]"
            )
        classes = output.object_class
        scores = output.object_score
        if classes.numel() == 0:
            # Nothing to do; ensure `active` exists and is empty.
            output.active = torch.zeros_like(classes, dtype=torch.bool)
            return output
        device = scores.device
        thr_gripper, thr_loading_platform, thr_pallet = (
            float(score_thresholds[0]),
            float(score_thresholds[1]),
            float(score_thresholds[2]),
        )

        # Helper: keep exactly the highest-scoring instance of a class if it passes the threshold.
        def _keep_best_one_for_class(class_id: int, threshold: float) -> torch.Tensor:
            cls_mask = classes == class_id
            if not torch.any(cls_mask):
                return torch.zeros_like(cls_mask, dtype=torch.bool)
            # Replace non-class scores with -inf so argmax picks the top within the class.
            masked_scores = torch.where(
                cls_mask, scores, torch.tensor(float("-inf"), device=device)
            )
            top_idx = torch.argmax(masked_scores)
            keep = torch.zeros_like(cls_mask, dtype=torch.bool)
            if scores[top_idx] >= threshold:
                keep[top_idx] = True
            return keep

        gripper_mask = _keep_best_one_for_class(1, thr_gripper)
        loading_platform_mask = _keep_best_one_for_class(2, thr_loading_platform)
        # Pallets: keep all above threshold.
        pallet_mask = (classes == 3) & (scores >= thr_pallet)

        final_mask = gripper_mask | loading_platform_mask | pallet_mask
        output.active = final_mask
        return output

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        outputs = self(batch)
        loss_dict = self.loss(outputs, batch)
        for k, v in loss_dict.items():
            weight = self.weight_dict[k] if "loss" in k else 1
            self._default_log(k, v * weight)
        # weigh losses and sum them for backpropagation
        weighted_loss_dict = {
            k: loss_dict[k] * self.weight_dict[k] for k in self.weight_dict.keys()
        }
        total_loss = sum(weighted_loss_dict.values())
        self._default_log("loss_train", total_loss)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Data, batch_idx: int) -> None:
        outputs = self(batch)
        # sample the training curve points for loss computation
        loss_dict = self.loss(outputs, batch)
        for k, v in loss_dict.items():
            weight = self.weight_dict[k] if "loss" in k else 1
            self._default_log(f"val_{k}", v * weight)
        without_aux = {
            k for k in self.weight_dict.keys() if k not in self.aux_weight_dict
        }
        self._default_log(
            "loss_val", sum(loss_dict[k] * self.weight_dict[k] for k in without_aux)
        )
        self._compute_metrics(batch, outputs)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        metrics = self.chamfer_map.compute()
        for key, value in metrics.items():
            self._default_log(f"val_{key}", value)
        self.chamfer_map.reset()

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        self.chamfer_map.activate_param_metrics = True

    @torch.no_grad()
    def test_step(self, batch: Data, batch_idx: int) -> None:
        outputs = self(batch)
        self._compute_metrics(batch, outputs)

    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        metrics = self.chamfer_map.compute()
        for key, value in metrics.items():
            self._default_log(key, value)
        
        value = self.chamfer_map.get_all_state_values_dict()    
        
        # TODO: remove

        with open(f'test_epoch_end_metrics.json', 'w') as f:
            json.dump(value, f, indent=4)
        self.chamfer_map.reset()

    def _compute_metrics(self, batch: Data, preds: dict[str, Tensor]):
        outputs = self.decode_predictions(batch, preds, reverse_norm=False)
        self.chamfer_map.update(outputs, batch)

    @torch.no_grad()
    def decode_predictions(
        self, batch: Data, preds: dict[str, Tensor], reverse_norm: bool = True
    ) -> list[Data]:
        outputs = []

        # Vectorized class prediction and score
        preds_class = preds["pred_class"].softmax(-1)
        object_class = preds_class.argmax(-1)  # (batch_size, num_preds)
        object_score = preds_class.max(-1).values  # (batch_size, num_preds)

        _, num_preds = object_class.shape
        for batch_idx in range(batch.batch_size):
            object_points = torch.zeros(num_preds, 64, 3).to(object_class.device)
            object_points[object_class[batch_idx] == 1] = transform_gripper_points(
                preds["pred_gripper_params"][batch_idx][
                    object_class[batch_idx] == 1, :3
                ],
                preds["pred_gripper_params"][batch_idx][
                    object_class[batch_idx] == 1, 3:7
                ],
                (
                    preds["pred_gripper_params"][batch_idx][
                        object_class[batch_idx] == 1, 7
                    ]
                    + 1.0
                )
                / 2.0,
                self.gripper_main_points,
                self.gripper_s1_points,
                self.gripper_s2_points,
                batch.scale[batch_idx].repeat((object_class[batch_idx] == 1).sum()),
            )
            object_points[object_class[batch_idx] == 2] = (
                transform_loading_or_pallet_points(
                    preds["pred_loading_platform_params"][batch_idx][
                        object_class[batch_idx] == 2, :3
                    ],
                    preds["pred_loading_platform_params"][batch_idx][
                        object_class[batch_idx] == 2, 3:7
                    ],
                    self.loading_platform_points,
                    batch.scale[batch_idx].repeat((object_class[batch_idx] == 2).sum()),
                )
            )
            object_points[object_class[batch_idx] == 3] = (
                transform_loading_or_pallet_points(
                    preds["pred_pallet_params"][batch_idx][
                        object_class[batch_idx] == 3, :3
                    ],
                    preds["pred_pallet_params"][batch_idx][
                        object_class[batch_idx] == 3, 3:7
                    ],
                    self.pallet_points,
                    batch.scale[batch_idx].repeat((object_class[batch_idx] == 3).sum()),
                )
            )

            output = Data(
                pos=batch.pos[batch.batch == batch_idx].clone(),  # point cloud
                object_class=object_class[batch_idx],  # class of each polyline
                object_score=object_score[batch_idx],  # score of polyline class
                gripper_params=preds["pred_gripper_params"][
                    batch_idx
                ],  # prediction of gripper parameters
                gripper_pos=preds["pred_gripper_params"][batch_idx][:, :3],
                loading_platform_params=preds["pred_loading_platform_params"][
                    batch_idx
                ],
                loading_platform_pos=preds["pred_loading_platform_params"][batch_idx][
                    :, :3
                ],
                pallet_params=preds["pred_pallet_params"][batch_idx],
                pallet_pos=preds["pred_pallet_params"][batch_idx][:, :3],
                object_points=object_points,  # points sampled on the gripper
                query_xyz=preds["query_xyz"][
                    batch_idx
                ],  # query points for the transformer
            )

            if reverse_norm:
                output.center = batch.center[batch_idx]
                output.scale = batch.scale[batch_idx]

                output = reverse_normalize_and_scale(
                    output,
                    extra_fields=[
                        "gripper_pos",
                        "loading_platform_pos",
                        "pallet_pos",
                        "object_points",
                        "query_xyz",
                    ],
                )
                output.gripper_params[:, :3] = output.gripper_pos
                output.loading_platform_params[:, :3] = output.loading_platform_pos
                output.pallet_params[:, :3] = output.pallet_pos
                del output.gripper_pos
                del output.loading_platform_pos
                del output.pallet_pos
                del output.center
                output.scale = torch.tensor(1.0).to(output.pos.device)
            outputs.append(output)

        return outputs

    def _default_log(self, name: str, value: Tensor) -> None:
        batch_size = (
            self.config.batch_size if self.training else self.config.batch_size_val
        )
        self.log(
            name,
            value,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        param_dict = None
        config = self.config
        if config.freeze_backbone:
            for param in self.pc_preencoder.parameters():
                param.requires_grad = False
            param_dict = self.parameters()
        elif config.lr != config.preencoder_lr:
            param_dict = [
                {
                    "params": [
                        p for n, p in self.named_parameters() if "pc_encoder" not in n
                    ]
                },
                {
                    "params": [
                        p for n, p in self.named_parameters() if "pc_encoder" in n
                    ],
                    "lr": self.config.preencoder_lr,
                },
            ]
        else:
            param_dict = self.parameters()
        # ----- OPTIMIZER -----
        optimizer = torch.optim.AdamW(param_dict, lr=self.config.lr)
        # ----- SCHEDULER -----
        if config.lr_warmup_epochs <= 0:
            # Only use step scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.lr_step,
                gamma=0.1,
            )
        else:
            # Use warmup + step scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.lr_warmup_start_factor,
                end_factor=1.0,
                total_iters=config.lr_warmup_epochs,
            )
            step_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.lr_step - config.lr_warmup_epochs,
                gamma=0.1,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, step_scheduler],
                milestones=[config.lr_warmup_epochs],
            )

        return [optimizer], {"scheduler": scheduler, "interval": "epoch"}
