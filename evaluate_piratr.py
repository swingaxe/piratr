import torch
import argparse
from pathlib import Path
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.loader import DataLoader
from piratr import (
    load_args,
    load_weights,
    build_model_config,
    build_model,
    build_dataset_config,
    build_dataset,
)

torch.set_float32_matmul_precision("highest")  # medium, high, highest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to checkpoint"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="path to config",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to input folder.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    pl.seed_everything(42)
    torch_geometric.seed_everything(42)
    parsed_args = parse_args()
    args = load_args(parsed_args.config)
    data_dir = parsed_args.path
    assert Path(data_dir).is_dir()
    dataset_config = build_dataset_config(args, data_dir, False)
    dataset = build_dataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model_params = build_model_config(args)
    model_params.auxiliary_loss = False
    model = build_model(model_params)
    load_weights(model, parsed_args.checkpoint)
    model.eval()
    model.to("cuda")

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
