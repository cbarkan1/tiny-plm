import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
import tomllib

import torch

from plm_pretraining.tiny_protein_lm import TinyProteinLM
from plm_pretraining.train import TrainConfig, build_vocab, create_dataloader, evaluate, train_one_epoch
from plm_pretraining.utils import save_loss_plot


@dataclass(frozen=True)
class RunConfig:
    seed: int
    epochs: int
    split_name: str
    device: str
    train_fasta: Path
    validation_fasta: Path
    checkpoint_path: Path
    run_metadata_path: Path
    train_loss_svg_path: Path
    max_len: int
    batch_size: int
    mask_prob: float
    lr: float
    weight_decay: float
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny protein LM from a TOML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML run config.")
    return parser.parse_args()


def load_config(config_path: Path) -> RunConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    train = raw["train"]
    model = raw["model"]

    return RunConfig(
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        split_name=str(raw["split_name"]),
        device=str(raw.get("device", "cpu")),
        train_fasta=Path(paths["train_fasta"]),
        validation_fasta=Path(paths["validation_fasta"]),
        checkpoint_path=Path(paths["checkpoint_path"]),
        run_metadata_path=Path(paths["run_metadata_path"]),
        train_loss_svg_path=Path(paths["train_loss_svg_path"]),
        max_len=int(train["max_len"]),
        batch_size=int(train["batch_size"]),
        mask_prob=float(train["mask_prob"]),
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        d_ff=int(model["d_ff"]),
        n_layers=int(model["n_layers"]),
        dropout=float(model["dropout"]),
    )


def main() -> None:
    args = parse_args()
    run_cfg = load_config(args.config)
    run_start = time.time()
    random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)
    device = torch.device(run_cfg.device)
    config = TrainConfig(
        max_len=run_cfg.max_len,
        batch_size=run_cfg.batch_size,
        mask_prob=run_cfg.mask_prob,
        lr=run_cfg.lr,
        weight_decay=run_cfg.weight_decay,
    )

    stoi, _ = build_vocab()
    pad_id = stoi["<pad>"]

    train_loader = create_dataloader(
        fasta_path=run_cfg.train_fasta,
        stoi=stoi,
        batch_size=config.batch_size,
        max_len=config.max_len,
        mask_prob=config.mask_prob,
        shuffle=True,
    )
    validation_loader = create_dataloader(
        fasta_path=run_cfg.validation_fasta,
        stoi=stoi,
        batch_size=config.batch_size,
        max_len=config.max_len,
        mask_prob=config.mask_prob,
        shuffle=False,
    )

    model = TinyProteinLM(
        vocab_size=len(stoi),
        max_len=config.max_len,
        d_model=run_cfg.d_model,
        n_heads=run_cfg.n_heads,
        d_ff=run_cfg.d_ff,
        n_layers=run_cfg.n_layers,
        dropout=run_cfg.dropout,
        pad_token_id=pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    initial_val_loss = evaluate(model, validation_loader, device, progress_desc="val   init")
    print(
        f"epoch=00 train_loss=nan val_loss={initial_val_loss:.4f} "
        f"train_exp=nan val_exp={math.exp(initial_val_loss):.3f}"
    )

    epoch_metrics: list[dict[str, float | int]] = []
    train_loss_steps: list[float] = []

    for epoch in range(1, run_cfg.epochs + 1):
        train_loss, batch_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            progress_desc=f"train e{epoch:02d}",
        )
        val_loss = evaluate(
            model,
            validation_loader,
            device,
            progress_desc=f"val   e{epoch:02d}",
        )
        train_loss_steps.extend(batch_losses)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_exp": math.exp(train_loss),
                "val_exp": math.exp(val_loss),
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_exp={math.exp(train_loss):.3f} val_exp={math.exp(val_loss):.3f}"
        )

    run_cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_cfg.run_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    run_cfg.train_loss_svg_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "seed": run_cfg.seed,
            "epochs": run_cfg.epochs,
        },
        run_cfg.checkpoint_path,
    )
    print(f"Saved checkpoint to {run_cfg.checkpoint_path}")

    final_val_loss = epoch_metrics[-1]["val_loss"] if epoch_metrics else initial_val_loss
    run_metadata = {
        "split_name": run_cfg.split_name,
        "seed": run_cfg.seed,
        "epochs": run_cfg.epochs,
        "train_fasta": str(run_cfg.train_fasta),
        "validation_fasta": str(run_cfg.validation_fasta),
        "checkpoint_path": str(run_cfg.checkpoint_path),
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "val_loss_delta": final_val_loss - initial_val_loss,
        "num_train_steps": len(train_loss_steps),
        "runtime_seconds": time.time() - run_start,
        "epoch_metrics": epoch_metrics,
        "train_loss_steps": train_loss_steps,
    }
    run_cfg.run_metadata_path.write_text(json.dumps(run_metadata, indent=2))
    print(f"Saved run metadata to {run_cfg.run_metadata_path}")

    save_loss_plot(train_loss_steps, run_cfg.train_loss_svg_path)
    print(f"Saved train loss plot to {run_cfg.train_loss_svg_path}")


if __name__ == "__main__":
    main()
