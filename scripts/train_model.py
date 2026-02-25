#!/usr/bin/env python3
"""Train essentiality MLP from ProteomeLM embeddings.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config config/model.yaml --epochs 50 --plot
    python scripts/train_model.py --config config/model_binary.yaml --output-name binary
    python scripts/train_model.py --early-stopping 15 --dropout 0.3
    python scripts/train_model.py --scheduler plateau --scheduler-patience 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.model import create_datasets_from_config
from src.model.network import EssentialityMLP
from src.model.train import (
    collect_predictions,
    compute_metrics,
    default_class_names,
    eval_epoch,
    make_loss_fn,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train essentiality MLP")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "model.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--plot", action="store_true", help="Save training curves plot")
    parser.add_argument(
        "--early-stopping", type=int, default=None,
        help="Stop if val metric does not improve for N epochs",
    )
    parser.add_argument(
        "--early-stop-on", choices=["loss", "auprc_ae"], default="loss",
        help="Metric for early stopping",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler", choices=["none", "plateau"], default="none",
        help="Learning rate scheduler (default: none)",
    )
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument(
        "--output-name", type=str, default="best_model",
        help="Base name for model checkpoint and history files (default: best_model)",
    )
    return parser.parse_args()


def load_class_names(config_path: Path, n_classes: int) -> list[str]:
    """Read class_names from the config's classification section, or use defaults."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("classification", {}).get("class_names")
    if names and len(names) == n_classes:
        return list(names)
    return default_class_names(n_classes)


def build_model(
    input_dim: int,
    n_classes: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[EssentialityMLP, torch.optim.Optimizer]:
    """Construct model and optimizer."""
    model = EssentialityMLP(
        input_dim=input_dim, n_classes=n_classes, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    return model, optimizer


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int,
    class_names: list[str],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[dict]]:
    """Execute the training loop with optional early stopping and LR scheduling.

    Returns:
        Tuple of (train_losses, val_losses, val_metrics_history).
    """
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    best_val_metric = float("-inf") if args.early_stop_on == "auprc_ae" else float("inf")
    epochs_without_improvement = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_metrics_history: list[dict] = []

    auprc_key = f"auprc_{class_names[0]}"

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        logits, labels = collect_predictions(model, val_loader, device)
        val_metrics = compute_metrics(
            logits, labels, n_classes=n_classes, class_names=class_names,
        )

        summary_keys = [("acc", "accuracy"), ("F1", "weighted_f1")]
        for name in class_names:
            summary_keys.append((f"AUPRC_{name[:4]}", f"auprc_{name}"))
        metric_str = " ".join(
            f"{tag}={val_metrics[key]:.3f}" if isinstance(val_metrics.get(key), float)
            else f"{tag}={val_metrics.get(key)}"
            for tag, key in summary_keys
        )
        lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if scheduler else ""
        print(
            f"Epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f}"
            f" | {metric_str}{lr_str}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)

        if scheduler is not None:
            scheduler.step(val_loss)

        if args.early_stop_on == "auprc_ae":
            current = val_metrics.get(auprc_key) or 0.0
            improved = current > best_val_metric
        else:
            current = val_loss
            improved = current < best_val_metric

        if improved:
            best_val_metric = current
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping is not None
            and epochs_without_improvement >= args.early_stopping
        ):
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {args.early_stopping} epochs)"
            )
            break

    return train_losses, val_losses, val_metrics_history


def save_history(
    train_losses: list[float],
    val_losses: list[float],
    val_metrics_history: list[dict],
    path: Path,
) -> None:
    """Save training history to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_metrics": val_metrics_history,
            },
            f,
            indent=2,
        )
    print(f"Saved loss history to {path}")


def plot_curves(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    """Save a training-vs-validation loss plot."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label="Train")
        ax.plot(val_losses, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs validation loss")
        ax.legend()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {path}")
    except ImportError as e:
        print(f"Could not plot (matplotlib issue): {e}")


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path,
    n_classes: int,
    class_names: list[str],
) -> dict:
    """Load best checkpoint and evaluate on the test set."""
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    test_logits, test_labels = collect_predictions(model, test_loader, device)
    return compute_metrics(
        test_logits, test_labels, n_classes=n_classes, class_names=class_names,
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    train_ds, val_ds, test_ds = create_datasets_from_config(args.config)
    n_classes = train_ds.n_classes
    class_names = load_class_names(args.config, n_classes)

    print(f"Classes: {n_classes} ({', '.join(class_names)})")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    loss_fn = make_loss_fn(train_ds.get_labels(), n_classes=n_classes, device=device)
    model, optimizer = build_model(
        input_dim=train_ds.embedding_dim,
        n_classes=n_classes,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{args.output_name}.pth"

    train_losses, val_losses, val_metrics_history = run_training_loop(
        model, train_loader, val_loader, loss_fn, optimizer, device,
        n_classes=n_classes, class_names=class_names,
        checkpoint_path=checkpoint_path, args=args,
    )

    history_path = project_root / "figures" / f"training_history_{args.output_name}.json"
    save_history(train_losses, val_losses, val_metrics_history, history_path)

    if args.plot:
        plot_curves(
            train_losses, val_losses,
            project_root / "figures" / f"training_curves_{args.output_name}.png",
        )

    test_metrics = evaluate_test_set(
        model, test_loader, device, checkpoint_path,
        n_classes=n_classes, class_names=class_names,
    )
    print(f"\n=== Test set (best checkpoint: {checkpoint_path.name}) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
