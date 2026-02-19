#!/usr/bin/env python3
"""Train essentiality MLP from ProteomeLM embeddings.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config config/model.yaml --epochs 50 --plot
    python scripts/train_model.py --early-stopping 15 --dropout 0.3
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
from torch.utils.data import DataLoader

from src.model import create_datasets_from_config
from src.model.network import EssentialityMLP
from src.model.train import (
    collect_predictions,
    compute_metrics,
    eval_epoch,
    make_loss_fn,
    train_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train essentiality MLP")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "model.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="Save training curves plot (requires matplotlib)")
    parser.add_argument("--early-stopping", type=int, default=None, help="Stop if val metric does not improve for N epochs")
    parser.add_argument("--early-stop-on", choices=["loss", "auprc_ae"], default="loss", help="Metric for early stopping")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (try 0.3-0.4 if overfitting)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (try 1e-3 if overfitting)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Datasets
    train_ds, val_ds, test_ds = create_datasets_from_config(args.config)
    train_labels = train_ds.get_labels()

    # 2. Loss (class-weighted CrossEntropy)
    loss_fn = make_loss_fn(train_labels, n_classes=3, device=device)

    # 3. Model and optimizer (dropout configurable for overfitting)
    model = EssentialityMLP(
        input_dim=train_ds.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 4. DataLoaders (shuffle=True for train, False for val/test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 5. Training loop
    (project_root / "models").mkdir(parents=True, exist_ok=True)
    best_val_metric = float("-inf") if args.early_stop_on == "auprc_ae" else float("inf")
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    val_metrics_history: list[dict] = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        logits, labels = collect_predictions(model, val_loader, device)
        val_metrics = compute_metrics(logits, labels, n_classes=3)
        val_metrics_history.append(val_metrics)

        metric_str = " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in [
                ("acc", val_metrics["accuracy"]),
                ("F1", val_metrics["weighted_f1"]),
                ("AUPRC_ae", val_metrics.get("auprc_always_essential")),
                ("AUPRC_cond", val_metrics.get("auprc_conditional")),
            ]
        )
        print(f"Epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} | {metric_str}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if args.early_stop_on == "auprc_ae":
            current = val_metrics.get("auprc_always_essential") or 0.0
            improved = current > best_val_metric
        else:
            current = val_loss
            improved = current < best_val_metric

        if improved:
            best_val_metric = current
            epochs_without_improvement = 0
            torch.save(model.state_dict(), project_root / "models" / "best_model.pth")
        else:
            epochs_without_improvement += 1

        if args.early_stopping is not None and epochs_without_improvement >= args.early_stopping:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stopping} epochs)")
            break

    # 6. Always save loss and metrics history (so plot/data always matches this run)
    (project_root / "figures").mkdir(parents=True, exist_ok=True)
    history_path = project_root / "figures" / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_metrics": val_metrics_history,
            },
            f,
            indent=2,
        )
    print(f"Saved loss history to {history_path}")

    # 7. Plot (optional; requires matplotlib)
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(train_losses, label="Train")
            ax.plot(val_losses, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training vs validation loss")
            ax.legend()
            fig.savefig(project_root / "figures" / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to figures/training_curves.png")
        except ImportError as e:
            print(f"Could not plot (matplotlib issue): {e}. Load figures/training_history.json in a notebook to plot.")

    # 8. Final test set evaluation (load best checkpoint)
    model.load_state_dict(torch.load(project_root / "models" / "best_model.pth", map_location=device))
    test_logits, test_labels = collect_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_logits, test_labels, n_classes=3)
    print("\n=== Test set (best checkpoint) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
