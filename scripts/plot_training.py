#!/usr/bin/env python3
"""Plot training curves from figures/training_history.json.

Run after training to generate figures/training_curves.png.
Use this if matplotlib fails during train_model.py (e.g. libstdc++ issue).
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
history_path = project_root / "figures" / "training_history.json"

if not history_path.exists():
    print(f"Not found: {history_path}. Run train_model.py first.")
    sys.exit(1)

with open(history_path) as f:
    data = json.load(f)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(data["train_losses"], label="Train")
ax.plot(data["val_losses"], label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs validation loss")
ax.legend()
out = project_root / "figures" / "training_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
