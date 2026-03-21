# Fitness regression autoresearch harness

Karpathy-style layout: **`prepare.py`** (fixed loaders + eval) and **`train.py`** (agent-editable model and loop).

- **[`RUNBOOK.md`](RUNBOOK.md)** — Full guide for **autonomous agents** (OpenClaw-like loops) **or** **you + an assistant**: prerequisites, commands, metrics contract, git/logging, failures.
- **[`program.md`](program.md)** — Short rules to paste into an agent prompt.

## Metrics (fixed in `prepare.py`)

- **Primary:** validation RMSE on `fit` (lower is better).
- **Secondary:** mean within-gene Spearman across conditions (higher is better). Implemented in [`src/model/regression_metrics.py`](../src/model/regression_metrics.py) for reuse in notebooks.

## Python environment

**No specific conda env is required** — only Python ≥ 3.10 and dependencies from [`pyproject.toml`](../pyproject.toml). If you already use a **conda env for PyTorch** (e.g. `pytorch`) for `model_reg.ipynb`, activate it and run `pip install -e .` once in the repo root, then use the same env for the harness. See **[`RUNBOOK.md`](RUNBOOK.md)** section *Python environment (conda, venv, uv)* for conda vs venv and for agents **not** creating duplicate environments.

## Run

From the **repository root** (so `src` imports work), with your chosen env activated:

```bash
python -m autoresearch_regression.train
```

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_SUBSET` | `processed` | `data/<subset>/` |
| `BATCH_SIZE` | `2048` | Loader batch size |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `AUTORESEARCH_EPOCHS` | `20` | Training epochs (baseline) |
| `LR` | `1e-3` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `HIDDEN1` / `HIDDEN2` / `DROPOUT` | 2048 / 512 / 0.3 | Baseline MLP shape |

## Experiment log

Copy `results.tsv.example` to `results.tsv` (gitignored) and append one row per run. Do not commit `results.tsv`.

## Dependencies

Uses project [`pyproject.toml`](../pyproject.toml) (including `scipy` for Spearman).
