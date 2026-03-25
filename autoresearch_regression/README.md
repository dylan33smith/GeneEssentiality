# Fitness regression autoresearch harness

Karpathy-style layout: **`prepare.py`** (fixed loaders + eval) and **`experiments/`** (one script per experiment).

- **[`RUNBOOK.md`](RUNBOOK.md)** — Full guide for **autonomous agents** (OpenClaw-like loops) **or** **you + an assistant**: prerequisites, commands, metrics contract, git/logging, failures.
- **[`program.md`](program.md)** — Short rules to paste into an agent prompt.

## Metrics (fixed in `prepare.py`)

- **Primary:** validation RMSE on `fit` (lower is better).
- **Secondary:** mean within-gene Spearman across conditions (higher is better). Implemented in [`src/model/regression_metrics.py`](../src/model/regression_metrics.py) for reuse in notebooks.

## File structure

```text
autoresearch_regression/
  prepare.py                    # fixed loaders + evaluate() — DO NOT EDIT
  run_all.py                    # run all (or selected) experiments, append results.tsv
  experiments/
    exp01_baseline.py           # MSE, default 2-hidden MLP
    exp02_wider.py              # 4096->2048, LayerNorm, higher dropout
    exp03_layernorm_gelu.py     # LayerNorm + GELU activations
    exp04_batchnorm.py          # BatchNorm MLP
    exp05_deep.py               # 4-layer MLP (2048->1024->512->256)
    exp06_huber.py              # Huber loss (tunable delta)
    exp07_residual.py           # gene_mean + condition offset subnetworks
    exp08_ranking_mse.py        # MSE + within-gene listwise ranking loss
    exp09_pairwise.py           # MSE + within-gene pairwise margin loss
    exp10_lr_sweep.py           # lower LR, less WD, longer patience
  archive/                      # old broken/superseded scripts + logs
  results.tsv                   # experiment log (gitignored)
  results.tsv.example           # template for results.tsv
  README.md  /  RUNBOOK.md  /  program.md
```

## Python environment

**No specific conda env is required** — only Python ≥ 3.10 and dependencies from [`pyproject.toml`](../pyproject.toml). If you already use a **conda env for PyTorch** (e.g. `pytorch`) for `model_reg.ipynb`, activate it and run `pip install -e .` once in the repo root, then use the same env for the harness. See **[`RUNBOOK.md`](RUNBOOK.md)** section *Python environment* for agents **not** creating duplicate environments.

## Run

From the **repository root** (so `src` imports work), with your chosen env activated:

```bash
# Single experiment
python -m autoresearch_regression.experiments.exp01_baseline

# All experiments (results appended to results.tsv)
python -m autoresearch_regression.run_all

# Selected experiments only
python -m autoresearch_regression.run_all --experiments exp01_baseline,exp06_huber

# Resume the default queue from a given experiment (skips earlier ones in order)
python -m autoresearch_regression.run_all --start-from exp05_deep

# Quick smoke-test on small dataset
DATA_SUBSET=mvp AUTORESEARCH_EPOCHS=2 python -m autoresearch_regression.run_all --experiments exp01_baseline
```

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_SUBSET` | `processed` | `data/<subset>/` |
| `BATCH_SIZE` | `2048` | Loader batch size |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `AUTORESEARCH_EPOCHS` | `8` | Max training epochs (early stopping may end sooner) |
| `EARLY_STOP_PATIENCE` | `3` | Stop if `val_rmse` does not improve for this many epochs (`0` = off) |
| `EARLY_STOP_MIN_DELTA` | `1e-4` | Minimum `val_rmse` improvement to reset the early-stop counter |
| `LR` | varies | AdamW learning rate (experiment-specific default) |
| `WEIGHT_DECAY` | varies | AdamW weight decay |
| `RANK_WEIGHT` | `0.1`/`0.2` | Ranking loss weight (exp08, exp09) |
| `HUBER_DELTA` | `0.5` | Huber loss delta (exp06) |
| `PAIR_MARGIN` | `0.1` | Pairwise margin (exp09) |

## Dependencies

Uses project [`pyproject.toml`](../pyproject.toml) (including `scipy` for Spearman).
