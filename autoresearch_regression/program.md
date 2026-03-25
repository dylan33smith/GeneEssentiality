# Autoresearch program (fitness regression)

Instructions for an autonomous coding agent working in **`autoresearch_regression/`**.

For prerequisites, full commands, git/logging, and human-in-the-loop workflows, see **[`RUNBOOK.md`](RUNBOOK.md)**.

**Python env:** There is **no** required conda environment name. Use the **existing** project environment (where `pip install -e .` was run). **Do not** create a new conda/venv unless a human explicitly asks — install missing deps into the current env instead.

## Scope

- **Edit files under `experiments/`** — model, optimizer, loss, training loop, hyperparameters, architecture.
- **Do not modify `prepare.py`** — it owns data loading and the official **validation RMSE** and **mean within-gene Spearman** definitions.
- **Do not change** [`src/model/regression_dataset.py`](../src/model/regression_dataset.py) or [`src/model/regression_metrics.py`](../src/model/regression_metrics.py) unless a human explicitly allows it (breaks reproducibility of the harness contract).

## Metrics

- **Primary (minimize):** `val_rmse` — printed as `val_rmse: ...` after training.
- **Secondary (maximize):** `mean_within_gene_spearman` — printed on the next line.
- If `val_rmse` improves (lower), keep the commit; if equal or worse, revert (unless exploring a different trade-off with human approval).

## Running experiments

```bash
cd /path/to/GeneEssentiality

# Single experiment
python -m autoresearch_regression.experiments.exp01_baseline

# All experiments
python -m autoresearch_regression.run_all

# Selected experiments
python -m autoresearch_regression.run_all --experiments exp01_baseline,exp06_huber

# Resume default queue from exp05 onward (after changing epochs, etc.)
python -m autoresearch_regression.run_all --start-from exp05_deep

# With log capture
python -m autoresearch_regression.run_all > autoresearch_regression/run.log 2>&1
```

grep metrics:

```bash
grep "^val_rmse:" autoresearch_regression/run.log
grep "^mean_within_gene_spearman:" autoresearch_regression/run.log
```

## Available experiments

| Name | Idea |
|------|------|
| `exp01_baseline` | MSE, 2-hidden MLP (2048→512) |
| `exp02_wider` | 4096→2048, LayerNorm, 0.4 dropout |
| `exp03_layernorm_gelu` | LayerNorm + GELU activations |
| `exp04_batchnorm` | BatchNorm + ReLU |
| `exp05_deep` | 4-layer MLP (2048→1024→512→256) |
| `exp06_huber` | Huber loss (delta=0.5) |
| `exp07_residual` | gene_mean + condition_offset subnetworks |
| `exp08_ranking_mse` | MSE + listwise within-gene ranking loss |
| `exp09_pairwise` | MSE + within-gene pairwise margin loss |
| `exp10_lr_sweep` | LR=8e-4, WD=5e-5, patience=8 |

## Logging results

`run_all.py` automatically appends rows to `results.tsv`. For manual runs, append a row (tab-separated, see `results.tsv.example`).

- `status`: `ok` or `crash`
- Do not commit `results.tsv` to git

## Budget

Default max epochs is **8** per experiment (`AUTORESEARCH_EPOCHS`). **Early stopping** is on by default (`EARLY_STOP_PATIENCE=3`, `EARLY_STOP_MIN_DELTA=1e-4`); set `EARLY_STOP_PATIENCE=0` to disable. Runs may end before the max epoch count.

## VRAM

Some increase is acceptable for meaningful `val_rmse` gains; avoid OOM — if a run crashes, log `crash` and revert.
