# Autoresearch program (fitness regression)

Instructions for an autonomous coding agent working in **`autoresearch_regression/`**.

For prerequisites, full commands, git/logging, and human-in-the-loop workflows, see **[`RUNBOOK.md`](RUNBOOK.md)**.

**Python env:** There is **no** required conda environment name. Use the **existing** project environment (where `pip install -e .` was run). **Do not** create a new conda/venv unless a human explicitly asks — install missing deps into the current env instead.

## Scope

- **Edit only `train.py`** — model, optimizer, loss, training loop, hyperparameters, architecture.
- **Do not modify `prepare.py`** — it owns data loading and the official **validation RMSE** and **mean within-gene Spearman** definitions.
- **Do not change** [`src/model/regression_dataset.py`](../src/model/regression_dataset.py) or [`src/model/regression_metrics.py`](../src/model/regression_metrics.py) unless a human explicitly allows it (breaks reproducibility of the harness contract).

## Metrics

- **Primary (minimize):** `val_rmse` — printed as `val_rmse: ...` after training.
- **Secondary (maximize):** `mean_within_gene_spearman` — printed on the next line.
- If `val_rmse` improves (lower), keep the commit; if equal or worse, revert (unless exploring a different trade-off with human approval).

## Running an experiment

```bash
cd /path/to/GeneEssentiality
python -m autoresearch_regression.train > autoresearch_regression/run.log 2>&1
```

grep metrics:

```bash
grep "^val_rmse:" autoresearch_regression/run.log
grep "^mean_within_gene_spearman:" autoresearch_regression/run.log
```

## Logging results

Append rows to `results.tsv` (tab-separated, copy from `results.tsv.example`). Suggested columns:

`commit`, `val_rmse`, `mean_within_gene_spearman`, `memory_gb`, `status`, `description`

- `status`: `keep`, `discard`, or `crash`
- Do not commit `results.tsv` to git

## Budget

Baseline uses **`AUTORESEARCH_EPOCHS`** (default 20). You may change the loop in `train.py` to wall-clock or a different epoch cap; document in `description` when comparing runs.

## VRAM

Some increase is acceptable for meaningful `val_rmse` gains; avoid OOM — if a run crashes, log `crash` and revert.
