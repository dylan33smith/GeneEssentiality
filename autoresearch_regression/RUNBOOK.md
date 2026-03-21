# Autoresearch harness — operator runbook

This document is for **anyone** running the fitness regression harness: an **autonomous coding agent** (OpenClaw-like loop, scripted runner, etc.) or **you + a human-in-the-loop assistant** in an IDE. It collects everything you need in one place.

Shorter agent-only rules live in [`program.md`](program.md). This file is the full picture.

---

## 1. What this harness is

- **Task:** Predict scalar fitness (`fit`) from a fixed ProteomeLM gene embedding concatenated with a nutrient/condition one-hot (same data as [`model_reg.ipynb`](../model_reg.ipynb)).
- **Pattern:** Like [karpathy/autoresearch](https://github.com/karpathy/autoresearch): **`prepare.py`** is fixed (data + official metrics); **`train.py`** is the only file meant for iterative experiments (model, loss, optimizer, loop).
- **Primary metric (minimize):** validation **RMSE** on `fit`.
- **Secondary metric (maximize):** **mean within-gene Spearman** (ranking conditions per gene). Implemented in [`src/model/regression_metrics.py`](../src/model/regression_metrics.py).

---

## 2. Prerequisites

| Requirement | Details |
|-------------|---------|
| **Working directory** | Always run commands from the **repository root** (`GeneEssentiality/`), not from inside `autoresearch_regression/`, so `import src...` resolves. |
| **Python env** | See **§3** — any env with Python ≥ 3.10 and project deps; **no specific conda name** is required. |
| **Data** | Under `data/<subset>/` (default `processed`, or `mvp` for smaller runs). Must include: `regression_dataset.parquet`, `mmseqs_splits.csv`, `condition_vocab_regression.json`, `ProtLM_embeddings_layer8/*.pt`. Built by your existing pipeline / notebook prerequisites. |
| **GPU** | Optional but typical; CPU runs work but are slow on full `processed` data. |

---

## 3. Python environment (conda, venv, uv)

**There is no required conda environment name** in this repo: no `environment.yml` is part of the harness contract. What matters is:

- **Python** `>= 3.10` ([`pyproject.toml`](../pyproject.toml) `requires-python`).
- **Dependencies** matching the project (notably **PyTorch**, **pandas**, **scipy**, etc.) — usually via `pip install -e .` from the repo root.

**If you already train the MLP in a conda env** (e.g. `conda activate pytorch`):

- Use **that same env** for `python -m autoresearch_regression.train` so PyTorch/CUDA match your notebook.
- One-time install (or after `pyproject.toml` changes):

  ```bash
  conda activate pytorch    # your env name here
  cd /path/to/GeneEssentiality
  pip install -e .
  ```

**Autonomous agents:** Do **not** create a new conda environment, venv, or `uv` project **unless the human explicitly asks**. Reuse the environment the project already uses; if a package is missing, install into **that** env (e.g. `pip install -e .` or `pip install scipy`), not a duplicate stack.

**Other setups** (`uv sync`, plain `venv`, system Python) are fine as long as the same dependency set is available — the harness does not care which tool created the env.

---

## 4. Files and responsibilities

| Path | Role |
|------|------|
| [`prepare.py`](prepare.py) | **Do not edit** during autoresearch. Loads data, builds `evaluate()` with fixed RMSE + within-gene Spearman. |
| [`train.py`](train.py) | **Edit this** for experiments: architecture, hyperparameters, training loop, optional wall-clock budget. |
| [`program.md`](program.md) | Short rules for an autonomous agent (scope, grep, `results.tsv`). |
| [`results.tsv.example`](results.tsv.example) | Template for the experiment log (copy to `results.tsv`; `results.tsv` is gitignored). |
| [`../src/model/regression_metrics.py`](../src/model/regression_metrics.py) | Shared metric math (use from notebooks too). Change only with human agreement — otherwise harness comparisons break. |
| [`../src/model/regression_dataset.py`](../src/model/regression_dataset.py) | Loaders and datasets. Same rule: no casual edits during autoresearch. |

---

## 5. How to run one experiment

From **repo root**:

```bash
python -m autoresearch_regression.train
```

Redirect full output to a log (recommended for agents):

```bash
python -m autoresearch_regression.train > autoresearch_regression/run.log 2>&1
```

**Parse the official scores** (also printed to stdout):

```bash
grep "^val_rmse:" autoresearch_regression/run.log
grep "^mean_within_gene_spearman:" autoresearch_regression/run.log
```

Optional extra lines: `n_genes_used_for_spearman`, `n_genes_single_row`, `n_genes_nan_rho`.

---

## 6. Environment variables (all optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_SUBSET` | `processed` | Subdirectory under `data/` (`processed` or `mvp`). |
| `BATCH_SIZE` | `2048` | Batch size for train/val/test loaders. |
| `NUM_WORKERS` | `4` | `DataLoader` worker processes (`0` = main process only). |
| `AUTORESEARCH_EPOCHS` | `20` | Epochs for the baseline loop in `train.py`. |
| `LR` | `1e-3` | AdamW learning rate (baseline). |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay. |
| `HIDDEN1`, `HIDDEN2`, `DROPOUT` | `2048`, `512`, `0.3` | Baseline MLP sizes (if still using default builder in `train.py`). |

Use **`DATA_SUBSET=mvp`** for quicker iteration; **`processed`** for full-scale runs (long runtime, large RAM/VRAM).

---

## 7. Evaluation contract (do not “improve” metrics in `train.py`)

- **Official** validation scores come **only** from [`prepare.evaluate`](prepare.py) after training. It aligns `val_df` row order with the validation `DataLoader` and calls [`regression_metrics`](../src/model/regression_metrics.py).
- Do **not** redefine RMSE or within-gene Spearman inside `train.py` for reporting — you can log extra diagnostics, but **keep/revert decisions** should use the printed `val_rmse` / `mean_within_gene_spearman` from `evaluate()`.

---

## 8. Git workflow (autonomous agent)

1. Work on a dedicated branch (e.g. `autoresearch/regression-<date>`).
2. Change **`train.py` only** (unless a human approves other files).
3. Commit: `git add autoresearch_regression/train.py && git commit -m "..."`.
4. Run training; capture `val_rmse` (lower is better).
5. **Keep** commit if primary metric improved; else **`git reset --hard`** to previous good commit (or mark `discard` in `results.tsv`).
6. Append one tab-separated row to **`results.tsv`** (see `results.tsv.example`). Do not commit `results.tsv` or `run.log` if your policy gitignores them.

---

## 9. Logging runs (`results.tsv`)

- Copy [`results.tsv.example`](results.tsv.example) to `results.tsv` once.
- Columns: `commit`, `val_rmse`, `mean_within_gene_spearman`, `memory_gb`, `status`, `description`.
- `status`: `keep`, `discard`, or `crash`.
- `description`: short note (e.g. “wider hidden, Huber loss”).

---

## 10. Failure modes

| Symptom | What to do |
|---------|------------|
| **OOM / CUDA OOM** | Reduce `BATCH_SIZE` or model size in `train.py`; log `crash` in TSV; revert commit if the idea didn’t help. |
| **Host RAM OOM** (process **killed** / “Killed”, no Python traceback) | System ran out of **CPU RAM** (parquet + embedding store + workers). Try `NUM_WORKERS=0` or `2`, smaller `BATCH_SIZE`, or `DATA_SUBSET=mvp`; close other jobs. |
| **Import errors** (`ModuleNotFoundError: src`, `No module named torch`) | Confirm cwd is repo root; **`conda activate`** (or equivalent) so you use the env where `pip install -e .` was run; reinstall with `pip install -e .`; check `PYTHONPATH` is not overriding `src`. |
| **FileNotFoundError** for parquet, splits, embeddings, or vocab | Confirm `data/<subset>/` exists and pipeline artifacts are built; check `DATA_SUBSET` (`processed` vs `mvp`) matches where files live. |
| **`ValueError` about gene_keys / embeddings** | Some rows reference genes missing from the embedding store — usually a **data build** issue, not `train.py`. Fix upstream data / embedding dir; do not “fix” by skipping rows in `train.py` without human agreement. |
| **Length mismatch / `evaluate` RuntimeError** | Do not change `val_df` vs loader ordering; use `build_loaders_and_val_frame()` from `prepare` as wired. Do not shuffle val loader or reorder `val_df` independently. |
| **DataLoader / worker crash** (segfault, hang, pickling errors) | Set **`NUM_WORKERS=0`** (single-process loading). Some clusters or NFS setups are flaky with `num_workers > 0`. |
| **Very long runs** | Use `DATA_SUBSET=mvp`, lower `AUTORESEARCH_EPOCHS`, or a wall-clock cap in `train.py` while exploring. Full `processed` can take hours per epoch. |
| **SSH session dies mid-run** | For long jobs, use **`tmux`**, **`screen`**, or a job scheduler so the train process survives disconnects. |
| **`NaN` / `Inf` in loss** | Often learning rate or loss scaling; reduce `LR` or switch loss in `train.py`; check for bad data rows if persistent. |
| **Disk full** | Large logs (`run.log`), checkpoints if you add them, or temp files — free space or redirect logs elsewhere. |

**Non-fatal but worth checking**

- **`n_genes_nan_rho` large** — many genes have constant `fit` or predictions within the group (or near-constant); Spearman is skipped. Compare to `n_genes_used_for_spearman`; not a crash, but can limit how informative the secondary metric is.
- **`mean_within_gene_spearman` near 0 while `val_rmse` improves** — model may be learning **global** level more than **within-gene** ranking; expected sometimes; use both metrics as in the runbook.

---

## 11. Human + assistant (interactive) workflow

- You run or ask the assistant to run: `python -m autoresearch_regression.train` (with env vars as needed).
- You review `val_rmse` and `mean_within_gene_spearman`; you or the assistant edit **`train.py`** for the next idea.
- No overnight loop required — each cycle is **run → read metrics → decide → edit**.

Same files and metrics as an autonomous agent; only the driver changes (you vs script).

---

## 12. Where to look next

- [`README.md`](README.md) — quick commands and env table.
- [`program.md`](program.md) — minimal prompt for a coding agent.
- [`prepare.py`](prepare.py) — exact `evaluate()` and loader wiring.
- [`train.py`](train.py) — starting point for all experiments.
