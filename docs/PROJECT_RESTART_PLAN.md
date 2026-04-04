# Project restart plan (living document)

**Status:** Draft — revise as we learn.  
**Starting point (minimum):** `data/raw/feba.db`, `data/raw/aaseqs`, and `data/media_composition.xlsx`.  
**Starting point (recommended, to avoid long recompute):** The above **plus** frozen ProteomeLM tensor bundles (see §1.4). Everything else downstream is rebuilt deliberately.

---

## 0. Project goal and scientific context

### 0.1 What this project is

We predict **quantitative gene fitness** from **protein sequence context** and **environmental / growth condition context**, using publicly available **Tn-Seq** (transposon-insertion sequencing) data from the **Fitness Browser** database — no new wet-lab work.

**The core scientific question:** Given a gene (represented by its protein sequence embedding) and a growth condition (medium, stressor, concentration, temperature, etc.), can we predict how important that gene is for bacterial survival under that condition?

### 0.2 What Tn-Seq fitness data is

In a Tn-Seq experiment, a library of transposon-insertion mutants (each with a different gene disrupted) is grown under a specific condition. Genes whose disruption causes a growth defect have **negative fitness scores**; genes whose disruption is neutral have fitness near zero.


| Term | Meaning |
| ---- | ------- |
| **`fit`** | Log₂ fitness ratio for a gene in one experiment. Negative = gene disruption hurts growth. **Primary regression target.** |
| **`t`** | T-statistic for `fit`; confidence in that row’s estimate. Larger **absolute** *t* → more confident the `fit` value. |
| **`cor12`** | Replicate correlation for the **experiment** (same for all rows in that experiment). Low → noisy screen. |
| **One row** | One (gene, experiment) pair: one `fit` measurement. |


### 0.3 Why this is hard (and interesting)

- The **same gene** can be essential in one medium and non-essential in another ("conditional essentiality"). Predicting this condition-dependence, not just average gene importance, is the harder and more scientifically valuable task.
- Gene-level prediction from sequence alone (ignoring condition) is easier but less useful — prior work (ProteomeLM-Ess paper) already does this.
- The fitness matrix (genes × conditions) is **extremely sparse**: most genes are only measured in a few conditions, and most conditions only cover a subset of organisms.

---

## 1. Raw data: what we start with

### 1.1 `data/raw/feba.db`

SQLite export of the Fitness Browser. Key tables:


| Table           | Key columns                                                                           | Notes                         |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------- |
| `Gene` (type=1) | `orgId`, `locusId`, coordinates, `gene_length`                                        | Protein-coding genes only     |
| `Experiment`    | `orgId`, `expName`, `media`, `condition_1..3`, concentrations, `temperature`, `cor12` | One row per experiment        |
| `GeneFitness`   | `orgId`, `locusId`, `expName`, `fit`, `t`                                             | One row per gene × experiment |
| `Organism`      | `orgId`, organism metadata                                                            |                               |
| `GeneDomain`    | domain annotations                                                                    |                               |
| `Ortholog`      | ortholog mappings                                                                     |                               |


**Approximate scale (full database):** ~221K genes, ~7,552 experiments, ~27.4M fitness rows, ~48 organisms.

### 1.2 `data/raw/aaseqs`

Raw amino acid sequences (FASTA-like). Gets copied to `proteins.fasta` during processing.

### 1.3 `data/media_composition.xlsx`

Excel workbook with media composition information. The v1 pipeline loader used **sheet index 2** (third sheet) for composition-style data — **this must be re-verified** in Phase 0 (sheet names, column headers, units).

**Coverage is an open question:** Not all experiments may be mappable to this table.

#### 1.3.1 Verification checklist (Phase 0, required)

Treat the workbook as **untrusted until verified**:

- List **all sheet names** and row counts; record which sheet is authoritative for medium → components.
- Document **every column** (name, unit, whether concentration or boolean presence).
- Join keys: how does a row in `experiments.media` (and related fields) map to a row in the workbook? Spot-check **≥ N** random media (e.g. 20) against primary literature or Fitness Browser UI.
- **Coverage:** % of benchmark experiments with an unambiguous mapping; list **unmapped** media and decide handling (separate bucket, manual map, or flag).
- **Internal consistency:** duplicate medium names, conflicting compositions, missing components, obvious typos.
- **Output:** a short **media composition data sheet** (markdown table or CSV) stored with the elucidation report — the canonical description of what we believe the Excel file says.

### 1.4 Precomputed ProteomeLM embeddings (optional but recommended starting artifact)

Embedding generation is **slow** (GPU, full proteomes). For a greenfield rebuild, **archive checksum-stable copies** of the tensors you intend to use so the pipeline does not depend on re-running ProteomeLM until gene lists or model choice change.


| Artifact (v1 naming)                          | Role                                                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `data/<subset>/ProtLM_embeddings_layer8/*.pt` | Per-organism tensors: `embeddings` (N × D), `group_labels` (list of `gene_key` strings). Used for **regression** and any model head that reads sequence context only (D = 1152 for ProteomeLM-L layer 8). |


**Do not use** archived `ProtLM_embeddings_with_labels_layer8` (or similar) for the restart: those files embed an integer `y` from **v1 essentiality rules** (`fit < -1`, 0.1/0.8 fractions, etc.), i.e. **arbitrary thresholds**. **Classification / multi-task** experiments must build labels from the **canonical `fit` table** using the **tail/quantile policy** in §2.3 and §9.2 (train-only fitting where required).

**Optional second tensor bundle:** If you compare **two hidden layers** from ProteomeLM (e.g. layer 8 vs last layer), that is a **separate** directory + manifest entry — not “labeled” `.pt` files.

**Manifest:** For each embedding bundle, record subset (`processed` / benchmark), file count, embedding dim, **layer index**, and SHA256 (or equivalent) so “same embeddings” is reproducible without relying on this repo’s old folder layout.

### 1.5 Archived per-organism FASTAs (optional)

If you already have **`organism_fastas/{orgId}.fasta`** (or equivalent) from v1, **keep them** as a checksum-documented archive (e.g. tarball or copy under `data/archive/fastas/`). They are useful for **re-embedding** or ad-hoc analysis.

**Restart pipeline policy:** **Do not** run **MMseqs2** or maintain **`mmseqs_splits.csv`** in the new pipeline. v1 used gene-cluster splits for regression; the restart uses **organism-level** splits only (§7). If homology-based splits are ever needed again, regenerate them **outside** the default workflow from these FASTAs.

---

## 2. Key concepts and definitions

### 2.1 `gene_key`

Unique gene identifier: `{orgId}:{locusId}` (e.g. `Keio:b0001`). Used throughout as the join key between fitness data and embeddings.

### 2.2 `t` vs `cor12`, Huber loss, and how they fit together

These three ideas answer **different** questions. They are **complementary**, not interchangeable.

#### What `t` is (per gene × experiment row)

For each row, the database gives a fitness estimate `fit` and a **t-statistic** `t` for that estimate. Intuitively, **t** measures how strongly the data support that this gene’s `fit` is different from zero (or from the null used in the original analysis — operationally: “how confident are we in *this gene’s* number in *this experiment*?”). Large absolute *t* → the estimated `fit` is more statistically stable for that row. Small absolute *t* → the measurement is compatible with noise / low insertion counts / weak signal.

- **Granularity:** **Row-level** (gene × experiment).
- **Use:** Down-weight uncertain **rows** in the loss, or stratify plots of `fit` by t decile.

#### What `cor12` is (per experiment)

`cor12` is a **replicate correlation for the whole experiment** (how well two halves of the library agree). It measures **experiment-level quality**: growth, library complexity, technical issues — not “this one gene is noisy.”

- **Granularity:** **Experiment-level** (same value for every gene in that `expName`).
- **Use:** Down-weight **all rows** from low-cor12 experiments, or filter experiments for a “strict” training slice.

#### How they differ (one sentence each)


| Quantity | Asks | Level |
| -------- | ---- | ----- |
| **`t`** | “Is *this gene’s* `fit` in *this experiment* well supported?” | Row |
| **`cor12`** | “Was *this experiment* overall reproducible?” | Experiment |


A gene can have high t in a mediocre experiment (strong local signal) or the opposite; both dimensions are useful.

#### What Huber loss is (per prediction error)

Training minimizes a **loss** between predicted and true `fit`. **Squared error (MSE)** punishes large errors **quadratically** — a few wild outliers can **dominate** the gradient. **Huber loss** behaves like **MSE for small errors** and like **absolute error (L1) for large errors** above a threshold δ. Large outliers still matter, but they **do not explode** the gradient the way MSE does.

- **Granularity:** **Model / loss** (depends on residual y - \hat{y}), not on the database column `t`.
- **Use:** Robustness to occasional **label outliers** or heavy tails in `fit`, independent of whether you also weight rows by t or cor12.

#### How they combine in one training recipe (conceptual stack)

1. **Row weight** w_i: combine experiment quality and row confidence, e.g. w_i = h(\text{cor12}_{e(i)}) \cdot g(|t_i|) with h,g bounded in (0, 1]. Multiply **each row’s loss** by w_i (or use weighted sums in the DataLoader).
2. **Loss shape:** Use **Huber** (or plain MSE) on the **weighted** residual. Huber addresses **outliers in `fit` or weird predictions**; weights address **which measurements we trust**.

So: **cor12** and **t** say “how much should this *measurement* count?”; **Huber** says “how should we treat *large prediction errors*?”

---

### 2.3 Labels for **classification / multi-task** (start simple)

**Scope:** Only when an experiment has a **classification head** (e.g. multi-task). **Regression-only** runs ignore this.

**Do not use** v1 `ProtLM_embeddings_with_labels_*` (arbitrary thresholds; §1.4).

**Default rule:** Define “deleterious / essential-like” for a row using a **left-tail cutoff on `fit`** justified from the **empirical distribution** (histograms are sharply peaked at 0 with a long negative tail — e.g. `figures/*/06_histogram_fitness_scores.png`). Concretely: fix a **quantile** q on **train** (e.g. `fit` ≤ empirical q-quantile of `fit` on training rows, optionally **per organism**). **Document q** and show **brief sensitivity** (e.g. one or two alternate quantiles, or a small figure: fraction labelled vs q). Any q used to **supervise** val/test must be computed on **train only**.

**Later (only if multitask looks worthwhile):** Richer gene-level classes, mixture/elbow rules, external gold standards, etc. — out of scope for the first multitask experiments; add to the design log if you pursue them.

**Gaussian / mean ± kσ threshold?** The marginal `fit` histogram is **not** close to a single Gaussian (spike near 0, asymmetric tails), so “fit Normal, flag below μ − kσ” is **easy to mis-specify**: μ and σ are dominated by the central mass, and k is another knob. It can still be tried as an **ablation** (train-only μ, σ, flag `fit < μ − kσ`, sweep k) and **compared** to the quantile rule — but **quantile is the default** because it matches “unusually deep in the left tail” without assuming a shape. (Robust μ/σ, e.g. on a trimmed central band, is a middle ground if you want a parametric story.)

### 2.4 ProteomeLM embeddings

**ProteomeLM** is a protein language model built on ESM-C 600M. It produces **whole-proteome contextual** embeddings — each gene's embedding depends not just on its own sequence but on the full proteome of its organism.

- **Model (v1):** `Bitbol-Lab/ProteomeLM-L`
- **Layer:** 8 (hidden state extraction)
- **Dimension:** 1152 per gene
- **Format:** One `.pt` file per organism, containing `embeddings` (N × 1152) and `group_labels` (list of `gene_key` strings)

**Why "whole-proteome contextual" matters for splits:** Under **organism-level** holdout, the entire proteome (and all its embeddings) is excluded from train — a clean match to “generalise to a **new species**.” (v1 used **gene-cluster (mmseqs)** splits; those are **out of scope** for the restart pipeline — §1.5, §7.)

**Constraint: No fine-tuning** unless forced later. Frozen embeddings only.

### 2.5 Condition encoding (v1): base medium **and** stressors

The regression dataset builder uses **four separate categorical fields**, each with its **own vocabulary and its own contiguous block** in the one-hot vector:


| Field                                       | Typical role                                                                                             |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `media`                                     | Base medium (e.g. LB, M9)                                                                                |
| `condition_1`, `condition_2`, `condition_3` | Additional stressors / supplements / drug names (often empty); **each column has its own set of levels** |


Implementation: `CONDITION_COLS = ("media", "condition_1", "condition_2", "condition_3")` — for each row, exactly **one** index is active **per** field (multinomial-style block, not multi-hot within a field). Total condition width = sum of four vocab sizes (~530 for full `processed` in v1).

**Concentrations / units:** `concentration_1..3`, `units_1..3`, `temperature`, etc. exist on `Experiment` but were **not** part of the default regression one-hot in v1 — adding them as explicit features (bucketed continuous or categorical) is a planned upgrade.

**Problem — "ecosystem silos":** Different organism clades tend to use **non-overlapping media**. A model can learn to predict fitness from **media ID** as a proxy for organism. That motivated a **connected-media benchmark** (§12).

**Problem — chemistry ignored for the base medium:** One-hot on `media` string does not encode shared nutrients; `media_composition.xlsx` is meant to address that (§8).

---

## 3. Prior experiments: roadmap, synopsis, and results

This section is the **experiment memory** for the restart: what was tried, on what harness, and what we concluded. **Numerical scores are not committed in git** (`results.tsv` is gitignored); fill in from your local logs or notebooks when publishing a comparison table.

### 3.1 Tracks (two different tasks — do not merge numbers blindly)


| Track                            | Input                         | Target               | Split style (v1)                  | Typical metrics                                        |
| -------------------------------- | ----------------------------- | -------------------- | --------------------------------- | ------------------------------------------------------ |
| **Gene-level classification**    | ProteomeLM embedding only     | **Train tail-quantile on `fit`** (not `with_labels`) | **Organism-level** holdout        | Accuracy, per-class AUPRC                              |
| **Gene × experiment regression** | Embedding ∥ condition one-hot | `fit` | **Organism-level** holdout (restart); v1 harness used **mmseqs** — not repeated | Val RMSE, mean within-gene Spearman (+ `n_genes_used`) |


Regression is **strictly harder** and **not comparable** to classification numbers on the same scale.

**Historical vs restart (avoid mixing claims):** Ballpark metrics in §3.3.3 used **MVP + mmseqs val** (v1 only). The restart uses **benchmark + organism-level splits only** — **no mmseqs in the pipeline** (§1.5, §7). Old numbers are **calibration only** until recomputed under the new protocol.

### 3.2 Classification track (gene-level, v1)

- **Model:** MLP on frozen embedding; class-weighted cross-entropy; early stopping on rare-class AUPRC.
- **Labels (v1):** Derived from **arbitrary** `fit`/fraction thresholds; also stored in **`ProtLM_embeddings_with_labels_*`** — **discard for restart** (§1.4, §2.3).
- **Reported ballpark (internal notes):** test accuracy ~72–75%; **essential** AUPRC ~0.41–0.44; **non_essential** AUPRC ~0.91–0.95 (imbalanced classes). Numbers are **not** comparable to future tail/quantile-label runs until recomputed.
- **Lesson:** Strong baseline for “is this gene usually essential?” **without** condition; does not solve conditional fitness prediction.

### 3.3 Regression harness (`autoresearch_regression/`)

- **Task:** Predict `fit` from `[1152-d ProteomeLM ∥ condition one-hot]`.
- **Official metrics:** `val_rmse`, `mean_within_gene_spearman`, `n_genes_used_for_spearman`, plus counts of single-row genes and NaN Spearman (see `src/model/regression_metrics.py`).

#### 3.3.1 Phase 1 experiments (archived under `autoresearch_regression/archive/`)


| ID    | Idea                                   |
| ----- | -------------------------------------- |
| exp01 | MSE + 2-layer MLP — reference          |
| exp02 | Wider MLP + LayerNorm + higher dropout |
| exp03 | LayerNorm + GELU                       |
| exp04 | BatchNorm variant                      |
| exp05 | Deeper 4-layer MLP                     |
| exp06 | Huber loss (outliers)                  |
| exp07 | Residual: gene mean + condition offset |
| exp08 | MSE + listwise ranking auxiliary       |
| exp09 | MSE + pairwise ranking loss            |
| exp10 | LR / WD sweep variant                  |


**Synopsis:** Architecture and loss tweaks moved **val RMSE** modestly; **within-gene Spearman** stayed **low** — suggesting the bottleneck is **signal / encoding / split**, not only depth or MSE vs Huber.

#### 3.3.2 Phase 2 experiments (active modules `exp21`–`exp31`)


| ID    | Idea                                                                             |
| ----- | -------------------------------------------------------------------------------- |
| exp21 | Pairwise margin + MSE, standard batching (control for exp22–23)                  |
| exp22 | Pairwise + **gene-packed** batches (more within-gene pairs per step)             |
| exp23 | Pairwise gene-packed + normalisation tweak                                       |
| exp24 | **Listwise** ranking + gene-packed                                               |
| exp25 | **Residual** trunk + listwise                                                    |
| exp26 | Residual + pairwise                                                              |
| exp27 | Residual + **dual** ranking terms                                                |
| exp28 | Residual + tuned hyperparameters                                                 |
| exp29 | **Huber** + residual                                                             |
| exp30 | **Wide offset** residual variant                                                 |
| exp31 | Residual with **frozen gene head** early in training (stabilise offset learning) |


**Synopsis:** Gene-packed and ranking losses target **within-gene condition ordering**; residual structures target **gene-level baseline + condition-specific residual**. Gains over exp01-style MLP were **incremental** on Spearman in exploratory runs; nothing approached “solved” conditional ranking.

#### 3.3.3 Ballpark numbers (from `autoresearch_regression/testing.ipynb` — recompute after rebuild)

- **Subset / split:** As in that notebook (typically MVP + **mmseqs val** — v1 only; restart replaces with organism-level).
- **Global-mean null RMSE → val:** ~**0.80** (val `fit` std ~0.80; with **held-out genes** (mmseqs or organism), per-gene train mean **collapses** to global mean for val genes).
- **Trained MLP (representative runs):** val RMSE ~**0.718–0.728** vs null ~0.80.
- **Within-gene Spearman:** ~**0.035–0.048** (very low; always report **`n_genes_used_for_spearman`**).

**Action item:** When restarting, re-run null + best baseline on the **new** benchmark and splits and **replace** this subsection with a small results table (protocol → null RMSE → model RMSE → Spearman).

### 3.4 Critical insights (synthesis)

1. **Gene-mean baseline collapses to global mean** under **organism-level** holdout (and under v1 gene-cluster splits) for val genes — so **condition information** is what models must learn; memorising gene averages is not available.
2. **Within-gene Spearman** is the right readout for “ranks conditions for the same gene”; **RMSE** is a global scalar that can hide poor ranking.
3. **Media / organism shortcuts** are structurally possible with naive one-hot — benchmark design and chemistry-aware features matter.
4. **Ranking / residual / gene-packed** changes are **second-order** compared to **split choice, condition encoding, and data weighting (cor12, row-level *t*)**.
5. **Engineering:** `drop_last=True` on **val** loader breaks evaluation; prefer `NUM_WORKERS=0` on flaky NFS; always **verify** embedding–fitness row alignment after joins.

### 3.5 Restart — **experiment 1** (first modeling run after Phase 0)

**Goal:** Decide empirically whether **keeping noisy rows with smooth weights** helps more than it hurts, before layering fancier losses or architectures.

**Setup (fixed across arms):** One **baseline architecture** with an explicit **gene pathway** and **condition pathway** (e.g. residual / two-tower style: gene embedding → gene-level baseline; condition features → offset; concat or sum — match whatever you adopt as “baseline” in code). Same **split protocol**, **hyperparameters**, **seeds** (or same seed list) wherever possible.


| Arm                        | Data                                                                                                                                    | Loss / weights                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **A — weighted full data** | All rows in benchmark (or canonical train mask) | Sample weights from **cor12** and **absolute *t*** (documented functions); primary loss **Huber** or MSE (pick one and hold fixed). |
| **B — strict slice** | Same split, but **drop rows** failing hard gates (e.g. cor12 ≥ 0.4 and absolute *t* ≥ 2 — tune gates from Phase 0 distributions) | Unweighted or unit weight; same loss as A. |


**Report:** val **RMSE**, **within-gene Spearman** (+ `n_genes_used`), and optional **multi-task head** metrics if present. Compare **A vs B** and **both vs null baselines** (§7.4).

**Interpretation:** If **A ≥ B** on held-out metrics, noisy rows (down-weighted) add usable signal. If **B ≫ A**, the full table is dominated by noise or shortcuts and strict training data (or stronger weighting / Huber δ tuning) is warranted. This experiment is **not** optional for the restart narrative — it grounds the weighting design.

---

## 4. Non-goals and constraints


| Decision                                            | Rationale                                                                                            |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **No ProteomeLM fine-tuning** (unless forced later) | Frozen embeddings only; isolates PLM contribution.                                                   |
| **Benchmark, not "MVP"**                            | Quality-filtered subset for fair comparison, not a product.                                          |
| **One canonical table, not tiers** | One long table + weight / flag columns; cor12 and row-level *t*-based weighting instead of parallel dataset tiers. |


---

## 5. Phase 0 — Data elucidation (do this first)

Before rebuilding models, answer **what the signal even is**. Outputs guide splits, targets, encoding, and essentiality rules.

### 5.1 Variance decomposition (priority)

**Goal:** Estimate how much of `fit` variance is attributable to **gene**, **condition** (or media profile), and **gene × condition interaction** — ANOVA-style, mixed model, or similar on an appropriate subset.

**Why:** Sets an empirical ceiling on learnable condition-specific signal.

### 5.2 Investigation checklist

- Distribution of `fit` and `t` (global, by organism, by experiment). **Plot:** histograms, ECDFs, violin by organism (top-N by count).
- **cor12** distribution; scatter / hexbin vs residual σ or fit.
- Sparsity: density of gene × condition matrix **per organism**; CDF of “# conditions per gene.”
- **Organism balance:** bar charts of rows / genes / experiments per `orgId`.
- **If multitask is planned:** `fit` histogram / ECDF; pick and document a **train-derived tail quantile** for the binary cutoff; **short sensitivity** (e.g. 2–3 values of q). Optional: compare to **μ − kσ** on train as a sanity ablation (see §2.3).
- **Media composition:** full verification (§1.3.1); bipartite **organism–media** graph (plot); table of media degree (# organisms).
- **Connected-media audit:** For the proposed benchmark filter, verify **empirically** that retained media meet the connectivity rule (§12), not only name prefixes.
- Duplicates / conflicts for the same gene × experiment.
- Embedding sanity: norm distributions, missing keys.

### 5.3 Visualisation requirement

Phase 0 deliverables should be **graph-first**: every major checklist item should have at least one **published figure** (saved PNG/PDF + source notebook cell) in the elucidation report. Tables alone are insufficient for threshold and skew decisions.

### 5.4 Deliverables

- **Data elucidation report** (notebook + figures + short prose).
- **Design decisions log** (“we chose X because finding Y”).
- **Data manifest** checksums for raw files, optional embedding bundles, and each canonical Parquet version.

---

## 6. More data and noise-aware signal

### 6.1 Expanding the dataset

Inventory → normalise schema → provenance table → append to canonical long table. Gate on Phase 0 (volume vs quality bottleneck).

### 6.2 cor12 and t as noise-aware signal

See **§2.2** for definitions and the conceptual stack (weights + Huber). **Policy:** implement **weighted training** as the default path; use a **strict row filter** as the **B arm** of **§3.5 experiment 1**, not as an afterthought ablation.

---

## 7. Split design: organism-level × conditions

**Restart pipeline:** Splits are **organism-level** only (leave-one-organism-out, leave-k-organisms-out, or a fixed val/test organism set). **MMseqs2 / gene-cluster CSVs are not produced or consumed** in the default workflow. Per-organism **FASTAs** may be **archived** for re-embedding or rare ad-hoc work (§1.5).

### 7.1 Organism-level splits (only split on the gene / species axis)


| Split | Held-out unit | Claim |
| ----- | ------------- | ----- |
| **Organism-level** | All rows for one or more `orgId` | Generalise to a **new species** |


Formats: **LOOO**, **leave-k-organisms-out**, or a single **val organism / test organism** assignment — document which in the split manifest.

### 7.2 Axis B — Conditions (second axis)


| Split  | Claim                                                     |
| ------ | --------------------------------------------------------- |
| **B0** | Conditions (as encoded) appear in train for some genes    |
| **B1** | Some media/stressors **never** in train — new environment |


### 7.3 Evaluation grid

- **Primary:** Organism held out + **B0** (conditions not held out).
- **Secondary:** Organism held out + **B1** (condition held out) where data density allows.

### 7.3.1 Historical note (v1 only)

Gene-cluster (**mmseqs**) splits were used in the old regression harness for “new gene, seen conditions.” They are **not** part of the restart; do not wire `mmseqs_splits.csv` into the new `splits/` stage.

### 7.4 Baselines under each protocol


| Baseline                                   | What it tests                |
| ------------------------------------------ | ---------------------------- |
| Global train mean → val                    | Floor                        |
| Per-experiment mean (train) → val          | Condition identity           |
| Per-organism mean (when val organism appears in train — rare under LOOO) | Species effect |
| Embedding nearest-neighbour (train) → val  | Sequence similarity transfer |


**How often to compute:** Once **per evaluation protocol** (each distinct train/val/test index definition on the canonical table). **Not** once per neural experiment: every model run on the same protocol **reuses** the same baseline numbers (recompute only if the canonical table, split masks, or benchmark definition changes).

**Critical lesson:** Under **organism-level** val, per-gene train mean for val organisms’ genes is undefined → with global fallback it equals **global mean**.

### 7.5 Metric reporting under organism split

RMSE/MAE; **Δ vs protocol-specific null RMSE**; within-gene Spearman + **`n_genes_used_for_spearman`**; stratify by minimum condition count in val; per-organism breakdown for LOOO.

---

## 8. Condition encoding: chemistry + stressors

### 8.1 v1 recap

Four **separate** one-hot blocks: `media`, `condition_1`, `condition_2`, `condition_3` — stressor fields already have **their own dimensions** (separate vocab per column). Concentrations/units/temperature were not in the default one-hot.

### 8.2 Planned upgrades

- **Chemistry:** Nutrient multi-hot or concentration vector from verified `media_composition.xlsx` (§1.3.1).
- **Stressors:** Keep categorical one-hot per `condition_*` **or** add **bucketed concentrations** / **units** as extra features when Phase 0 shows they carry signal.

---

## 9. Targets and multi-task learning

### 9.1 Regression

Primary target: **`fit`** (transforms only with Phase 0 justification and leakage controls). **No** classification column required.

### 9.2 Multi-task (only when the experiment includes a classification head)

Joint prediction of `fit` + a binary (or simple) label from **`fit`** via **§2.3** (train tail-quantile; no `with_labels` artifacts). Combine regression and classification losses (head weight is a hyperparameter). **Regression-only** experiments skip this section.

---

## 10. ProteomeLM usage (frozen)

If starting from §1.4 archives, **skip regeneration** until gene set or layer choice changes. Otherwise: per-organism FASTA → ProteomeLM-L → extract layer 8 → `.pt` + manifest.

**Fusion:** Concat baseline; residual / FiLM / late fusion as controlled experiments on the **same** canonical inputs.

---

## 11. Code and repository structure (modular, research-first)

**Intent:** Stages are **separate packages or top-level dirs** with clear inputs/outputs (not one undifferentiated `src/` + misc `scripts/`). Names are **illustrative** — pick one tree when scaffolding and record it in the design log.

### 11.1 Suggested stage layout

| Stage | Responsibility | Reads | Writes |
| ----- | -------------- | ----- | ------ |
| **`data_processing/`** | Raw → canonical tables, joins, weights, benchmark masks | `data/raw/*`, `media_composition.xlsx` | Versioned Parquet (or similar) + **manifest** (checksums, row counts) |
| **`data_analysis/`** | Phase 0 only: EDA, variance decomposition, figures | Canonical read-only | Notebooks, `figures/`, elucidation report, design log |
| **`embeddings/`** (optional stage) | FASTA → frozen `.pt` | Canonical gene lists | `ProtLM_embeddings_layer*/` + manifest |
| **`splits/`** | Train/val/test index files per **protocol** | Canonical + benchmark definition | `splits/<protocol>/*.csv` or `.parquet` |
| **`modeling/`** | Dataset classes, models, train loop, experiment configs | Canonical + embeddings + split indices | Checkpoints, run logs |
| **`evaluation/`** | Aggregate metrics, tables, plots vs nulls | Run outputs | Tables for paper / benchmark dashboard |

**Shared utilities** (`shared/`, `lib/`, or minimal `src/`): paths, config loading, logging — **no** business logic that belongs in a single stage.

### 11.2 Where files live vs v1

- **Immutable inputs** stay under **`data/raw/`** (and checked-in or documented **media** Excel).
- **Regenerated artifacts** should not silently overwrite without a version bump: prefer **`data/derived/<version>/`** or **`artifacts/<build_id>/`** plus manifest, instead of only `data/processed/` with unclear lineage.
- **Migrating this repo:** existing `src/pipeline/`, `scripts/`, `autoresearch_regression/` map onto the stages above incrementally; the plan does not require deleting v1 until the new paths are wired.

### 11.3 Engineering rules (all stages)

Versioned manifests, **saved run configs** (YAML/JSON per run), **documented seeds** (single seed + optional multi-seed list). No `drop_last` on **val** loaders.

### 11.4 Run identity (`run_id`)

Every training or evaluation run should be **uniquely addressable** for debugging and papers.

**`run_id`** (UUID or `YYYYMMDD_HHMMSS_<short_git_sha>_<slug>`) ties together in one folder or sidecar JSON:

| Field | Purpose |
| ----- | -------- |
| **`run_id`** | Primary key for the run |
| **Git commit** | `git rev-parse HEAD` (dirty flag if working tree changed) |
| **Config** | Full resolved YAML/JSON (hyperparameters, paths, benchmark version) |
| **Data manifest hash** | Checksum of canonical Parquet / benchmark build the run used |
| **Split protocol id** | e.g. `LOOO_v1`, `organisms_heldout_Keio` |
| **Seeds** | Python / PyTorch / NumPy (and CUDA if applicable) |
| **Environment** | Optional: `pip freeze` path or container image digest |

Evaluation rows in the benchmark table (§13 M6) should include **`run_id`** and **split protocol id** so numbers are reproducible without guesswork.

### 11.5 Automated tests (low lift, high value)

**Goal:** Catch join bugs, off-by-one vocab errors, and split mistakes **before** long GPU jobs.

**`data_processing/` / canonical tables**

- **Join integrity:** After `fitness` ⋈ `experiments`, **no null** on required keys (`orgId`, `expName`, `media`, …); row-count change matches expectation.
- **Key uniqueness:** No duplicate (`orgId`, `locusId`, `expName`) rows unless the schema explicitly allows them.
- **Referential sanity:** Every `(orgId, expName)` in `fitness` exists in `experiments`; `orgId` in tables exists in `organisms` (or document exceptions).
- **Benchmark mask:** Applying the benchmark filter twice yields the **same** row count; optional snapshot hash of filtered `expName` set.
- **Ranges:** `cor12` in \[−1, 1\] (or documented DB range); condition indices in vocab bounds after encoding.

**`splits/`**

- **Partition:** Train / val / test **disjoint** on row id or (`gene_key`, `expName`); **union** covers the intended benchmark subset (or explicit “held-out unlabelled” policy).
- **Organism rule:** Under organism-level protocol, **no val `orgId`** appears in train (for that fold).

**`embeddings/` ↔ modeling**

- **Coverage:** Every `gene_key` required by the training split appears in the embedding store; embedding **dim** constant across `.pt` files; **no NaN** in a sample of tensors.
- **Order / alignment:** For a fixed list of rows, `(gene_key → vector)` matches the order the `Dataset` produces (regression v1 had real bugs here).

**`evaluation/` / metrics**

- **Golden values:** Hand-check **RMSE** / **Spearman** on a tiny toy tensor or CSV vs **numpy** / **scipy** (or pandas) reference.
- **Edge cases:** Spearman with **one** row per gene → undefined; metric code returns NaN or skips — assert documented behaviour.

**`modeling/`**

- **Config load:** Required keys present; unknown keys rejected or logged (avoid silent typos).
- **Loader contract:** Val `DataLoader` has `drop_last=False` (or batch size 1 for val) so `len(predictions) == len(val_df)` (§3.4).

Run these in **CI** on a **tiny fixture** (subset Parquet + fake embeddings) so tests stay fast; optional nightly job on full benchmark manifest.

---

## 12. Benchmark definition (“connected media”) — make it **robust**

**Intent:** Reduce **organism ↔ medium shortcutting** by restricting to media that appear for **multiple** organisms (exact count **TBD** in Phase 0, e.g. ≥ 2 or ≥ 3 organisms).

### 12.1 v1 heuristic (starting point only)


| Rule                  | Value                   |
| --------------------- | ----------------------- |
| `media` **prefix** in | `LB`, `RCH2`, `M9`      |
| `cor12`               | ≥ 0.2                   |
| Exclude `expGroup`    | `plant`                 |
| Exclude `media`       | `Potato Dextrose Broth` |


**Approximate v1 scale:** ~138K genes, ~2,164 experiments, ~8.7M fitness rows, ~27 organisms.

### 12.2 Robust definition (required after Phase 0)

1. **Construct the organism–medium bipartite graph** on the rows passing quality filters (cor12, etc.).
2. **Define “connected medium”** as a medium node with degree ≥ **K** organisms (K chosen from the graph — document K).
3. **Filter experiments** to those media; **verify** the prefix rule did not accidentally **include** a rare medium or **exclude** a truly shared one.
4. **Report:** table of included media, organism count per medium, and **before/after row counts**.
5. **Stability:** if K or prefixes change, bump benchmark **version** in the manifest.

---

## 13. Milestones


| #    | Milestone               | Deliverable                                                               |
| ---- | ----------------------- | ------------------------------------------------------------------------- |
| M0   | Lock inputs             | Checksums: raw DB, aaseqs, Excel, optional embedding bundles              |
| M1   | Phase 0                 | Elucidation report + figures + media verification + connected-media graph |
| M2   | Canonical tables        | Long fitness + experiment + weights + composition join                    |
| M3   | Splits                  | **Organism-level** index files per protocol (no mmseqs in default pipeline) |
| M3.5 | Null baselines          | **Per protocol**, once — stored table for all models                      |
| M4   | Embeddings              | Use §1.4 or regenerate + manifest                                         |
| M5   | **Experiment 1** (§3.5) | Weighted-full vs strict-slice, same gene+condition baseline               |
| M5b  | Baseline + follow-ons   | Other architectures / losses after noise design is clear                  |
| M6   | Benchmark table         | Nulls + experiment 1 + later models; tag each row with **evaluation protocol** (split + benchmark version) |


---

## 14. Dependencies and environment

Python ≥ 3.10, PyTorch, pandas/pyarrow, scipy, openpyxl; ProteomeLM + GPU for **optional** embedding generation; **pytest** (or stdlib `unittest`) for automated checks. **MMseqs2 not required** for the restart pipeline (§1.5, §7).

---

## 15. Open questions

- Functional form for **cor12** and **t** weights (and floors/caps).
- **K** organisms for “connected medium” and whether prefix filter is **redundant** after graph filter.
- Chemical vocabulary size and handling of **unmapped** media.
- Whether to add **temperature / concentration** buckets to the default feature vector.
- Whether a **second ProteomeLM hidden layer** (separate `.pt` directory) improves over layer 8 for regression or multitask.
- LOOO feasibility per organism (minimum rows for stable metrics).
- Licensing for additional Tn-Seq sources.

---

## 16. What success looks like

Success is **always defined relative to the null baseline on the same val set and evaluation protocol**:

1. Phase 0 completes variance decomposition + **visual** elucidation + **verified** media composition + **graph-based** connected-media benchmark.
2. **Regression:** val RMSE **meaningfully below** **null RMSE** (same split, same rows) — report **absolute Δ** and **relative %** (e.g. “10% reduction vs global mean”). **Do not** hard-code a numeric ceiling such as 0.80; null RMSE **changes** with subset and split.
3. **Within-gene Spearman** is clearly positive (e.g. ρ above zero with CI or permutation check) with **`n_genes_used_for_spearman`**; optionally compare to a **permutation** or **shuffle** baseline.
4. Reproducible benchmark table: manifest + code + this document.

**Stretch:** B1 condition-held-out; multi-task helps or is neutral; chemistry features beat string one-hot on matched protocols.

---

## 17. Robustness — extra checks (easy to skip)

- **`run_id` bundle:** Every published number traceable via §11.4.
- **Automated tests:** Keep §11.5 green on CI with a small fixture.
- **Licence / attribution:** Fitness Browser export (`feba.db`) and any added Tn-Seq sources — document what redistribution and publication allow.
- **Reproducibility:** Pin **Python + PyTorch + CUDA** (or container); store **full `pip freeze`** (or lockfile) next to the manifest for published results.
- **Leakage audit:** Tail-quantile **q** for classification (§2.3) must be fit on **train** only; document whether Phase 0 exploratory plots ever influenced a numeric threshold used on val/test.
- **Class imbalance:** Multitask binary head will be **very imbalanced** (rare “tail” rows); report **AUPRC** / balanced metrics, not accuracy alone.
- **Embedding coverage:** Assert **every** training row’s `gene_key` exists in the embedding store; log **dropped row count** if not.
- **Sign / convention:** Confirm `fit` sign convention (negative = defect) matches across **all** merged tables and loss definitions.
- **Organism-level minimum N:** Before LOOO, define **minimum val rows (or genes)** per organism; exclude or merge sparse organisms so metrics are not noise.
- **Negative controls:** Optional **label shuffle** or **gene↔embedding shuffle** on val to show the model beats a broken baseline.

---

*Last updated: 2026-04-04. Living document — revise as we learn.*