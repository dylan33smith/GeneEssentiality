# Regression vs essentiality: why fitness prediction is harder

## Data alignment (verified)

The pipeline has been verified end-to-end:

- **Script:** `scripts/verify_regression_data_alignment.py`
- **Checks:** (1) Sampled rows from `regression_dataset.parquet` match `fitness.parquet` on (orgId, locusId, expName) and fit. (2) Dataset `__getitem__(i)` returns the same gene (via embedding index) and fit as parquet row `i`. (3) Embedding store indexing matches the order of `group_labels` / `embeddings` in each `.pt` file.

**Conclusion:** Fitness is not associated with the wrong gene/experiment. The dataloader and model receive correctly aligned (embedding, one-hot, fit) triples.

---

## Why essentiality looked good but regression doesn’t

| Aspect | Essentiality (ProteomeLM-Ess) | Fitness regression |
|--------|-------------------------------|---------------------|
| **Target** | One label per **gene** (always / conditional / non_essential), derived by aggregating over experiments (e.g. frac_essential_confident). | One **continuous** value per **(gene, experiment)**. |
| **Task** | “What is this gene’s typical essentiality?” → Largely determined by **gene identity** (embedding). | “What is this gene’s fitness in **this** condition?” → Depends on **gene–condition interaction**. |
| **Splits** | Often by **organism** (train on some species, test on others) or by gene. | By **gene** (mmseqs_splits) → test set has **unseen genes**, so the model must generalize to new genes and new conditions. |
| **Signal** | Essentiality is an aggregate; noise averages out. | Each fitness value is a single, noisy measurement (log2 fitness + t-stat). |
| **Input** | Embedding only (gene). | Embedding + condition one-hot (gene + experiment). Model must learn how condition modulates gene effect. |

So:

1. **Essentiality** is mostly “predict gene property from sequence/embedding.” The embedding is very informative; condition is secondary (it’s already baked into the aggregate label).
2. **Regression** is “predict gene–condition outcome.” The model must use both embedding and condition and learn their interaction; the target is noisier and the generalization setting (unseen genes) is harder.

That’s why regression can look “unimpressive” even when the pipeline is correct.

---

## What to try next

- **Baselines:** Predict mean fitness per gene (ignore condition), or mean fitness per experiment (ignore gene). Compare your model’s val MSE to these; if it’s not clearly better, the model isn’t using gene–condition interaction.
- **Noise / scale:** Check distribution of `fit` (e.g. std, outliers). Consider standardizing the target or using a robust loss (e.g. Huber) if there are heavy tails.
- **Generalization:** Try a **condition split** (train on a subset of experiments, validate on held-out experiments) to see if the model is learning condition effects; compare to the current **gene split** (unseen genes).
- **Capacity / regularization:** Try a larger model (deeper/wider) with stronger regularization (dropout, weight decay) and early stopping on val loss.
- **Learning rate:** Try a smaller LR (e.g. 3e-4) and/or a scheduler (e.g. ReduceLROnPlateau on val MSE).
