# Project Overview

## Title and goal

**Context-aware prediction of bacterial gene essentiality.** The project aims to predict quantitative gene fitness (and essentiality class) from genotype and growth condition: given a gene and an environment (media, stressor, etc.), predict fitness (e.g., log₂ score) and whether the gene is essential, conditionally essential, or non-essential in that context.

## Data source

All data come from the **Fitness Browser** database: a public resource aggregating genome-wide gene fitness measurements from **transposon insertion sequencing (Tn-Seq)** across multiple bacterial species and growth conditions. No new wet-lab experiments are performed in this project; the pipeline uses processed exports (Parquet tables, FASTA, condition vocabulary) derived from that database.

## MVP vs full database

- **MVP (minimum viable product):** A or “connected” subset chosen to include organisms with overlap in conditions. Inclusion: organisms that use shared media (**LB**, **RCH2_defined**, or **M9**). Quality filter: experiments with replicate correlation **cor12 ≥ 0.2** only (replicate correlation measures agreement between identical sample measurements). Results in: **27 organisms**, **2,164 experiments**, **138,518 genes**, **8,688,562** fitness records. Used for development and training.
- **Processed (full):** The complete exported database: **48 organisms**, **7,552 experiments**, **221,005 genes**, **27,410,721** fitness records. Used for reference and for building assets (e.g., orthologs) that span all organisms.

## Current implementation status

**Implemented:**

- **Data loaders** (`src/data_io`): Path resolution (project root, `data/`, `processed/`, `mvp/`) and loading of Parquet tables (genes, experiments, fitness, organisms) and media Excel (sheet 2). All access via `src` functions; no raw paths in shared code.
- **Fitness and essentiality** (`src/fitness`): Vectorized aggregation of raw fitness to gene-level statistics, and assignment of **essentiality class** using the 80%/10% rules (confident = |t| ≥ 2, essential in an experiment = fit &lt; −1; gene-level: &gt;80% → always_essential, 10–80% → conditional, &lt;10% → non_essential, no confident exps → no_data).
- **ProteomeLM embedding step** (`src/pipeline/generate_embeddings.py`): Produces proteome-contextualized embeddings (ESM-C + ProteomeLM) from per-organism FASTAs. Run via `scripts/run_pipeline.py`. **Done:** Embeddings in `data/mvp/ProtLM_embedddings/` (one .pt per organism: `embeddings`, `group_labels`). See `notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md`.
- **Y label step** (`src/pipeline/label_embeddings.py`): **Done.** Adds `y` (essentiality_class → 0/1/2/-1) to each .pt; output in `data/mvp/ProtLM_embeddings_with_labels/`. Run via `scripts/run_pipeline.py`.

**Not yet implemented:**

- **MLP baseline (immediate next step):** Load .pt from `data/mvp/ProtLM_embeddings_with_labels/` (embeddings + y); filter to y ≠ -1. Train MLP: embedding → 3-class. Organism-based split. Metrics: AUPRC (always_essential, conditional), accuracy, weighted F1.

These are planned in the project protocol (README).

## Discrepancies resolved

This documentation is aligned with the **current codebase** (`src/`, `scripts/`) and **actual data** (Parquet row counts and vocabulary sizes from the files on disk). Where older notes differed (e.g. experiment counts, vocab sizes, embedding format), the numbers and formats in this doc set reflect the code and current data. Legacy notes are kept in `notes/archive/` for reference only.
