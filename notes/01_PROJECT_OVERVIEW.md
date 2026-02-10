# Project Overview

## Title and goal

**Context-aware prediction of bacterial gene essentiality.** The project aims to predict quantitative gene fitness (and essentiality class) from genotype and growth condition: given a gene and an environment (media, stressor, etc.), predict fitness (e.g., log₂ score) and whether the gene is essential, conditionally essential, or non-essential in that context.

## Data source

All data come from the **Fitness Browser** database: a public resource aggregating genome-wide gene fitness measurements from **transposon insertion sequencing (Tn-Seq)** across multiple bacterial species and growth conditions. No new wet-lab experiments are performed in this project; the pipeline uses processed exports (Parquet tables, FASTA, condition vocabulary) derived from that database.

## MVP vs full database

- **MVP (minimum viable product):** A “terrestrial” or “connected” subset chosen to reduce confounding. Inclusion: organisms that use shared media (**LB**, **RCH2_defined**, or **M9**). Quality filter: experiments with replicate correlation **cor12 ≥ 0.2** only. Result: **27 organisms**, **2,164 experiments**, **138,518 genes**, **8,688,562** fitness records. Used for development and training.
- **Full (processed):** The complete exported database: **48 organisms**, **7,552 experiments**, **221,005 genes**, **27,410,721** fitness records. Used for reference and for building assets (e.g., orthologs) that span all organisms.

## Current implementation status

**Implemented:**

- **Data loaders** (`src/data_io`): Path resolution (project root, `data/`, `processed/`, `mvp/`) and loading of Parquet tables (genes, experiments, fitness, organisms) and media Excel (sheet 2). All access via `src` functions; no raw paths in shared code.
- **Fitness and essentiality** (`src/fitness`): Vectorized aggregation of raw fitness to gene-level statistics, and assignment of **essentiality class** using the 80%/10% rules (confident = |t| ≥ 2, essential in an experiment = fit &lt; −1; gene-level: &gt;80% → always_essential, 10–80% → conditional, &lt;10% → non_essential, no confident exps → no_data).
- **ESM-C embedding script** (`scripts/generate_esmc_embeddings.py`): Standalone script that reads a FASTA, runs the ESM-C model (e.g. `esmc_300m`), and writes a `.pt` file in the format expected by ProteomeLM (`inputs_embeds`, `group_embeds`, `group_labels`). Does not import ProteomeLM or `src`.

**Not yet implemented:**

- ML model (Transformer/MLP) consuming embeddings and conditions to predict fitness.
- Integration of embeddings (loading `protein_embeddings.pt`) into the main pipeline.
- Sliding window (target gene ±5 neighbors) for operon context.
- Condition encoding using `condition_vocab.json` (multi-hot vectors).
- Train/validation/test splits (e.g. homology-based at 50% identity).

These are planned in the project protocol (README).

## Discrepancies resolved

This documentation is aligned with the **current codebase** (`src/`, `scripts/`) and **actual data** (Parquet row counts and vocabulary sizes from the files on disk). Where older notes differed (e.g. experiment counts, vocab sizes, embedding format), the numbers and formats in this doc set reflect the code and current data. Legacy notes are kept in `notes/archive/` for reference only.
