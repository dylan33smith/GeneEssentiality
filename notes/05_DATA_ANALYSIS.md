# Data Analysis

Evidence-based description of the data lifecycle: what the data is, where it comes from, how it is filtered and transformed, and what is (or will be) used as model inputs. All metrics and schemas below were computed from the actual Parquet tables and optional assets via the project’s loaders (see `scripts/data_discovery.py`).

---

## 1. Executive summary

The project uses **genes**, **experiments**, **fitness**, and **organisms** tables (Fitness Browser–derived Parquet), plus optional **sequences** (FASTA) and **protein embeddings** (`.pt`). The goal is **four-way essentiality classification** (always_essential, conditional, non_essential, no_data) from the 80%/10% rule over confident experiments; optionally **fitness prediction** in future. The pipeline is **context-aware**: genotype + condition → outcome (fitness and essentiality depend on both gene and environment).

---

## 2. Stage I: Processed data (full database)

### Description

These files are the first version of the data available inside the project. They live under `data/processed/` and comprise four Parquet tables: **genes**, **experiments**, **fitness**, **organisms**. They are derived from the **Fitness Browser** database (Tn-Seq gene fitness across bacterial species and conditions). Raw upstream exports from the Fitness Browser are outside the repo; this stage is the full exported database before any MVP filtering.

### Schema

Column names and types below match the actual files and `src/data_io.py` / `notes/02_DATA_DICTIONARY.md`.

**genes.parquet** (one row per gene)

| Column | Type | Description |
|--------|------|-------------|
| orgId | object | Organism identifier (links to organisms.parquet). |
| locusId | object | Gene locus identifier (unique per organism). |
| sysName | object | Systematic name (e.g. b0001). |
| scaffoldId | object | Chromosome/plasmid ID. |
| begin | int64 | Genomic start (bp). |
| end | int64 | Genomic end (bp). |
| gene_length | int64 | Length in bp. |
| type | int64 | Gene type (e.g. CDS). |
| strand | object | + or -. |
| gene | object | Gene symbol (optional). |
| desc | object | Description (optional). |
| GC | float64 | GC content (optional). |
| n_total_experiments | int64 | Number of experiments with a fitness value for this gene. |
| n_confident_experiments | int64 | Number of experiments where \|t\| ≥ 2. |
| frac_not_confident | float64 | 1 − (n_confident / n_total). |
| mean_fit_confident | float64 | Mean fitness over confident experiments only. |
| frac_essential_all | float64 | Fraction of all experiments where fit < −1. |
| frac_essential_confident | float64 | Fraction of confident experiments where fit < −1. |
| essentiality_class | object | Label from 80%/10% rule: always_essential, conditional, non_essential, no_data. |

**experiments.parquet** (one row per experiment)

| Column | Type | Description |
|--------|------|-------------|
| orgId | object | Organism. |
| expName | object | Experiment name (unique with orgId). |
| expDesc | object | Description. |
| expGroup | object | Experiment group label. |
| mutantLibrary | object | Library identifier. |
| media | object | Media name (e.g. LB, M9). |
| mediaStrength | float64 | Strength/category. |
| condition_1 | object | Primary condition (e.g. stressor). |
| concentration_1, units_1 | object | Concentration and units for condition_1. |
| condition_2, concentration_2, units_2 | object | Optional condition 2. |
| condition_3, concentration_3, units_3 | object | Optional condition 3. |
| temperature | object | Growth temperature. |
| pH | object | pH. |
| aerobic | object | Aerobic/anaerobic. |
| vessel | object | Vessel. |
| liquid | object | Liquid. |
| shaking | object | Shaking. |
| cor12 | float64 | Replicate correlation (quality); MVP uses cor12 ≥ 0.2. |
| maxFit | float64 | Optional. |
| nGenerations | float64 | Optional. |

**fitness.parquet** (one row per gene–experiment)

| Column | Type | Description |
|--------|------|-------------|
| orgId | object | Organism. |
| locusId | object | Gene. |
| expName | object | Experiment. |
| fit | float64 | Log₂ fitness; fit < −1 → essential in this experiment. |
| t | float64 | t-statistic; \|t\| ≥ 2 → confident. |

**organisms.parquet** (one row per organism)

| Column | Type | Description |
|--------|------|-------------|
| orgId | object | Organism identifier. |
| division | object | Taxonomic division. |
| genus | object | Genus. |
| species | object | Species. |
| strain | object | Strain. |
| taxonomyId | int64 | NCBI taxonomy ID. |

### Statistics (Stage I)

- **Row counts:** genes 221,005; experiments 7,552; fitness 27,410,721; organisms 48.
- **Unique (gene, organism) in fitness:** 182,447. Unique (orgId, expName) in fitness: 7,552.
- **fit:** min −12.25, max 17.95, mean −0.097.
- **t:** min −43.33, max 113.48, mean −0.18.
- **cor12 (experiments):** min 0.10, max 0.79, mean 0.28. **62.02%** of experiments have cor12 ≥ 0.2.
- **frac_essential_confident (genes, non-null):** min 0, max 1, mean 0.067. **n_confident_experiments:** min 0, max 713, mean 43.2.
- **Nulls:** genes have nulls in frac_not_confident, mean_fit_confident, frac_essential_all, frac_essential_confident (41,059 rows with no confident experiments); experiments have 5,107 nulls in nGenerations; fitness and organisms have no nulls in core columns.

### Snapshot (Stage I)

First three rows (representative).

**genes**

| orgId | locusId | sysName | n_total_experiments | n_confident_experiments | frac_essential_confident | essentiality_class |
|-------|---------|---------|---------------------|-------------------------|--------------------------|--------------------|
| ANA3 | 7022496 | Shewana3_4141 | 107 | 11 | 0.0 | non_essential |
| ANA3 | 7022497 | Shewana3_4142 | 107 | 14 | 0.0 | non_essential |
| ANA3 | 7022498 | Shewana3_4143 | 0 | 0 | — | no_data |

**experiments**

| orgId | expName | media | condition_1 | cor12 |
|-------|---------|-------|------------|-------|
| ANA3 | set1H1 | ShewMM_noNitrogen | L-Asparagine | 0.326 |
| ANA3 | set1H10 | ShewMM_noNitrogen | L-Arginine | 0.337 |
| ANA3 | set1H13 | ShewMM_noCarbon | L-Alanine | 0.369 |

**fitness**

| orgId | locusId | expName | fit | t |
|-------|---------|---------|-----|-----|
| ANA3 | 7022496 | set1H1 | −0.252 | −0.398 |
| ANA3 | 7022496 | set1H10 | −0.110 | −0.198 |
| ANA3 | 7022496 | set1H13 | 0.276 | 0.583 |

**organisms**

| orgId | division | genus | species | strain |
|-------|----------|------|--------|--------|
| ANA3 | Gammaproteobacteria | Shewanella | sp. | ANA-3 |
| BFirm | Betaproteobacteria | Burkholderia | phytofirmans | PsJN |
| Btheta | Bacteroidetes | Bacteroides | thetaiotaomicron | VPI-5482 |

---

## 3. Stage II: MVP data (quality-filtered subset)

### Description

Same schema as Stage I, but restricted to the MVP subset in `data/mvp/`: **terrestrial** (connected) organisms that use shared media (**LB**, **RCH2_defined**, or **M9**), and experiments with **cor12 ≥ 0.2** only. Used for development and training to reduce confounding and media-only shortcuts (see README and `notes/01_PROJECT_OVERVIEW.md`).

### Transformation logic (the “why”)

- **Quality:** Only experiments with replicate correlation cor12 ≥ 0.2 are kept, so the MVP has 100% of experiments above that threshold (processed has 62%).
- **Comparability:** Organisms are restricted to those using LB, RCH2_defined, or M9 so that media overlap forces the model to use gene and condition information.
- **Scale:** Fewer organisms (27 vs 48) and experiments (2,164 vs 7,552) for a manageable training set.

Filtering that produces the MVP Parquet files is done **upstream** of the repo (the MVP tables are pre-built). Duplicates and missing values in the pipeline that produced these files are not re-documented here; the genes table has the same optional nulls (e.g. frac_essential_confident for genes with n_confident_experiments = 0) as in Stage I.

### Schema

Same column set as Stage I for genes, experiments, fitness, and organisms. No columns exist only in one stage.

### Statistics (Stage II)

- **Row counts:** genes 138,518; experiments 2,164; fitness 8,688,562; organisms 27.
- **Drops from Stage I:** genes −82,487; experiments −5,388; fitness −18,722,159; organisms −21.
- **Unique (gene, organism) in fitness:** 115,743. Unique (orgId, expName): 2,164.
- **fit:** min −11.77, max 15.07, mean −0.105.
- **t:** min −43.33, max 113.48, mean −0.19.
- **cor12 (experiments):** min 0.20, max 0.62, mean 0.30. **100%** of experiments have cor12 ≥ 0.2 (by construction).
- **frac_essential_confident (genes, non-null):** min 0, max 1, mean 0.066. **n_confident_experiments:** min 0, max 252, mean 22.1.
- **Nulls:** genes have 24,462 nulls in frac_not_confident / mean_fit_confident / frac_essential_all / frac_essential_confident; experiments have 1,290 nulls in nGenerations.

### Snapshot (Stage II)

Same format as Stage I; first three rows of each table (same genes/experiments as in processed for this organism).

**genes**

| orgId | locusId | sysName | n_total_experiments | n_confident_experiments | frac_essential_confident | essentiality_class |
|-------|---------|---------|---------------------|-------------------------|--------------------------|--------------------|
| ANA3 | 7022496 | Shewana3_4141 | 95 | 11 | 0.0 | non_essential |
| ANA3 | 7022497 | Shewana3_4142 | 95 | 14 | 0.0 | non_essential |
| ANA3 | 7022498 | Shewana3_4143 | 0 | 0 | — | no_data |

**experiments / fitness / organisms:** Same structure as Stage I; rows are the MVP subset (fewer experiments and organisms).

---

## 4. Stage III: Model-ready data (labels and optional embeddings)

### Transformation logic (the “why”)

**Gene-level labels** come from the fitness table and the 80%/10% rule implemented in `src/fitness.py`:

1. **Confident experiments:** \|t\| ≥ 2 (see `CONFIDENT_T_THRESHOLD = 2.0`).
2. **Essential in an experiment:** fit < −1 (see `ESSENTIALITY_FIT_THRESHOLD = -1.0`).
3. **Per-gene aggregates:** Over confident experiments only, compute the fraction where fit < −1 (`frac_essential_confident`) and the count of confident experiments (`n_confident`). The function `aggregate_fitness_to_genes(fitness)` returns these and the class; `get_essentiality_class(frac_essential_confident, n_confident)` assigns the label.
4. **Four classes:**  
   - **no_data:** n_confident == 0.  
   - **always_essential:** frac_essential_confident > 0.8.  
   - **conditional:** 0.1 ≤ frac_essential_confident ≤ 0.8.  
   - **non_essential:** frac_essential_confident < 0.1 (and n_confident > 0).

The pre-built **genes.parquet** tables (processed and mvp) already contain `n_confident_experiments`, `frac_essential_confident`, and `essentiality_class`; they were produced by the same logic (e.g. via an upstream script or `aggregate_fitness_to_genes` plus merge).

**Embeddings:** Per-organism .pt files are produced by `scripts/generate_proteomelm_embeddings.py` (FASTA → ESM-C 600M → ProteomeLM transformer → .pt). Each .pt contains `embeddings` (tensor shape `[N, D]`) and `group_labels` (list of N strings `orgId:locusId`). Embeddings are **proteome-contextualized**: each gene's embedding reflects its sequence (via ESM-C) plus its relationship to the rest of that organism's proteome (via ProteomeLM). See `notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md`. **Embeddings are not yet integrated into the pipeline:** no code under `src/` loads these files or joins them to gene labels.

### Structure

- **Gene-level table:** The **genes** DataFrame (processed or mvp) is the model-ready label source: one row per gene with `essentiality_class` and optional numeric columns (e.g. frac_essential_confident, n_confident_experiments). Shape: (138,518, 18) for MVP; (221,005, 18) for processed (18 columns as of discovery).
- **Embeddings (current state):** Per-organism .pt files in `data/mvp/ProtLM_embedddings/`. With y labels: `data/mvp/ProtLM_embeddings_with_labels/` (each .pt: `embeddings`, `group_labels`, `y`).

### Class balance (MVP)

From the discovery script, gene-level essentiality class counts and percentages:

| Class | Count | Percentage |
|-------|-------|------------|
| always_essential | 1,668 | 1.20% |
| conditional | 17,385 | 12.55% |
| non_essential | 93,807 | 67.72% |
| no_data | 25,658 | 18.52% |

**Total genes (MVP):** 138,518.

The distribution is **highly imbalanced**: most genes are non_essential or no_data; always_essential is a small minority. This motivates metrics such as **AUPRC** (Area Under Precision-Recall Curve) for evaluation rather than accuracy alone.

### MLP baseline (planned — immediate next step)

A simple MLP will take the ProteomeLM embedding of a gene and predict essentiality class. **Input:** ProteomeLM embeddings only (no media, no condition encoding). **Data:** Load from `data/mvp/ProtLM_embeddings_with_labels/` (each .pt: embeddings, group_labels, y). **Exclude no_data:** filter to y ≠ -1 (~112,860 genes). **Class balance:** always_essential ~1.5%, conditional ~12.5%, non_essential ~67.7% of the filtered set. Use **class-weighted CrossEntropyLoss** (inverse frequency) to handle imbalance. **Goal:** Assess whether ProteomeLM embeddings alone yield relatively good results for always_essential and conditional prediction.

**Train/val/test split:** **Organism-based** (split by orgId, not by gene). Assign organisms to train/val/test (e.g. ~19/4/4 of the 27 MVP organisms); all genes from a held-out organism go to val or test. Rationale: ProteomeLM embeddings are proteome-contextualized, so holding out an organism means holding out a novel proteomic context—a natural generalization test. See `notes/06_HOMOLOGY_VS_ORGANISM_SPLITS.md` and `notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md`.

**Metrics:** accuracy, weighted F1, **AUPRC for always_essential**, **AUPRC for conditional** (primary interest).

**Architecture:** input dim = embedding size (e.g. 1152 for ProteomeLM-S), 1–2 hidden layers (256–512 units), ReLU, dropout, output 3 classes.

### Current vs intended state

- **Implemented:** Gene-level labels in `genes.parquet` (processed and mvp); load via `load_genes(subset)`. Logic in `src/fitness.py` for computing the same from raw fitness.
- **Not yet implemented:** Loading ProteomeLM .pt file(s) in `src`, joining embeddings to labels, and feeding them into an ML model. Organism FASTAs in `data/mvp/organism_fastas/` (27 files) are the input to `scripts/generate_proteomelm_embeddings.py`; output format is documented in `notes/02_DATA_DICTIONARY.md`.

---

## Conventions

- **Terminology:** essentiality_class, frac_essential_confident, n_confident_experiments (or n_confident in code), cor12, subset="processed" vs "mvp", and the four class names as above.
- **Loading:** Prefer Parquet via `src.data_io` (`load_genes`, `load_experiments`, `load_fitness`, `load_organisms`); media metadata via `load_media_experiments()` from `data/media_composition.xlsx` (sheet index 2).
- **Stages:** Processed (Stage I) → MVP (Stage II) → model-ready (Stage III: gene-level stats + labels ± embeddings), aligned with what exists in the repo.
