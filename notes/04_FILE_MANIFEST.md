# File manifest

Purpose of each file or directory in the project (code and key docs). Data assets are described in `02_DATA_DICTIONARY.md`; this manifest focuses on code and documentation.

---

## Root directory

| File | Purpose |
|------|---------|
| README.md | Project protocol: MVP plan, architecture, V2 roadmap, execution checklist. |
| pyproject.toml | Project metadata, dependencies, Ruff linting config (line-length 100, Python 3.10+). |
| requirements.txt | Dependency list (e.g. torch, pandas, pyarrow, biopython, esm, tqdm, openpyxl). |
| prompts.txt | User prompts and notes. |
| model.ipynb | Model development notebook (MLP baseline, embedding loading). |
| data_analysis.ipynb | Data exploration and analysis notebook. |

---

## config/

| File | Purpose |
|------|---------|
| config/pipeline.yaml | Pipeline configuration: data paths, step parameters (extract_data, classify_genes, filter_mvp, create_fastas, embeddings, labels). |

---

## src/

| File | Purpose |
|------|---------|
| src/__init__.py | Package initialization; exports public API (load_genes, load_experiments, load_fitness, load_organisms, load_media_experiments, inspect_table, fitness constants and functions). |
| src/config.py | Pipeline configuration loading (YAML → typed dataclasses). |
| src/data_io.py | Path resolution and data loaders (Parquet, Excel). |
| src/fitness.py | Essentiality classification thresholds and functions (vectorized aggregation, class assignment). |

---

## src/pipeline/

| File | Purpose |
|------|---------|
| src/pipeline/__init__.py | Pipeline orchestration; exports DataPipeline, STEP_ORDER. |
| src/pipeline/steps.py | Base class for pipeline steps; run_step_with_logging. |
| src/pipeline/extract_data.py | Step: extract raw data from source DB to Parquet. |
| src/pipeline/classify_genes.py | Step: compute essentiality_class for processed or mvp subset. |
| src/pipeline/filter_mvp.py | Step: filter to MVP organisms (media overlap, cor12 ≥ 0.2). |
| src/pipeline/create_fastas.py | Step: create per-organism FASTA files from sequences. |
| src/pipeline/generate_embeddings.py | Step: ProteomeLM embeddings from organism FASTAs → .pt (embeddings, group_labels). Output to `data/mvp/ProtLM_embedddings/`. See notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md. |
| src/pipeline/label_embeddings.py | Step: add y labels to .pt files; output to `data/mvp/ProtLM_embeddings_with_labels/`. |

---

## scripts/

| File | Purpose |
|------|---------|
| scripts/run_pipeline.py | Orchestrates the data pipeline. Usage: `--step`, `--from`, `--to`, `--config`. |
| scripts/encode_esmc.py | Standalone utility: ESMC-only encoding of organism FASTAs (not part of main pipeline). |
| scripts/data_discovery.py | Loads data via src.data_io, computes metrics for notes/05_DATA_ANALYSIS.md. |
| scripts/create_data_visuals.py | Generates visualizations for processed and MVP subsets (figures/). |

---

## notes/

| File | Purpose |
|------|---------|
| notes/01_PROJECT_OVERVIEW.md | High-level project summary (goals, data source, MVP vs full, implementation status). |
| notes/02_DATA_DICTIONARY.md | Data schemas and structures (Parquet, condition_vocab, embeddings, counts). |
| notes/03_LOGIC_FLOW.md | Pipeline flow and Mermaid diagram; planned steps noted. |
| notes/04_FILE_MANIFEST.md | This file: manifest of root, config/, src/, scripts/, and notes/. |
| notes/05_DATA_ANALYSIS.md | Data lifecycle, schemas, statistics, MLP baseline plan. |
| notes/06_HOMOLOGY_VS_ORGANISM_SPLITS.md | Reasoning on homology-based vs organism-based train/val/test splits. |
| notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md | Research on ProteomeLM: what it does, what embeddings represent (proteome-contextualized). |
| notes/Paper_DataSection_Draft.md | Research paper draft (preserved). |

**Legacy notes:** Outdated notes (DataAccess.md, DataFilteringAnalysis.md, DB_GeneralAnalysis.md, EMBEDDINGS_ORIGIN.md) have been moved to `notes/archive/` for reference only.

---

## Exclusions

- **../ProteomeLM/** — External tool (immutable); lives as a sibling project at the same level as GeneEssentiality (not inside this repo). Not documented here.
- **data/** — Data files and directories; documented in `02_DATA_DICTIONARY.md`, not listed as code in this manifest.
