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
| testing.ipynb | Exploration notebook: uses `src` loaders, vectorized groupby patterns, inspects MVP data. |

---

## src/

| File | Purpose |
|------|---------|
| src/__init__.py | Package initialization; exports public API (load_genes, load_experiments, load_fitness, load_organisms, load_media_experiments, inspect_table, fitness constants and functions). |
| src/data_io.py | Path resolution and data loaders (Parquet, Excel). |
| src/fitness.py | Essentiality classification thresholds and functions (vectorized aggregation, class assignment). |

---

## scripts/

| File | Purpose |
|------|---------|
| scripts/generate_proteomelm_embeddings.py | ProteomeLM embedding generator: reads a single-organism FASTA (--input), uses ESM-C 600M + ProteomeLM to produce proteome-contextualized embeddings per gene, writes .pt with `embeddings` and `group_labels` (orgId:locusId). Output to `data/mvp/ProtLM_embedddings/`. See notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md. |
| scripts/add_y_labels_to_embeddings.py | Adds y label vectors to .pt files: reads from ProtLM_embedddings/, joins to genes.parquet, maps essentiality_class → 0/1/2/-1, writes to ProtLM_embeddings_with_labels/. |

---

## notes/

| File | Purpose |
|------|---------|
| notes/01_PROJECT_OVERVIEW.md | High-level project summary (goals, data source, MVP vs full, implementation status). |
| notes/02_DATA_DICTIONARY.md | Data schemas and structures (Parquet, condition_vocab, embeddings, counts). |
| notes/03_LOGIC_FLOW.md | Pipeline flow and Mermaid diagram; planned steps noted. |
| notes/04_FILE_MANIFEST.md | This file: manifest of root, src/, scripts/, and notes/. |
| notes/05_DATA_ANALYSIS.md | Data lifecycle, schemas, statistics, MLP baseline plan. |
| notes/06_HOMOLOGY_VS_ORGANISM_SPLITS.md | Reasoning on homology-based vs organism-based train/val/test splits. |
| notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md | Research on ProteomeLM: what it does, what embeddings represent (proteome-contextualized). |
| notes/Paper_DataSection_Draft.md | Research paper draft (preserved). |
| notes/exp_condition_exploration.ipynb | Exploratory analysis notebook (preserved). |

**Legacy notes:** Outdated notes (DataAccess.md, DataFilteringAnalysis.md, DB_GeneralAnalysis.md, EMBEDDINGS_ORIGIN.md) have been moved to `notes/archive/` for reference only.

---

## Exclusions

- **ProteomeLM/** — External tool (immutable); not part of this codebase and not documented here.
- **data/** — Data files and directories; documented in `02_DATA_DICTIONARY.md`, not listed as code in this manifest.
