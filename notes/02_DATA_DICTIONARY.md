# Data Dictionary

Schemas and semantics for all data assets used by the GeneEssentiality pipeline. Counts and vocabulary sizes are from the current data files (Parquet and `condition_vocab.json`).

---

## Parquet tables

All Parquet files live under `data/processed/` (full database) or `data/mvp/` (MVP subset). Load via `src.data_io`: `load_genes(subset)`, `load_experiments(subset)`, `load_fitness(subset)`, `load_organisms(subset)` with `subset in ("processed", "mvp")`.

### genes.parquet

One row per gene. Columns (from `src/data_io.py` and `src/fitness.py`):

| Column | Semantics |
|--------|-----------|
| orgId | Organism identifier (links to organisms.parquet). |
| locusId | Gene locus identifier (unique per organism). |
| sysName | Systematic name (e.g. b0001). |
| scaffoldId | Chromosome/plasmid ID. |
| begin, end | Genomic coordinates (bp). |
| gene_length | Length in bp. |
| type | Gene type (e.g. CDS). |
| strand | + or -. |
| gene | Gene symbol (optional). |
| desc | Description (optional). |
| GC | GC content (optional). |
| n_total_experiments | Number of experiments with a fitness value for this gene. |
| n_confident_experiments | Number of experiments where \|t\| ≥ 2. |
| frac_not_confident | 1 − (n_confident / n_total). |
| mean_fit_confident | Mean fitness over confident experiments only. |
| frac_essential_all | Fraction of all experiments where fit &lt; −1. |
| frac_essential_confident | Fraction of confident experiments where fit &lt; −1. |
| essentiality_class | Label from 80%/10% rule: `always_essential`, `conditional`, `non_essential`, or `no_data`. |

**Essentiality class rule** (from `src/fitness.py`): `n_confident == 0` → `no_data`; `frac_essential_confident > 0.8` → `always_essential`; `0.1 ≤ frac_essential_confident ≤ 0.8` → `conditional`; `frac_essential_confident < 0.1` (and n_confident &gt; 0) → `non_essential`. Confident = \|t\| ≥ 2; essential in an experiment = fit &lt; −1.

### experiments.parquet

One row per experiment. Columns (from `src/data_io.py`):

| Column | Semantics |
|--------|-----------|
| orgId | Organism. |
| expName | Experiment name (unique key with orgId). |
| expDesc | Description. |
| expGroup | Experiment group label. |
| mutantLibrary | Library identifier. |
| media | Media name (e.g. LB, M9). |
| mediaStrength | Strength/category. |
| condition_1 | Primary condition (e.g. stressor). |
| concentration_1, units_1 | Concentration and units for condition_1. |
| temperature | Growth temperature. |
| aerobic | Aerobic/anaerobic. |
| cor12 | Replicate correlation (quality); MVP uses cor12 ≥ 0.2. |
| maxFit, nGenerations | Optional experiment metadata. |

Additional columns may exist; see file or `inspect_table(load_experiments(subset), "experiments")`.

### fitness.parquet

One row per (gene, experiment). Columns:

| Column | Semantics |
|--------|-----------|
| orgId | Organism. |
| locusId | Gene. |
| expName | Experiment. |
| fit | Log₂ fitness (e.g. from Tn-Seq). fit &lt; −1 → essential in this experiment. |
| t | t-statistic for the estimate; \|t\| ≥ 2 → confident. |

### organisms.parquet

One row per organism. Columns:

| Column | Semantics |
|--------|-----------|
| orgId | Organism identifier. |
| division | Taxonomic division. |
| genus, species, strain | Taxonomy. |
| taxonomyId | NCBI taxonomy ID. |

---

## Other data

### condition_vocab.json

Path: `data/mvp/condition_vocab.json`. Structure: keys are category names; values are lists of unique strings found in the MVP experiments (for building multi-hot or embedding indices). **Not loaded by current `src` code**; intended for future condition encoding.

| Key | Description | Actual size |
|-----|-------------|-------------|
| media | Media names | 28 |
| condition_1 | Primary condition labels | 189 |
| composite | Composite condition labels | 448 |
| expGroup | Experiment group | 18 |
| units_1 | Units for concentration_1 | 12 |
| aerobic | Aerobic/anaerobic | 3 |
| temperature | Temperature categories | 9 |

Sizes are from the current file on disk.

### media_composition.xlsx

Path: `data/media_composition.xlsx`. **Sheet index 2** (third sheet) is loaded by `load_media_experiments()`. Contains media/experiment composition metadata; column semantics are defined by the Excel source.

### Embeddings (ProteomeLM .pt files)

Produced by `scripts/generate_proteomelm_embeddings.py` from a single-organism FASTA (e.g. `data/mvp/organism_fastas/Keio.fasta`). One .pt per organism; combine or load separately for full MVP. **Not loaded by `src`**; embeddings are not yet integrated into the pipeline.

**Format (per-organism .pt):**

- `embeddings`: `torch.Tensor` of shape `(N, D)` — ProteomeLM contextualized embedding per gene (output of ProteomeLM transformer; D is model-dependent, e.g. 1152 for default ProteomeLM-S logits).
- `group_labels`: `list` of length N — gene identifiers in format `orgId:locusId` (same order as rows in `embeddings`), for joining to `genes.parquet` (e.g. match to `orgId` + `locusId` or a composite key).

**Pipeline:** FASTA (headers `>orgId:locusId`) → ESM-C 600M (via ProteomeLM package) → ProteomeLM transformer → save `embeddings` + `group_labels`. Run from repo root with ProteomeLM on PYTHONPATH.

---

## Row counts (from Parquet)

| Subset | genes | experiments | fitness | organisms |
|--------|-------|-------------|---------|-----------|
| **mvp** | 138,518 | 2,164 | 8,688,562 | 27 |
| **processed** | 221,005 | 7,552 | 27,410,721 | 48 |

These counts are from the current Parquet files and are the source of truth for documentation.
