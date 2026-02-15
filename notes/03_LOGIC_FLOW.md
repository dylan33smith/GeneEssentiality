# Logic flow

Step-by-step pipeline from path resolution to gene-level stats and essentiality class. Planned steps (embeddings, model training) are noted as not yet implemented.

---

## Pipeline steps

### 1. Path resolution

- **Project root:** `_project_root()` — from environment variable `GENEESSENTIALITY_ROOT` if set, otherwise the parent of the directory containing `src/`.
- **Data directories:** `get_data_dir()` → `project_root/data`; `get_processed_dir()` → `data/processed`; `get_mvp_dir()` → `data/mvp`. All loaders use these helpers; do not hardcode `data/` paths in shared code.

### 2. Load data

Choose subset `"processed"` or `"mvp"`. Call:

- `load_genes(subset)` → genes DataFrame
- `load_experiments(subset)` → experiments DataFrame
- `load_fitness(subset)` → fitness DataFrame
- `load_organisms(subset)` → organisms DataFrame
- Optionally: `load_media_experiments()` (reads `data/media_composition.xlsx`, sheet 2)

### 3. Fitness aggregation (optional)

`aggregate_fitness_to_genes(fitness)` takes the raw fitness table (columns `orgId`, `locusId`, `expName`, `fit`, `t`) and returns a DataFrame with one row per `(orgId, locusId)` and columns:

- `n_total_experiments`, `n_confident_experiments` (confident = \|t\| ≥ 2)
- `frac_not_confident`, `mean_fit_confident`
- `frac_essential_all`, `frac_essential_confident` (essential in an experiment = fit &lt; −1)
- `essentiality_class`

Implementation uses vectorized groupby/agg only (no row-wise loops). Gene metadata (scaffoldId, begin, end, etc.) is not added; merge with the genes table if needed.

### 4. Essentiality classification

`get_essentiality_class(frac_essential_confident, n_confident)` returns a class label (or array of labels). Rules (from `src/fitness.py`):

- `n_confident == 0` → `"no_data"`
- `frac_essential_confident > 0.8` → `"always_essential"`
- `0.1 ≤ frac_essential_confident ≤ 0.8` → `"conditional"`
- `frac_essential_confident < 0.1` (and n_confident &gt; 0) → `"non_essential"`

If you use `aggregate_fitness_to_genes`, `essentiality_class` is already computed; otherwise call `get_essentiality_class` on your own aggregates.

---

## Data flow diagram

```mermaid
flowchart TD
    subgraph paths[Path Resolution]
        Root[Project Root]
        DataDir[data/]
        ProcessedDir[processed/]
        MvpDir[mvp/]
    end
    subgraph loaders[src.data_io Loaders]
        LoadGenes[load_genes]
        LoadExps[load_experiments]
        LoadFitness[load_fitness]
        LoadOrgs[load_organisms]
        LoadMedia[load_media_experiments]
    end
    subgraph dataframes[DataFrames]
        Genes[genes DataFrame]
        Exps[experiments DataFrame]
        Fitness[fitness DataFrame]
        Orgs[organisms DataFrame]
    end
    subgraph fitness_module[src.fitness]
        Aggregate[aggregate_fitness_to_genes]
        GetClass[get_essentiality_class]
    end
    subgraph output[Output]
        GeneStats[Gene-level stats + essentiality_class]
    end
    Root --> DataDir
    DataDir --> ProcessedDir
    DataDir --> MvpDir
    ProcessedDir --> LoadGenes
    MvpDir --> LoadGenes
    LoadGenes --> Genes
    LoadFitness --> Fitness
    Fitness --> Aggregate
    Aggregate --> GeneStats
    Genes --> GetClass
    GetClass --> GeneStats
```

---

## Planned steps (not implemented)

- **Embeddings integration:** Generate per-organism ProteomeLM .pt files via `scripts/generate_proteomelm_embeddings.py`; load the .pt file(s) (keys `embeddings`, `group_labels`) and merge with genes by `group_labels` (orgId:locusId); not yet wired into `src`.
- **Condition encoding:** Use `condition_vocab.json` to build multi-hot (or other) condition vectors.
- **Model training:** Transformer/MLP consuming embeddings + condition vectors → fitness (or essentiality); not implemented.
- **Train/val/test splits:** Homology-based (e.g. 50% identity threshold) to avoid leakage; planned per README, not yet implemented.
