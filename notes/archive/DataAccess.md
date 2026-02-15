# Data Access Guide

> **Note**: This documents the **processed parquet files** created by the data pipeline.
> For raw database analysis, see `DB_GeneralAnalysis.md` and `ConditionsAnalysis.md`.

## Data Location

All processed data is in `data/` with two main directories:

```
data/
├── processed/     # Full database (all 48 organisms)
│   ├── genes.parquet
│   ├── experiments.parquet
│   ├── fitness.parquet
│   ├── organisms.parquet
│   ├── domains.parquet
│   ├── orthologs.parquet
│   └── proteins.fasta
└── mvp/           # Filtered subset (27 organisms, quality-filtered)
    ├── genes.parquet
    ├── experiments.parquet
    ├── fitness.parquet
    ├── organisms.parquet
    ├── condition_vocab.json
    └── sequences/proteins.fasta
```

**Use `mvp/` for training** - it has the quality filters applied (cor12 ≥ 0.2) and only includes organisms with overlapping media types (LB, RCH2, M9).

## File Format

All tables are **Parquet files** - a columnar format that's fast to read with pandas:

```python
import pandas as pd

# Load any table
genes = pd.read_parquet("data/mvp/genes.parquet")
fitness = pd.read_parquet("data/mvp/fitness.parquet")
```

---

## Table Schemas

### 1. genes.parquet - Gene metadata (138,518 rows in MVP)

| Column | Type | Description |
|--------|------|-------------|
| orgId | str | Organism ID (e.g., "Keio") |
| locusId | str | Gene ID (e.g., "b0001") |
| sysName | str | Systematic name |
| scaffoldId | str | Chromosome/scaffold |
| begin | int | Start position (bp) |
| end | int | End position (bp) |
| gene_length | int | Gene length in bp (computed: end - begin + 1) |
| type | int | Gene type (1 = protein-coding) |
| strand | str | "+" or "-" |
| gene | str | Gene name (often empty) |
| desc | str | Gene description |
| GC | float | GC content (0-1) |
| n_total_experiments | int | Total fitness records for this gene (all, regardless of confidence) |
| n_confident_experiments | int | Number of experiments with \|t\| >= 2 |
| frac_not_confident | float | Fraction of total experiments that are not confident |
| mean_fit_confident | float | Mean fitness across confident experiments |
| frac_essential_all | float | Fraction of ALL fitness records where fit < -1 |
| frac_essential_confident | float | Fraction of CONFIDENT records where fit < -1 |
| essentiality_class | str | Classification: "always_essential", "conditional", "non_essential", "no_data" |

**Essentiality classification** (based on confident measurements only, |t| >= 2):
- `always_essential`: Essential in >80% of confident experiments (4,457 genes in MVP)
- `conditional`: Essential in 10-80% of confident experiments (21,547 genes in MVP)
- `non_essential`: Essential in <10% of confident experiments (62,018 genes in MVP)
- `no_data`: No confident fitness measurements (50,496 genes in MVP)

**Genomic order**: Sort by `(orgId, scaffoldId, begin)`.

---

### 2. experiments.parquet - Experimental conditions (2,244 rows in MVP)

| Column | Type | Description |
|--------|------|-------------|
| orgId | str | Organism ID |
| expName | str | Experiment name |
| expDesc | str | Experiment description |
| expGroup | str | Category (Stress, Carbon source, Nitrogen source, etc.) |
| mutantLibrary | str | Library used (for bias correction) |
| media | str | Growth media (LB, RCH2_defined_noCarbon, etc.) |
| mediaStrength | float | Media dilution factor |
| condition_1 | str | Primary stressor/condition |
| concentration_1 | str | Concentration value |
| units_1 | str | Units (mM, mg/ml, %, etc.) |
| condition_2 | str | Secondary condition (often solvent like DMSO) |
| concentration_2 | str | Secondary concentration |
| units_2 | str | Secondary units |
| condition_3 | str | Tertiary condition (rare) |
| concentration_3 | str | Tertiary concentration |
| units_3 | str | Tertiary units |
| temperature | str | Growth temperature (e.g., "30") |
| pH | str | pH value |
| aerobic | str | "Aerobic", "Anaerobic", or "Microaerobic" |
| vessel | str | Growth vessel type |
| liquid | str | Liquid/solid medium |
| shaking | str | Shaking speed |
| cor12 | float | Replicate correlation (quality metric, filtered ≥ 0.2) |
| maxFit | float | Maximum fitness in experiment |
| nGenerations | float | Number of generations grown |

---

### 3. fitness.parquet - Target variable (8,985,609 rows in MVP)

| Column | Type | Description |
|--------|------|-------------|
| orgId | str | Organism ID |
| locusId | str | Gene ID |
| expName | str | Experiment name |
| **fit** | float | **Log2 fitness score (TARGET VARIABLE)** |
| **t** | float | **T-statistic for significance** |

**Interpretation:**
- fit < -2: Gene is essential/important in this condition
- fit ≈ 0: Gene is neutral
- fit > 1: Gene knockout is beneficial
- |t| ≥ 2: Statistically significant

---

### 4. organisms.parquet - Organism metadata (27 rows in MVP)

| Column | Type | Description |
|--------|------|-------------|
| orgId | str | Organism ID (e.g., "Keio", "Putida") |
| division | str | Taxonomic division |
| genus | str | Genus name |
| species | str | Species name |
| strain | str | Strain identifier |
| taxonomyId | int | NCBI taxonomy ID |

---

### 5. domains.parquet - Protein domains (456,712 rows, processed/ only)

| Column | Type | Description |
|--------|------|-------------|
| orgId | str | Organism ID |
| locusId | str | Gene ID |
| domainDb | str | Database: "PFam" or "TIGRFam" |
| domainId | str | Domain accession (e.g., "PF00001") |
| domainName | str | Domain name |
| begin | int | Start position in protein |
| end | int | End position in protein |
| score | float | HMM score |
| evalue | float | E-value |

---

### 6. orthologs.parquet - For cluster-based splits (2,838,750 rows, processed/ only)

| Column | Type | Description |
|--------|------|-------------|
| orgId1 | str | Organism ID of gene 1 |
| locusId1 | str | Locus ID of gene 1 |
| orgId2 | str | Organism ID of gene 2 |
| locusId2 | str | Locus ID of gene 2 |
| ratio | float | Sequence similarity ratio |

---

### 7. condition_vocab.json - Label encodings (MVP only)

Structure:
```json
{
  "media": {"vocab": {...}, "inverse": {...}, "size": 32},
  "condition_1": {"vocab": {...}, "inverse": {...}, "size": 192},
  "expGroup": {"vocab": {...}, "inverse": {...}, "size": 19},
  "units_1": {"vocab": {...}, "inverse": {...}, "size": 12},
  "composite": {"vocab": {...}, "inverse": {...}, "size": 456},
  "aerobic": {"vocab": {...}, "inverse": {...}, "size": 3},
  "temperature": {"vocab": {...}, "inverse": {...}, "size": 9},
  "metadata": {"n_experiments": 2244, "n_organisms": 27, ...}
}
```

Usage:
```python
import json
with open("data/mvp/condition_vocab.json") as f:
    vocab = json.load(f)

# Get media encoding
media_idx = vocab['media']['vocab']['LB']  # → integer index

# Get condition encoding  
cond_idx = vocab['condition_1']['vocab']['Sodium nitrite']  # → integer index

# Vocab sizes for creating vectors
n_media = vocab['media']['size']  # 32
n_conditions = vocab['condition_1']['size']  # 192
```

---

### 8. proteins.fasta - Protein sequences

FASTA format with header `>orgId:locusId`:
```
>acidovorax_3H11:Ac3H11_1
VRAHLFHHGPLCGVTHFAAAPGRGFLHVLRRGEMVVTHQPQAGS...
>acidovorax_3H11:Ac3H11_2
VPDLQRFTTCKEITMSDHSSTARVALVDRTAASGPARALLDQIH...
```

- **processed/proteins.fasta**: 221,030 sequences (all organisms, copied from raw)
- **mvp/proteins.fasta**: 138,518 sequences (filtered to MVP genes only)

Note: The processed FASTA has ~25 more sequences than genes.parquet (221,005) because it's a direct copy from the raw data, while genes.parquet filters to protein-coding genes (type=1) only.

```python
from Bio import SeqIO

sequences = {
    rec.id: str(rec.seq)
    for rec in SeqIO.parse("data/mvp/proteins.fasta", "fasta")
}
# sequences["acidovorax_3H11:Ac3H11_1"] → "VRAHLFHHGPLCG..."
```

---

## Accessing Data by Project Step

### Step 1: Data Engineering - Build training dataset

```python
import pandas as pd

# Load core tables
genes = pd.read_parquet("data/mvp/genes.parquet")
experiments = pd.read_parquet("data/mvp/experiments.parquet")
fitness = pd.read_parquet("data/mvp/fitness.parquet")

# Join fitness with experiment metadata
training_data = fitness.merge(
    experiments[['orgId', 'expName', 'media', 'condition_1', 'concentration_1', 
                 'units_1', 'temperature', 'aerobic', 'expGroup']],
    on=['orgId', 'expName']
)
```

### Step 2: ESM-2 Embeddings - Load sequences

```python
from Bio import SeqIO

sequences = {
    rec.id: str(rec.seq)
    for rec in SeqIO.parse("data/mvp/proteins.fasta", "fasta")
}

# Get sequence for a gene
gene_key = f"{orgId}:{locusId}"
seq = sequences[gene_key]
```

### Step 3: Cluster Splits - Use orthologs

```python
orthologs = pd.read_parquet("data/processed/orthologs.parquet")

# Filter to MVP organisms
mvp_orgs = set(pd.read_parquet("data/mvp/organisms.parquet")['orgId'])
mvp_orthologs = orthologs[
    orthologs['orgId1'].isin(mvp_orgs) & 
    orthologs['orgId2'].isin(mvp_orgs)
]

# Build graph for clustering (genes with high ratio are similar)
# Use for train/val/test splits to prevent sequence leakage
```

### Step 4: Condition Encoding - Multi-hot vectors

```python
import json
import numpy as np

with open("data/mvp/condition_vocab.json") as f:
    vocab = json.load(f)

def encode_condition(media, condition):
    """Create multi-hot vector for media + condition."""
    media_vec = np.zeros(vocab['media']['size'])
    cond_vec = np.zeros(vocab['condition_1']['size'])
    
    if media in vocab['media']['vocab']:
        media_vec[vocab['media']['vocab'][media]] = 1
    if condition in vocab['condition_1']['vocab']:
        cond_vec[vocab['condition_1']['vocab'][condition]] = 1
    
    return np.concatenate([media_vec, cond_vec])

# Example: encode an experiment's condition
vec = encode_condition("LB", "Sodium nitrite")  # shape: (32 + 192,) = (224,)
```

---

## Quick Reference

| What you need | File | Key columns |
|---------------|------|-------------|
| Target variable | `mvp/fitness.parquet` | fit, t |
| Gene features | `mvp/genes.parquet` | gene_length, GC, strand, begin, end |
| Genomic order | `mvp/genes.parquet` | orgId, scaffoldId, begin (sort order) |
| Essentiality class | `mvp/genes.parquet` | essentiality_class, frac_essential |
| Experiment conditions | `mvp/experiments.parquet` | media, condition_1, expGroup, aerobic |
| Protein sequences | `mvp/proteins.fasta` | FASTA format |
| Domain features | `processed/domains.parquet` | domainDb, domainId, domainName |
| Train/test splits | `processed/orthologs.parquet` | orgId1, locusId1, orgId2, locusId2, ratio |
| Condition encodings | `mvp/condition_vocab.json` | vocab dictionaries |
| Organism info | `mvp/organisms.parquet` | genus, species, taxonomyId |

---

## Data Statistics

### MVP Subset (quality-filtered)
| Dataset | Count |
|---------|-------|
| Organisms | 27 |
| Genes | 138,518 |
| Experiments | 2,244 |
| Fitness records | 8,985,609 |
| Protein sequences | 138,518 |

### Vocabulary Sizes
| Vocabulary | Size |
|------------|------|
| Media types | 32 |
| Conditions (condition_1) | 192 |
| Experiment groups | 19 |
| Composite (media_condition) | 456 |
| Aerobic states | 3 |
| Temperatures | 9 |

### Full Database (processed/)
| Dataset | Count |
|---------|-------|
| Organisms | 48 |
| Genes | 221,005 |
| Experiments | 7,552 |
| Fitness records | 27,410,721 |
| Domain annotations | 456,712 |
| Ortholog pairs | 2,838,750 |
