# Data — First Draft (Research Article)

*Detailed first-draft description of the data for the context-aware gene essentiality project. Trim and refine for submission.*

---

## Data source and provenance

The data used in this study come from the **Fitness Browser** database, a public resource that aggregates genome-wide gene fitness measurements from transposon insertion sequencing (Tn-Seq) experiments across multiple bacterial species and growth conditions. The database integrates results from many primary studies (dozens of publications are linked in the resource) and provides a unified schema for genes, organisms, experiments, and fitness scores. We did not perform any new wet-lab experiments; all data were obtained from the processed database (or its source exports) and then filtered and transformed as described below.

**Data location and format.** After processing, the data are stored in two main tiers: a **full processed** set (all organisms and experiments that pass basic schema checks) and an **MVP (minimum viable product)** subset used for model development and training. All tables are stored in Parquet format. The full processed set lives in `data/processed/` and includes genes, experiments, fitness, organisms, protein domains, ortholog pairs, and protein sequences in FASTA form. The MVP subset lives in `data/mvp/` and contains the same core tables (genes, experiments, fitness, organisms, protein sequences) plus a condition vocabulary file (`condition_vocab.json`) used for encoding growth conditions. Protein sequences in the MVP are restricted to genes that appear in the MVP gene table (one sequence per gene, keyed by `orgId:locusId`). Domain and ortholog tables are used only from the full processed set when constructing homology-based train/validation/test splits, because clustering is computed across the full database before subsetting to MVP organisms.

---

## Full database scale (pre-filtering)

The Fitness Browser database in the form we use comprises **48 organisms**, **7,552 experiments**, and **27,410,721 gene–experiment fitness records**. The gene table contains **221,005 protein-coding genes** (after restricting to type = protein-coding). These numbers refer to the full processed export before any quality or cohort filters. The database also includes substantial annotation: **456,712 protein domain annotations** (PFam and TIGRFam), **2,838,750 ortholog pairs** linking genes across organisms by sequence similarity, and **221,030 protein sequences** in FASTA format (the small excess over the gene count is due to the FASTA being a direct copy from the raw resource while the gene table applies a strict protein-coding filter). Organisms are predominantly Proteobacteria (35 of 48); two archaeal organisms (Methanococcus) are present. Gene-level summary statistics in the full database: mean gene length ~968 bp, mean GC content ~59%; about 8.9% of genes are short (<300 bp), which is relevant for Tn-Seq library bias (fewer insertions in short genes). Approximately 76–85% of protein-coding genes have at least one fitness measurement; coverage varies by organism and experiment set.

---

## Subset definition: the MVP (“terrestrial” / connected) cohort

To control for confounding between growth medium and organism and to force the model to use gene-level and condition-level information rather than media identity, we restrict training and evaluation to a **connected subset** of the database. We refer to this as the **MVP subset** (or “terrestrial” subset). The defining criterion is **organism–media overlap**: we include only organisms that use at least one of the shared, commonly used media types **LB**, **RCH2_defined** (and its variants), or **M9** (and its variants). These media are used by a large, overlapping set of model organisms (e.g., Escherichia coli, Pseudomonas spp., Klebsiella spp., Ralstonia spp.). In contrast, the full database contains “ecosystem silos”: for example, marine-specific media are used only for marine organisms, and methanogen media only for archaea. If all data were pooled, a model could achieve spuriously good performance by predicting fitness from media type alone. The MVP subset therefore includes **27 organisms** and **2,244 experiments** (after the quality filter described below), covering roughly 60% of the experiments in the full database and forming a single connected component with respect to media usage. Excluded organisms are those that appear only in experiments using media not shared with this core set (e.g., some marine, methanogen, or other specialist media). The MVP gene table has **138,518 genes** (all protein-coding), and the MVP fitness table has **8,985,609** gene–experiment records. Protein sequences in the MVP FASTA match these 138,518 genes one-to-one.

---

## Quality filtering of experiments

Replicate reproducibility in the Fitness Browser is summarized by **cor12**, the correlation between replicate samples within an experiment. In the full database, the mean cor12 is approximately 0.28, and about **38% of experiments** have cor12 &lt; 0.2. We treat low cor12 as an indicator of unreliable fitness estimates and **exclude all experiments with cor12 &lt; 0.2** from the MVP. Thus, the 2,244 MVP experiments are the intersection of (i) experiments belonging to organisms in the MVP cohort and (ii) experiments with cor12 ≥ 0.2. No further experiment-level filters are applied for the MVP; gene-level and fitness-level handling (e.g., use of t-statistics for confidence) are described in the following sections.

---

## Gene-level data and essentiality classification

The **gene table** (genes.parquet) provides one row per gene with identifiers (orgId, locusId, sysName), genomic position (scaffoldId, begin, end, strand), and derived features: **gene_length** (end − begin + 1), **GC** content (0–1), and **type** (we retain only type = 1, protein-coding). Optional fields include gene name, description, and counts/summaries of fitness records (see below). **Genomic position** (scaffoldId, begin, end) is available for each gene; genomic order is defined by sorting genes by (orgId, scaffoldId, begin).

For each gene, we compute summary statistics over its fitness records across experiments. These include: **n_total_experiments** (number of experiment-level fitness values), **n_confident_experiments** (number of records with |t| ≥ 2), **frac_not_confident** (fraction of records with |t| &lt; 2), **mean_fit_confident** (mean fitness over confident records only), **frac_essential_all** (fraction of all records with fit &lt; −1), and **frac_essential_confident** (fraction of confident records with fit &lt; −1). Using the confident measurements only, we assign an **essentiality_class** to each gene:

- **always_essential**: essential in &gt;80% of confident experiments (4,457 genes in MVP);
- **conditional**: essential in 10–80% of confident experiments (21,547 genes in MVP);
- **non_essential**: essential in &lt;10% of confident experiments (62,018 genes in MVP);
- **no_data**: no confident fitness measurements (50,496 genes in MVP).

The threshold “essential” here is fitness &lt; −1 (log₂ scale). These classes are used for evaluation (e.g., AUPRC for essential vs non-essential) and optionally for stratified analysis; they are not used as model inputs. The strong imbalance (many non_essential and no_data, fewer always_essential and conditional) reflects both biology and the zero-inflated distribution of fitness (see below).

---

## Experiment and condition metadata

The **experiments table** (experiments.parquet) has one row per experiment and describes the growth environment. Key fields include:

- **Identifiers and categories:** orgId, expName, expDesc, **expGroup** (e.g., Stress, Carbon source, Nitrogen source, Antibiotics, control, pH, metal limitation).
- **Growth medium:** **media** (e.g., LB, RCH2_defined, RCH2_defined_noCarbon, M9 minimal media_noCarbon), **mediaStrength** (dilution factor when applicable).
- **Primary condition:** **condition_1** (the main stressor or variable, e.g., “Sodium nitrite”, “D-Glucose”, “Cobalt chloride hexahydrate”), **concentration_1**, **units_1** (e.g., mM, mg/ml, vol%). In the full database, ~98.6% of experiments have concentration data for condition_1.
- **Secondary/tertiary conditions:** condition_2/3, concentration_2/3, units_2/3; condition_2 is often a solvent (e.g., DMSO, EtOH) in ~47% of experiments; condition_3/4 are rare.
- **Physical parameters:** **temperature** (e.g., 30, 37, 25 °C), **pH**, **aerobic** (Aerobic, Anaerobic, Microaerobic), **vessel**, **liquid** (liquid vs solid), **shaking**.
- **Quality and design:** **cor12** (replicate correlation; we keep only cor12 ≥ 0.2 in MVP), **maxFit**, **nGenerations**, **mutantLibrary**.

In the full database, condition categories are represented as follows: Stress (~2,854 experiments, 38 organisms), Carbon source (~1,838, 36 organisms), Nitrogen source (~1,093, 33 organisms), Antibiotics (~428, 36 organisms), plus control and other groups. Physical conditions: temperature is most commonly 30 °C (~63%), 37 °C (~17%), 25 °C (~12%); ~70% of experiments are Aerobic, ~27% Anaerobic; the vast majority are liquid phase. Concentration units are predominantly mM (~69%), mg/ml (~18%), vol% (~3%). In the MVP, the **condition vocabulary** (condition_vocab.json) defines label encodings for model input: **media** (28 distinct media types in the MVP), **condition_1** (189 distinct primary conditions), **expGroup** (18 categories), **units_1** (12), **composite** (448 unique media–condition_1 combinations), **aerobic** (3), **temperature** (9). The composite vocabulary is used when we encode a single “media + primary stressor” combination; the hierarchical option is to concatenate separate one-hot or multi-hot vectors for media and condition_1 (e.g., 28 + 189 = 217 dimensions).

---

## Fitness (target variable)

The **fitness table** (fitness.parquet) contains one row per (organism, gene, experiment) and provides the **target variable** for prediction. Fields are orgId, locusId, expName, **fit**, and **t**.

- **fit**: Log₂ fitness score. Interpretation: fit &lt; −2 indicates the gene is strongly detrimental (essential or highly important) in that condition; fit ≈ 0 indicates roughly neutral; fit &gt; 1 indicates the knockout is beneficial in that condition. We use fit directly as the regression target and also derive binary “essential” labels (e.g., fit &lt; −1 or fit &lt; −2) for classification metrics.
- **t**: T-statistic associated with the fitness estimate. |t| ≥ 2 is typically used as a significance threshold; we use it to define “confident” measurements for gene-level summaries and optional training weighting.

**Distribution (full database):** The fitness distribution is **highly zero-inflated**. Approximately **94.8%** of all fitness records fall in the “neutral” range (−1 to 1); only about **1.84%** have fit &lt; −2 (strongly essential). About **88.5%** of records have |t| &lt; 2 (not statistically significant). This has direct implications for modeling: (1) a naive mean-squared error loss would be dominated by neutral points; we therefore use a **weighted loss** that upweights essential (e.g., fit &lt; −1) relative to neutral (e.g., fit &gt; −0.5); (2) for essential-gene detection we report **AUPRC** (area under the precision–recall curve) in addition to regression metrics, because the positive class (essential) is rare. The same qualitative distribution holds in the MVP subset, with 8,985,609 records.

---

## Protein sequences and embeddings

Protein sequences are stored in FASTA format with headers `>orgId:locusId`. The MVP FASTA contains **138,518 sequences**, one per gene in the MVP gene table. We do not use raw sequences directly in the model; instead, we precompute **protein language model embeddings** (ESM-2, e.g. esm2_t33_650M_UR50D) once per sequence and store them on disk. Each gene is therefore represented by a fixed-size vector (e.g., 1280 dimensions). We do not fine-tune the protein encoder in the MVP. To correct for **library bias** (e.g., short genes having fewer insertions and potentially biased fitness estimates), we can concatenate **gene_length** and optionally normalized read-based features to the embedding before feeding it to the downstream model. Sequence and embedding coverage is 100% for MVP genes by construction.

---

## Train/validation/test split and homology control

To evaluate **generalization** rather than memorization of similar sequences, we use a **homology-based split**. Proteins are clustered by sequence similarity (e.g., 50% identity using MMseqs2 or the existing ortholog table). Clusters are assigned to train, validation, and test (e.g., 70% / 15% / 15%) such that **no protein in the test set has &gt;50% sequence identity to any protein in the training set**. Thus, test performance reflects ability to predict fitness for genes that are not closely related to training genes. The ortholog table (2,838,750 pairs in the full processed set) is used to build or validate these clusters; for the MVP we restrict to ortholog pairs within the 27 MVP organisms when constructing the split. We also plan **generalization benchmarks**: (A) hold out all experiments for a specific condition (e.g., Sodium nitrite) to test chemical generalization; (B) hold out one organism entirely (e.g., Pseudomonas putida) to test organism generalization. Care is taken that no (gene, experiment) fitness value appears in more than one of train/val/test (no leakage across splits).

---

## Condition encoding for the model (MVP)

For the MVP, conditions are encoded as **discrete labels**, not as continuous chemical or molecular features. We use the condition vocabulary to map (media, condition_1, and optionally expGroup, temperature, aerobic) to integer indices and then to **multi-hot or one-hot vectors**. Two natural options: (1) **composite**: one vocabulary of size 448 (media_condition_1 combinations), one-hot; (2) **hierarchical**: concatenate a media vector (size 28) and a condition_1 vector (size 189). The model receives this condition vector as a single “condition” token (or as multiple tokens if we add temperature/aerobic). We do not yet use molecular structure (e.g., SMILES) or continuous concentration in the MVP; that is reserved for a future extension (V2) to support zero-shot prediction for new chemicals.

---

## Summary statistics (quick reference)

| Quantity | Full (processed) | MVP |
|----------|-------------------|-----|
| Organisms | 48 | 27 |
| Genes (protein-coding) | 221,005 | 138,518 |
| Experiments | 7,552 | 2,244 (cor12 ≥ 0.2) |
| Fitness records | 27,410,721 | 8,985,609 |
| Protein sequences | 221,030 | 138,518 |
| Domain annotations | 456,712 | (use full for splits) |
| Ortholog pairs | 2,838,750 | (use full, filter to MVP orgs) |
| Media types (vocab) | — | 28 |
| Primary conditions (vocab) | — | 189 |
| Composite (media × condition_1) | — | 448 |

**Essentiality classes (MVP):** always_essential 4,457; conditional 21,547; non_essential 62,018; no_data 50,496.

---

## Caveats and limitations (for Discussion)

- **MVP scope:** Results apply to the terrestrial/connected subset only; marine, methanogen, and other silos are not represented.
- **Condition encoding:** Conditions are categorical; concentration and molecular structure are not used in the MVP.
- **Fitness quality:** Even with cor12 ≥ 0.2, many fitness values have |t| &lt; 2; the zero-inflated distribution may affect learning and metrics.
- **Polar effects:** Operon-induced confounding may affect fitness estimates; we do not explicitly model this in the MVP.
- **Library bias:** Short genes and other technical biases are partially addressed by including gene_length (and optionally read-based features) in the model.

---

*End of Data section draft. Revise for length and journal style; move technical implementation details (e.g., exact embedding dimensions, loss weights) to Methods.*
