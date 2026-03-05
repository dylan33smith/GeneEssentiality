# Lab Presentation: Context-Aware Prediction of Bacterial Gene Essentiality

**Synthesis for lab meeting. Overly detailed—trim as needed.**

---

## 1. Background: Transposon Insertion Sequencing (Tn-Seq)

### What problem does Tn-Seq address?

- We want to know which genes are **essential** for a bacterium in a given environment (e.g. LB, stress, host).
- “Essential” = the organism does not tolerate loss of that gene (no or very few viable colonies; gene is required for growth/survival under that condition).
- Testing every gene one-by-one with knockouts is not scalable. Tn-Seq provides a **genome-wide** readout in a single experiment.

### How Tn-Seq works (high level)

1. **Library construction:** A transposon (e.g. Mariner, Himar1) is used to create a pool of mutants: random insertions across the genome. Many cells, each with one (or few) insertions; collectively, insertions cover (in principle) the whole genome.
2. **Growth under a condition:** The pool is grown in a chosen environment (e.g. LB, M9, or LB + stressor). Cells that lost an essential gene die or are strongly outcompeted; cells with insertions only in non-essential regions survive and multiply.
3. **Sequencing:** DNA is extracted and the insertion sites are sequenced (e.g. by sequencing the junction between the transposon and the genome). You get **counts** of insertions per site (or per gene, after aggregation).
4. **From counts to fitness:**  
   - **Many insertions in a gene** → that gene tolerates disruption → **non-essential** in that condition.  
   - **Few or no insertions** → the gene is required for growth → **essential** in that condition.  
   Fitness is often reported as a **log₂ ratio** (e.g. compared to a reference or input pool): strong negative values indicate depletion (essential), values near zero indicate no strong effect (non-essential).

### Why this matters for the project

- Tn-Seq gives **condition-specific** essentiality: the same gene can be essential in one medium and non-essential in another (e.g. a nutrient biosynthesis gene only in minimal medium).
- Our **labels** come from aggregating these experiment-level fitness signals to a **gene-level** essentiality class (always essential, conditionally essential, non-essential) using rules on fraction of experiments where the gene was “essential” (see Data section).
- All of this is **no new wet-lab**: we use existing public Tn-Seq–derived fitness data from the Fitness Browser database.

---

## 2. Data

### Source: Fitness Browser (feba.db)

- **What it is:** A public resource that aggregates genome-wide gene fitness measurements from **transposon insertion sequencing (Tn-Seq)** across many bacterial species and growth conditions.
- **What we use:** Processed exports from that database—no new experiments. The pipeline reads from (or exports derived from) **feba.db** and produces Parquet tables, FASTA files, and condition vocabularies.

### Two data subsets

| Subset        | Organisms | Experiments | Genes   | Fitness records  | Use case                          |
|---------------|-----------|-------------|---------|------------------|-----------------------------------|
| **Processed** | 48        | 7,552       | 221,005 | 27,410,721       | Full DB; reference / future use   |
| **MVP**       | 27        | 2,164       | 138,518 | 8,688,562        | Training and development          |

- **MVP (Minimum Viable Product):** A “connected” subset so that organisms **share growth conditions** (media).  
  - **Inclusion:** Organisms that use at least one of **LB**, **RCH2_defined**, or **M9** media.  
  - **Quality filter:** Experiments with replicate correlation **cor12 ≥ 0.2** only (replicate agreement).  
- **Rationale:** In the full DB, media usage is siloed (e.g. “Marine Broth” only for some species). Training on everything risks learning “media shortcuts” (predicting phenotype from media type rather than gene × condition). The MVP forces overlap so the model must use **gene** (and eventually condition) information.

### Core tables (Parquet, after extraction from feba.db)

- **genes.parquet:** One row per gene. Key columns: `orgId`, `locusId`, `sysName`, genomic coordinates, and **aggregated fitness stats** used to assign essentiality class (see below).
- **experiments.parquet:** One row per experiment. Columns include `orgId`, `expName`, `media`, `condition_1`, `temperature`, `cor12`, etc. Defines *which* condition each fitness value comes from.
- **fitness.parquet:** One row per (gene, experiment). Columns: `orgId`, `locusId`, `expName`, **fit** (log₂ fitness), **t** (t-statistic for the estimate).  
  - **fit &lt; −1** → gene is treated as **essential in that experiment**.  
  - **|t| ≥ 2** → measurement is **confident**.
- **organisms.parquet:** One row per organism (orgId, taxonomy, etc.).

### Essentiality classification (80% / 10% rule)

Gene-level labels are **not** raw Tn-Seq counts; they are derived by aggregating fitness across experiments and applying fixed rules.

- **Per experiment:**  
  - **Confident:** |t| ≥ 2.  
  - **Essential in that experiment:** fit &lt; −1.
- **Per gene:**  
  - Compute **frac_essential_confident** = fraction of *confident* experiments in which the gene was essential (fit &lt; −1).  
  - **always_essential:** frac_essential_confident **&gt; 0.8** (essential in &gt;80% of confident experiments).  
  - **conditional:** 0.1 ≤ frac_essential_confident ≤ 0.8 (essential in 10–80% of experiments).  
  - **non_essential:** frac_essential_confident **&lt; 0.1** (and at least one confident experiment).  
  - **no_data:** no confident experiments (n_confident == 0).

These thresholds (0.8, 0.1, fit &lt; −1, |t| ≥ 2) are fixed in the codebase and applied consistently. The pipeline aggregates raw fitness to gene-level stats and then assigns **essentiality_class**; that class is what we predict (or a binarized version: essential vs non-essential).

### Pipeline steps (from raw DB to model inputs)

1. **extract_data:** Read feba.db → write Parquet (genes, experiments, fitness, organisms) under `data/raw` or `data/processed`.
2. **classify_genes:** Compute gene-level stats and essentiality_class (80%/10%) for the chosen subset (processed or MVP).
3. **filter_mvp:** Restrict to MVP organisms and apply filters (media prefixes LB/RCH2/M9, min cor12, excluded groups/media).
4. **create_fastas:** Build per-organism FASTA files (amino acid sequences) for MVP (and optionally processed) from the sequence source (e.g. `aaseqs`).
5. **generate_embeddings:** Run ProteomeLM (see below) on each organism FASTA → one `.pt` file per organism with `embeddings` and `group_labels` (gene IDs).
6. **label_embeddings:** Join essentiality_class to each embedding by gene ID; add **y** (0=always_essential, 1=conditional, 2=non_essential, −1=no_data) to the `.pt` files. Output: `data/mvp/ProtLM_embeddings_with_labels_layer8/` (or similar).

All paths and thresholds are configured in `config/pipeline.yaml` and `config/model.yaml` / `config/model_binary.yaml`.

---

## 3. ProteomeLM Synopsis

### What ProteomeLM is

- **ProteomeLM** is a transformer-based model that reasons over **entire proteomes** (all proteins of a species), not single sequences.
- It produces **contextualized** protein representations: each protein’s embedding depends on its **sequence** and on the **rest of that organism’s proteome**.

### Two-stage pipeline (how we get embeddings)

1. **ESM-C (ESM-Cambrian, 600M):** Each protein’s **amino acid sequence** is encoded by ESM-C into a fixed-dimensional vector. This part is sequence-only (structure/function from sequence).
2. **ProteomeLM:** The **set** of ESM-C vectors for one organism (one per protein) is fed into ProteomeLM. The model was trained as a **masked language model** over these “protein tokens”: some are masked, and the model reconstructs them from the others. So each output embedding incorporates **information from the whole proteome** (coevolution, functional context, interactions).

### What the embedding represents

- For a protein A in organism X:  
  - **Base:** ESM-C embedding of A’s sequence.  
  - **Plus:** Information from the entire proteome of X—how A relates to all other proteins in that organism.
- **Consequence:** The same gene (or ortholog) in two different species can have **different** ProteomeLM embeddings, because the proteomic context differs. So embeddings are **organism-/proteome-specific**.

### Why we use it for essentiality

- Essentiality depends on function, interactions, and cellular context. ProteomeLM embeddings encode **inter-protein dependencies** and proteome-scale context, which are relevant for “is this gene required?”
- The ProteomeLM paper (Bitbol-Lab) shows that an essentiality predictor (ProteomeLM-Ess) on top of ProteomeLM embeddings **outperforms** classifiers based on ESM-C alone and generalizes to held-out proteomes (e.g. E. coli, yeast). We use the same idea: embeddings → classifier.

### Technical choices in our pipeline

- **Model:** ProteomeLM-L (Hugging Face: `Bitbol-Lab/ProteomeLM-L`). ESM-C 600M then ProteomeLM.
- **Layer:** We extract embeddings from **layer 8** (config: `hidden_layer: 8`). The paper suggests intermediate layers work well for essentiality; we use a fixed layer for reproducibility.
- **Input:** One FASTA per organism (all proteins). Output: one `.pt` per organism with `(N, D)` embeddings and `group_labels` (e.g. `orgId:locusId`). **Label step:** We add **y** (0/1/2/−1) from our gene-level essentiality_class so each embedding has a class label for training.

---

## 4. Overview of the Model

### Task

- **Input:** One ProteomeLM embedding per gene (from the labeled `.pt` files).  
- **Output:** A **discrete essentiality class**.  
  - **3-class:** always_essential (0), conditional (1), non_essential (2).  
  - **Binary (optional):** essential (0) = always_essential + conditional merged; non_essential (1).

We do **not** (in the current baseline) use condition or media as input; we predict gene-level class from the embedding only. Condition-aware prediction is a planned extension (README / protocol).

### Architecture: EssentialityMLP

- **Structure:** Fully connected MLP: embedding → hidden layers → logits.
  - **Default hidden_dims:** [512, 256] (two hidden layers). Configurable (e.g. [256, 128] or [512, 256, 128] for depth).
  - Each hidden: Linear → ReLU → Dropout (default 0.2).
  - Final: Linear(hidden_last → n_classes). n_classes = 3 or 2.
- **Input dim:** Matches embedding dimension from ProteomeLM (e.g. 1152 for ProteomeLM-L layer 8).
- **Output:** Raw logits for each class; training uses CrossEntropyLoss (with optional class weights).

### Training setup

- **Loss:** CrossEntropyLoss. **Class weights:** Inverse-frequency (from training set) to handle imbalance (few always_essential/conditional, many non_essential).
- **Optimizer:** AdamW (default lr 1e-3, weight_decay 1e-4).
- **Scheduler (optional):** ReduceLROnPlateau on validation metric (loss or essential AUPRC).
- **Early stopping:** Validation metric (e.g. **essential-class AUPRC**) no improvement for N epochs (e.g. 15); best checkpoint saved by that metric.
- **Train/val/test:** **Organism-based split** (no organism appears in more than one split). Same split for 3-class and binary:
  - **Train:** 19 organisms (e.g. ANA3, Cup4G11, Dda3937, …).
  - **Val:** 4 (e.g. WCS417, Putida, DdiaME23, Burk376).
  - **Test:** 4 (e.g. Koxy, psRCH2, Korea, BFirm).

So we evaluate **generalization to unseen proteomes** (new species), which matches the ProteomeLM design (proteome-contextualized embeddings).

### Metrics

- **Accuracy:** Fraction of genes with correct predicted class.
- **Weighted F1:** Per-class F1, averaged with weights = class frequencies (handles imbalance).
- **AUROC (per class):** One-vs-rest AUROC; how well the model ranks that class vs others.
- **AUPRC (per class):** Area under the precision–recall curve for that class (one-vs-rest). **AUPRC for the essential class (or always_essential in 3-class)** is the primary metric we care about and use for model selection (best checkpoint, early stopping).

### Implementation and reproducibility

- **Script:** `scripts/train_model.py` (CLI: config path, epochs, batch size, lr, dropout, early stopping, scheduler, output name).  
- **Notebook:** `model.ipynb` for interactive runs, plots, and **parameter sweeps** (lr, dropout, weight decay, hidden_dims, batch size, scheduler, class weighting).
- **Configs:** `config/model.yaml` (3-class), `config/model_binary.yaml` (binary, same splits, label_map to merge always_essential + conditional → essential).

---

## 5. Comparison: ProteomeLM-Ess (paper) vs our pipeline

The following is extracted from the ProteomeLM paper (bioRxiv 2025.08.01.668221; `~/projects/ProteomeLM/paper.pdf`) and compared to our GeneEssentiality setup.

### 5.1 Datasets

| Aspect | ProteomeLM-Ess (paper) | Our pipeline (GeneEssentiality) |
|--------|------------------------|----------------------------------|
| **Source** | **OGEE database** [66]: aggregates gene essentiality from **127 experimental studies** across **91 species**. **213,608 labeled genes** total. | **Fitness Browser (feba.db)** only: Tn-Seq fitness data. MVP = **27 organisms**, ~**138k genes**, ~8.7M fitness records. |
| **Organisms / taxa** | 91 species (OGEE); they use **87 taxonomic IDs** after matching sequences (UniProt, NCBI, SGD, **Fitness Browser** [92]). | 27 bacteria (MVP: shared media LB/RCH2/M9, cor12 ≥ 0.2). |
| **How essentiality is determined** | **Study-dependent:** OGEE curates results from many different experiments (knockout screens, Tn-Seq, etc.). Each study has its own protocol and definition of “essential.” Labels are **binary** (essential vs non-essential) as provided by each study. | **Single rule:** We aggregate **Tn-Seq fitness** (fit, t) to gene level and apply the **80%/10% rule**: fraction of *confident* experiments (\|t\| ≥ 2) where fit &lt; −1 → always_essential (&gt;80%), conditional (10–80%), non_essential (&lt;10%). So essentiality is **uniformly defined** across all our genes (condition-aggregated). |
| **Label granularity** | Binary (essential / non-essential). | We support **3-class** (always_essential, conditional, non_essential) and **binary** (essential = always + conditional, non_essential). |

**Summary:** The paper uses a **large, heterogeneous** essentiality dataset (many studies, many species, mixed protocols). We use a **smaller, homogeneous** dataset (one database, one type of evidence (Tn-Seq), one aggregation rule, bacteria only).

### 5.2 Train / validation / test split

| Aspect | ProteomeLM-Ess (paper) | Our pipeline |
|--------|------------------------|-------------|
| **Split criterion** | **Sequence-similarity clustering** (MMSeqs2, **40% similarity**). Proteins in the same cluster are assigned to the **same** split (all in train, or all in val, or all in test). Split is at the **protein level** across all 83 training genomes. | **Organism-based.** Train / val / test are disjoint **organisms** (e.g. 19 / 4 / 4). Every gene from a test organism is in the test set. |
| **Goal of split** | Avoid the model using **sequence similarity** as a shortcut (orthologs in different species sharing the same cluster cannot be in different splits). | Evaluate **generalization to unseen proteomes** (new species). No protein from a test organism is seen at training time. |
| **Held-out for evaluation** | **5 genomes** held out from training: *S. cerevisiae* and **4 *E. coli* strains**. Plus JCVI-Syn1.0 and JCVI-Syn3A (synthetic cells, not in OGEE) for generalization. | 4 test organisms (e.g. Koxy, psRCH2, Korea, BFirm); 4 val organisms. |

**Summary:** They test generalization when **sequence-similar proteins** are withheld (homology-based). We test generalization when **entire proteomes** are withheld (organism-based). Both are valid but answer different questions.

### 5.3 Classification model

| Aspect | ProteomeLM-Ess (paper) | Our pipeline |
|--------|------------------------|-------------|
| **Architecture** | **Two-layer** fully connected: input (embedding) → **hidden size 2048** → ReLU → **dropout 0.5** → output **2 logits** (binary). | **MLP** with configurable **hidden_dims** (default **[512, 256]**), ReLU, **dropout 0.2**, → **2 or 3 logits**. |
| **Input preprocessing** | **Genome-wide normalization:** embeddings are normalized using the **mean and standard deviation computed over the proteome** before being fed to ProteomeLM-Ess. | **No** per-proteome normalization; we use embeddings as produced by ProteomeLM. |
| **Embedding layer** | Can use embeddings from **any** ProteomeLM layer; **best: layer 8 of ProteomeLM-L**. Same as we use. | We use **layer 8** of ProteomeLM-L (config: `hidden_layer: 8`). |
| **Loss** | Binary classification (essential vs non-essential); paper says “training procedure and objective” same as ProteomeLM-PPI (binary cross-entropy with early stopping on validation). | **CrossEntropyLoss** with **inverse-frequency class weights** (imbalanced classes). Optional **ReduceLROnPlateau**; early stopping on **validation AUPRC** (essential class) or loss. |
| **Best reported performance** | **AUROC 0.93** (layer 8, ProteomeLM-L). On held-out *E. coli*: 71% of essential genes predicted essential, 2% of non-essential predicted essential. | **Test essential AUPRC ~0.41–0.44** (binary); test accuracy ~72–75%. AUPRC for non_essential ~0.91–0.95. |

**Summary:** Their classifier is **larger** (one hidden layer of 2048, dropout 0.5) and uses **genome-wide normalized** embeddings; they report **binary** essentiality and **AUROC 0.93** on their split. We use a **smaller** MLP (e.g. 512→256), **no** embedding normalization, **class-weighted** loss, and report **AUPRC** and accuracy on an **organism-held-out** test set, with lower essential AUPRC (~0.41–0.44). Differences in **dataset**, **split**, and **evaluation metric** (AUROC vs AUPRC, different test sets) prevent a direct numerical comparison.

### 5.4 Takeaways

- **Datasets:** OGEE = many studies, many species, mixed definitions; we = one Tn-Seq DB, one rule, bacteria-only, finer-grained (3-class possible).
- **Splits:** Paper = homology-aware (cluster by sequence); we = organism-based (unseen proteomes).
- **Model:** Paper = larger MLP + per-proteome normalization; we = smaller MLP, no normalization, class weights, AUPRC-driven early stopping.
- **Metrics:** Paper emphasizes AUROC; we emphasize **AUPRC for the essential class** (better for imbalanced data and for prioritizing finding essential genes).
- **Reproducibility:** Our pipeline is fully specified (config, 80/10 rule, organism split); OGEE and their exact MMSeqs2 split would need to be reproduced for a strict like-for-like comparison.

---

## 6. Synthesis of Results

### Data scale (MVP, labeled embeddings)

- **Train:** ~78k–79k genes (19 organisms); **Val:** ~17k–18k; **Test:** ~16k–17k (numbers can vary slightly with exact pipeline/version).
- **Class balance (typical):** Strong imbalance toward non_essential (e.g. ~82–86% non_essential in train/val/test for binary; always_essential and conditional are minority in 3-class). Hence class-weighted loss and reporting AUPRC for the rare class.

### 3-class model (always_essential, conditional, non_essential)

- **Training:** Validation loss often increases after early epochs while train loss keeps decreasing (overfitting to train organisms). Early stopping and checkpoint selection by **validation AUPRC (always_essential)** are used to pick the best model.
- **Typical validation metrics (illustrative):** Accuracy in the ~0.72–0.82 range; weighted F1 similar; **AUPRC for always_essential** in the ~0.42–0.45 range; AUPRC for non_essential much higher (~0.94–0.95) because that class is abundant and easier to predict.
- **Interpretation:** The model learns to separate non_essential from essential reasonably well; discriminating **always_essential** from **conditional** is harder (both are “essential in some sense”), which is reflected in the lower AUPRC for the always_essential class.

### Binary model (essential vs non_essential)

- **Setup:** Same embeddings and organism split; labels remapped so essential = always_essential + conditional, non_essential unchanged.
- **Typical results (from runs in this repo):**  
  - **Test accuracy:** ~0.72–0.75.  
  - **Test weighted F1:** ~0.75–0.76.  
  - **Test AUPRC (essential):** ~0.41–0.44.  
  - **Test AUPRC (non_essential):** ~0.91–0.95.  
- Again, **essential** is the harder class (minority, and we care most about it); **non_essential** is predicted with high precision/recall.

### What the numbers suggest

- **ProteomeLM embeddings + MLP** give a usable baseline: clear separation of non_essential vs essential (binary) and reasonable 3-class accuracy, with best checkpoint chosen by essential-class AUPRC.
- **Generalization to held-out organisms** is non-trivial (validation loss rises with more training), but early stopping and AUPRC-based selection yield a model that generalizes to the test set at the levels above.
- **Limitations:** (1) No condition input yet—we predict a single gene-level class, not “essential in condition C.” (2) Essential-class AUPRC (~0.41–0.45) leaves room for improvement (architecture, data, or features). (3) Class imbalance and the cost of missing essential genes may warrant further tuning (thresholds, weighting, or metrics).

### Next steps (from README / protocol)

- **Condition-aware prediction:** Add media/condition as input (e.g. multi-hot or embeddings) so the model can predict “essential in condition C” (future phase).
- **MLP improvements:** Deeper/wider nets, batch size, scheduler, class weighting—all are available in the notebook sweep for systematic comparison.
- **Risks to mitigate:** Library bias (gene length), media confounding (partially addressed by MVP and organism split).

---

## 7. One-slide / one-minute version (if you need to trim to the minimum)

- **Goal:** Predict bacterial gene essentiality (and eventually fitness by condition) from genotype and context.
- **Data:** Fitness Browser (Tn-Seq); MVP = 27 organisms, shared media (LB/RCH2/M9), ~8.7M fitness records → gene-level essentiality (80%/10% rule).
- **Representation:** ProteomeLM embeddings (sequence + whole-proteome context), one per gene, layer 8.
- **Model:** MLP (embedding → 3-class or binary). Organism split; early stopping and best checkpoint by essential-class AUPRC.
- **Results:** Test accuracy ~72–75% (binary), essential AUPRC ~0.41–0.44; non_essential AUPRC high (~0.91–0.95). Baseline works; condition-aware model and further tuning are next.

---

*End of synthesis. Remove or shorten sections as needed for time.*
