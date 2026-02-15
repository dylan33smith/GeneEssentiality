PROJECT PROTOCOL: DeepConditional (v2.0)
Title: Context-Aware Prediction of Bacterial Gene Essentiality

### 1. EXECUTIVE SUMMARY & DATA STRATEGY
------------------------------------
Objective: Predict quantitative gene fitness (t-scores) based on protein sequence and environmental conditions.

CRITICAL INSIGHT FROM DATA ANALYSIS:
The database contains distinct "Ecosystem Silos" where media usage does not overlap (e.g., Marine Broth is never used for E. coli). 
- Risk: A model trained on all data may rely on "Media Shortcuts" (e.g., predicting phenotypes based solely on the media type rather than the stressor).
- Strategy: The project is split into a "Connected MVP" (high-overlap data) and a "Universal V2" (full sparse data).

### 2. PHASE I: THE "CONNECTED" MVP
-------------------------------
Goal: Prove the "Genotype + Stress = Phenotype" hypothesis in a controlled data environment.

2.1 Data Selection (The "Terrestrial" Subset)
Instead of using all 48 organisms, the MVP will restrict training to the largest "Connected Component" of the database:
* Inclusion Criteria:
  - Organisms using 'LB', 'RCH2_defined', or 'M9' media.
  - This covers ~24 organisms (Pseudomonas, E. coli, Klebsiella, etc.) and ~60% of total experiments.
* Justification:
  - These organisms share media types, forcing the model to look at the GENE differences to explain fitness, not just the media differences.

2.2 Condition Representation (MVP Level)
* Feature: Hierarchical Multi-Hot Encoding.
  - Vector A (Media): One-Hot encoding of the 15-20 relevant media types (LB, RCH2, etc.).
  - Vector B (Stress): One-Hot encoding of the ~100 distinct chemical stressors in this subset.
  - Input = Concatenate(Vector A, Vector B).

### 3. PHASE II: ARCHITECTURE & TRAINING
------------------------------------
Goal: A Transformer that fuses gene (protein) representation with environmental context.

3.1 The Model: "Context-Aware Transformer"
* Input Dimensions: (Batch_Size, Sequence_Length=2, Embedding_Dim=1280)
* Token Structure:
  - Token 0: [Condition_Embedding] (The "Query" Context)
  - Token 1: [Gene_Embedding] (The target gene's protein embedding)
* Mechanism:
  - The Self-Attention layers allow Token 0 (Condition) to attend to the gene embedding.
  - Example: If Condition="Antibiotic X", the model learns condition-specific fitness from the gene's sequence representation.

3.2 Representation (Pre-Computed)
* Protein Encoder: ESM-2 (esm2_t33_650M_UR50D).
  - Action: Run inference ONCE. Save vectors to disk. Do not fine-tune (too expensive for MVP).
* Metadata Injection:
  - Concatenate [Gene_Length, Normalized_Reads] to the protein vector to correct for library bias.

3.3 Training Objectives
* Primary Loss: Weighted MSE.
  - Weight samples with Fitness < -1.0 (Essential) by 5x.
  - Weight samples with Fitness > -0.5 (Neutral) by 1x.
* Metric:
  - AUPRC (Area Under Precision-Recall Curve) for detecting essential genes.

### 4. PHASE III: VALIDATION (THE "HONEST" SPLIT)
---------------------------------------------
Goal: Ensure the model isn't just memorizing homologous genes.

4.1 Homology Splitting
* Protocol:
  1. Cluster all proteins in the subset at 50% Identity (using MMseqs2).
  2. Assign Clusters to Train/Val/Test (70/15/15).
  3. Strict Rule: No sequence in Test can have >50% identity to any sequence in Train.

4.2 The "Generalization" Benchmarks
* Benchmark A (Chemical Generalization):
  - Hold out "Sodium Nitrite" experiments entirely.
  - Can the model predict Nitrite sensitivity based on generic "Stress" patterns?
* Benchmark B (Organism Generalization):
  - Hold out "Pseudomonas putida" entirely.
  - Can the model predict its fitness using only E. coli/Klebsiella training data?

### 5. PHASE IV: FUTURE DEVELOPMENT (V2 EXPANSION)
----------------------------------------------
Goal: Expand to the full database and "Zero-Shot" chemical prediction.

5.1 Handling the "Silos" (Marine/Anaerobes)
* Action: Introduce "Phylogenetic Embeddings."
  - Add a token representing the organism's evolutionary distance (e.g., 16S rRNA embedding).
  - This helps the model adjust for the fact that Marine bacteria have different baselines than Gut bacteria.

5.2 Chemical "First Principles" (The SMILES Upgrade)
* Limitation of MVP: It treats "Copper" and "Zinc" as random IDs (0 and 1). It doesn't know they are both metals.
* Upgrade:
  - Replace One-Hot Stress vectors with Molecular Graph Embeddings (e.g., ChemBERTa or Mol2Vec).
  - Why: Enables "Zero-Shot" prediction. The model can predict toxicity for a *new* chemical if it is structurally similar to a known one.

### 6. EXECUTION CHECKLIST (PRIORITIZED)
------------------------------------

STEP 1: Data Engineering (Week 1-2)
[MVP] Write SQL to extract the "Terrestrial Subset" (Organisms using LB/RCH2).
[MVP] Generate ESM-2 (or ProteomeLM) embeddings for this subset.
[MVP] Create "Cluster Splits" (50% identity).

STEP 2: Baseline Modeling (Week 3)
[MVP] Train a simple "Mean Regressor" (predicts average fitness of the gene across all conditions). *This is your baseline to beat.*
[MVP] Train a simple "Random Forest" (using just gene features).

STEP 3: Transformer Development (Week 4-5)
[MVP] Build the 'ContextTransformer' class in PyTorch.
[MVP] Implement the Weighted MSE loss.
[MVP] Train on the Terrestrial Subset.

STEP 4: Analysis & V2 Planning (Week 6+)
[Report] Measure performance on "Held Out Organism" (Pseudomonas putida).
[V2] Scrape SMILES strings for all 350 conditions.
[V2] Integrate Marine/Anaerobe data.

### 7. RISK MANAGEMENT
------------------
* Risk: "Library Bias" (Small genes look essential because they get fewer hits).
  - Mitigation: Ensure 'gene_length' is a scalar input to the final MLP head.
* Risk: "Media Confounding" (Model learns LB = Fitness X, regardless of gene).
  - Mitigation: The "Terrestrial Subset" strategy minimizes this by ensuring multiple organisms share the same media.