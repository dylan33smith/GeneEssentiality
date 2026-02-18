# ProteomeLM: What It Does and What the Embeddings Represent

Research summary based on the ProteomeLM paper (bioRxiv 2025.08.01.668221), GitHub repository, and related documentation.

---

## 1. Overview

**ProteomeLM** is a transformer-based language model that reasons on **entire proteomes** from species spanning the tree of life. Unlike protein language models (ESM-2, ESM-C) that operate on individual sequences, ProteomeLM learns **contextualized** protein representations by leveraging the **whole proteomic context**.

**Key distinction:** ProteomeLM embeddings are **not** sequence-only. They integrate (1) sequence-derived information from ESM-C and (2) **proteome-scale context**—the set of other proteins in the same organism.

---

## 2. Architecture and Pipeline

### 2.1 Two-stage process

1. **ESM-C encoding:** Each protein's amino acid sequence is embedded by **ESM-Cambrian (ESM-C)** 600M, a protein language model from EvolutionaryScale. This yields a fixed-dimensional embedding per protein—sequence-derived, capturing structural and functional signals from the sequence alone.

2. **ProteomeLM contextualization:** The set of ESM-C embeddings (one per protein in the proteome) is fed into ProteomeLM. ProteomeLM is trained as a **masked language model**: a subset of protein embeddings is masked, and the model must reconstruct them using the **remaining unmasked protein embeddings from the same proteome**. By learning this reconstruction task, ProteomeLM learns dependencies between proteins encoded by a given genome.

### 2.2 What the output represents

The final embedding of protein A in organism X is:

- **Base:** ESM-C embedding of A's sequence (sequence-derived)
- **Plus:** Information from the entire proteome of X—i.e., how A relates to all other proteins in that organism (inter-protein dependencies, functional constraints, coevolution at the proteome level)

**Critical implication:** The embedding of the same protein (or a close ortholog) in two different organisms will differ, because the proteomic context differs. E. coli *dnaA* and Pseudomonas *dnaA* may have similar ESM-C bases (similar sequence) but different ProteomeLM outputs (different proteome context).

---

## 3. Training Objective and Functional Encoding

- **Training:** ProteomeLM is trained to reconstruct masked protein embeddings from the context of the rest of the proteome. A custom polar loss minimizes the difference between ESM-C and ProteomeLM embeddings in a protein-family–specific manner.

- **No positional encoding:** ProteomeLM does not use genomic position (unlike genome language models). Gene order is not encoded.

- **Functional encoding:** Uses **OrthoDB orthologous groups** to capture evolutionary/functional relationships across genes. This helps the model learn coevolution between proteins (e.g., presence/absence patterns across genomes).

---

## 4. Emergent Capabilities

- **Protein–protein interactions (PPI):** Attention coefficients spontaneously capture PPI without explicit PPI training. The model learns which proteins "attend to" each other when reconstructing masked proteins.

- **Gene essentiality:** ProteomeLM-Ess is a supervised predictor (two-layer MLP) that takes ProteomeLM embeddings and predicts essentiality. It **outperforms** classifiers based on ESM-C alone, demonstrating that proteome-contextualized information improves essentiality prediction. Best performance uses embeddings from intermediate layers (e.g., layer 8 of ProteomeLM-L).

- **Cross-taxon generalization:** ProteomeLM-Ess generalizes to held-out proteomes (e.g., E. coli, S. cerevisiae, synthetic cells JCVI-Syn1.0, JCVI-Syn3A) that were not in the training set.

---

## 5. What Our Script Produces

From the `generate_embeddings` pipeline step (`src/pipeline/generate_embeddings.py`), run via `scripts/run_pipeline.py`:

- **Input:** Single-organism FASTA (all proteins from one genome)
- **Process:** ESM-C encodes each sequence → ProteomeLM processes the full set of embeddings (whole proteome) → outputs contextualized embeddings
- **Output:** `output.logits` — one embedding per protein, aligned to `group_labels` (orgId:locusId). Stored in `data/mvp/ProtLM_embedddings/` (one .pt per organism). With y labels: the `label_embeddings` step produces `data/mvp/ProtLM_embeddings_with_labels/`.

The embeddings are **proteome-contextualized**: each gene's embedding reflects its relationship to the rest of that organism's proteome.

---

## 6. Implications for Our Project

### 6.1 Embeddings are organism-specific

A gene's ProteomeLM embedding is tied to the organism's proteome. The same gene in two organisms (or orthologs) can have different embeddings because the proteomic context differs.

### 6.2 Organism-based split

When we hold out an organism, we hold out an entire proteome. All embeddings from that organism were computed in a proteomic context the model never saw during training. We are testing: **can the model predict essentiality for proteins in a novel proteomic context?** This is a meaningful generalization test for ProteomeLM.

### 6.3 Homology vs organism split (revised)

**Previous (incorrect) assumption:** Similar sequences → similar embeddings → organism split allows "ortholog leakage" (Pseudomonas dnaA similar to E. coli dnaA in training).

**Revised understanding:** With ProteomeLM, orthologs in different organisms have embeddings contextualized by different proteomes. The proteome-specific context can make their embeddings distinct. Organism split therefore tests generalization to **novel proteomes**, not just novel sequences. Homology split tests generalization to **novel sequence families**; both are valid but answer different questions.

### 6.4 Why ProteomeLM may help essentiality

Essentiality depends on protein function, interactions, and cellular context. ProteomeLM embeddings encode inter-protein dependencies and proteome-scale functional constraints—exactly the kind of information that could inform essentiality. The paper shows ProteomeLM-Ess outperforms ESM-C–based classifiers, supporting this.

---

## 7. References

- Malbranke, C., Zalaffi, G.P., Bitbol, A.-F. (2025). ProteomeLM: A proteome-scale language model allowing fast prediction of protein-protein interactions and gene essentiality across taxa. bioRxiv 2025.08.01.668221.
- GitHub: https://github.com/Bitbol-Lab/ProteomeLM
- Hugging Face: Bitbol-Lab/ProteomeLM-S (and XS, M, L)
- ESM-C: EvolutionaryScale ESM Cambrian (protein language model)
