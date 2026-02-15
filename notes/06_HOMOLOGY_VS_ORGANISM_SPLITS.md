# Homology-based vs organism-based splits: reasoning

## Context

**ProteomeLM embeddings** are **proteome-contextualized**: each protein's embedding integrates (1) sequence-derived information from ESM-C and (2) the **whole proteomic context**—the set of other proteins in the same organism. See `notes/07_PROTEOMELM_EMBEDDINGS_RESEARCH.md` for details.

**Implication:** The embedding of a gene in organism X reflects its relationship to the entire proteome of X. Orthologs in different organisms (e.g., E. coli *dnaA* vs Pseudomonas *dnaA*) can have different embeddings because they are contextualized by different proteomes.

**Essentiality labels** are gene-level aggregates over experiments (80%/10% rule). The MLP baseline predicts essentiality from the embedding alone—no condition or media information.

---

## Why homology-based splits could be better than organism-based

### 1. Sequence-family leakage (ESM-C base)

The ESM-C component of ProteomeLM produces sequence-derived embeddings. Orthologs (similar sequences) will have similar ESM-C bases. With organism split, held-out organisms contain orthologs of training genes—e.g., Pseudomonas *dnaA* is similar in sequence to E. coli *dnaA*. The model might partly rely on this sequence similarity rather than proteome-contextualized signals.

### 2. Novel sequence space

Homology split ensures **no test gene has >50% identity to any train gene**. We test on genuinely novel sequence families, not orthologs. This tests whether the model learned features that generalize across diverse sequence space.

### 3. Stricter generalization

Homology split forces prediction for genes whose sequences are not similar to training genes. It tests sequence-level generalization, which may be a harder and more conservative evaluation.

---

## Why organism-based splits are used for the MLP baseline

### 1. Proteome-contextualized embeddings change the picture

With ProteomeLM, **organism split = proteome split**. When we hold out an organism, we hold out an entire proteome. All embeddings from that organism were computed in a proteomic context the model never saw. We are testing: **can the model predict essentiality for proteins in a novel proteomic context?** This is a meaningful and appropriate generalization test for ProteomeLM—the embeddings themselves are organism/proteome-specific.

### 2. Ortholog "leakage" is mitigated

Orthologs in different organisms have embeddings contextualized by different proteomes. ProteomeLM adds organism-specific context on top of the ESM-C base. So Pseudomonas *dnaA* and E. coli *dnaA* may have different ProteomeLM embeddings despite similar sequences—the proteome context can make them distinct. Organism split therefore tests generalization to novel proteomes, not just novel sequences.

### 3. Aligns with ProteomeLM-Ess evaluation

The ProteomeLM paper evaluates ProteomeLM-Ess by holding out entire proteomes (e.g., E. coli, S. cerevisiae) and reports strong generalization. Our organism-based split follows the same evaluation paradigm.

### 4. Simpler implementation

No MMseqs2 or clustering required. Split by orgId.

### 5. Practical question

"Can we predict essentiality for a new bacterial species?"—directly relevant for deployment when encountering new organisms.

---

## Summary

| Split type | Tests | Trade-off |
|------------|-------|-----------|
| Organism | Generalization to novel proteomes (new species) | Aligns with ProteomeLM design; simpler |
| Homology | Generalization to novel sequence families | Stricter sequence-level test; requires MMseqs2 |

For the current MLP baseline, **organism split** is used. It is well-suited to ProteomeLM embeddings, which are proteome-contextualized. Homology-based evaluation remains an option for future, sequence-level generalization assessment.
