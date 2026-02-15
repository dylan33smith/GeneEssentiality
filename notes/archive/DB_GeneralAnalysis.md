=== FITNESS BROWSER DATABASE - NON-CONDITION DATA ANALYSIS ===

DATABASE STRUCTURE:
- 86 tables total
- Core: Gene (228,709), Organism (48), Experiment (7,552), GeneFitness (27.4M)
- 48 FitByExp_* tables (per-organism fitness)
- Annotation: KEGG (200K), SEED (177K), SwissProt (79K), Metacyc (39K)
- Domains: PFam (384K), TIGRFam (73K)
- Cross-species: Ortholog (2.8M), Cofit (13.6M)

GENE TABLE:
- 221,005 protein-coding genes
- Fields: orgId, locusId, scaffoldId, begin, end, strand, GC, gene, desc
- Avg gene length: 968 bp
- Avg GC: 59.2%
- 96% have no standard gene name (use locusId)
- Length distribution: 8.9% small (<300bp) - TnSeq bias target

ORGANISM TABLE:
- 48 organisms
- Fields: orgId, division, genus, species, strain, taxonomyId
- Dominated by Proteobacteria (35/48)
- 2 Archaea (Methanococcus)

GENEFITNESS (TARGET VARIABLE):
- 27.4M records (gene × experiment)
- Fields: orgId, locusId, expName, fit (log2 score), t (t-statistic)
- CRITICAL: 94.8% neutral (-1 to 1) - zero-inflated
- Essential (fit < -2): 1.84% of records
- Significant (|t| >= 2): 11.5% of records
- Coverage: ~85% of protein-coding genes have fitness data

GENOME SEQUENCES (ScaffoldSeq):
- 174 scaffolds across 48 organisms
- Full nucleotide sequences available
- Genome sizes: 3-8 Mb

PROTEIN DOMAINS (GeneDomain):
- 456,712 domain annotations
- PFam: 384K domains, 189K genes (85.6%)
- TIGRFam: 73K domains, 64K genes (28.9%)
- Fields: domainId, domainName, begin, end, score, evalue, ec

GENE FEATURES (GeneFeature):
- 781,841 features
- Types: non-cytoplasmic (319K), transmembrane (250K), cytoplasmic (163K), signal peptide (50K)
- From SignalP/TMHMM predictions

FUNCTIONAL ANNOTATIONS:
- BestHitKEGG: 200K genes (90.5% coverage)
- SEEDAnnotation: 177K genes (80.3% coverage)
- BestHitSwissProt: 79K genes (35.8% coverage)
- SEEDClass: 53K genes with EC numbers, 5K with TC numbers

CROSS-SPECIES DATA:
- Ortholog: 2.8M pairs across 48 organisms
- Cofit: 13.6M gene pairs with correlated fitness
- ConservedCofit: 183K conserved co-fitness relationships
- Use for: cluster-based splits, operon detection

SPECIFIC PHENOTYPES:
- 38,525 high-confidence gene-experiment pairs
- Curated essential/important genes
- Use for: validation, confident training examples

COMPOUNDS TABLE:
- 1,223 compounds with MW and CAS numbers
- Use for: SMILES retrieval for chemical embeddings

MEDIA COMPONENTS:
- 10,439 component entries
- Defines composition of ~100 media types
- Example: LB = Tryptone (10 g/L) + Yeast Extract (5 g/L) + NaCl (5 g/L)

PATHWAY DATA:
- MetacycPathwayCoverage: 71K entries, 3,512 pathways
- KEGGMap: 326 pathway maps
- KEGGMember: 83K gene-KO assignments

PUBLICATIONS:
- 20 primary research papers
- Key: Wetmore15, Price18, Liu21

DATA INTEGRITY:
- 100% of GeneFitness records link to Gene table
- 76.5% of protein-coding genes have fitness data
- 87% have domain annotations
- 90.5% have KEGG hits

ML PIPELINE RECOMMENDATIONS:
1. Extract: JOIN Gene + GeneFitness + Experiment with quality filters
2. Features: gene_length, GC, strand, domains, localization
3. Splits: Use Ortholog table for cluster-based cross-validation
4. Class balance: Weight loss for fitness < -1, use AUPRC metric

KEY FIELDS FOR PROJECT:
- Gene: locusId, scaffoldId, begin, end, strand, GC
- GeneFitness: fit (target), t (confidence)
- Experiment: media, condition_1, temperature, aerobic, cor12
- GeneDomain: domainId, domainName (for features)
- GeneFeature: featureType (localization features)
- Ortholog: for cluster-based splits
- Cofit: for operon/module detection