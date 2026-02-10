=== FITNESS BROWSER - COMPREHENSIVE CONDITION ANALYSIS FOR ML ===

PROJECT: Context-Aware Gene Essentiality Prediction
Goal: Predict fitness = f(Genotype, LocalContext, Environment)

DATABASE STATS:
- 7,552 experiments, 48 organisms
- 27.4M fitness records, 169K genes
- 350 unique conditions, 112 media types
- 1,220 unique Media_Condition combinations

CONDITION FIELDS:
- condition_1: Primary treatment (98.6% have concentration data)
- condition_2: Secondary (47% are solvents - DMSO/EtOH)
- condition_3/4: Rare (121/11 experiments)
- Units: mM (69%), mg/ml (18%), vol% (3%), other (10%)

PHYSICAL CONDITIONS:
- Temperature: 30°C (63%), 37°C (17%), 25°C (12%)
- Aerobic: 70% Aerobic, 27% Anaerobic
- Phase: 96% Liquid, 1% Solid
- Shaking: orbital/0rpm/200rpm most common

QUALITY METRICS:
- cor12 (replicate correlation): mean 0.28
- 38% of experiments have cor12 < 0.2 (poor quality)
- Recommend filtering cor12 >= 0.2

FITNESS DISTRIBUTION (CRITICAL):
- 94.8% Neutral (-1 to 1) ← ZERO INFLATION WARNING
- 1.84% Essential (< -2)
- 88.5% not statistically significant (|t| < 2)
Implication: Use weighted loss, AUPRC metric

CONDITION CATEGORIES:
- Stress: 2,854 experiments, 38 organisms
- Carbon source: 1,838 experiments, 36 organisms  
- Nitrogen source: 1,093 experiments, 33 organisms
- Antibiotics: 428 experiments, 36 organisms

CROSS-SPECIES COVERAGE:
- 40 conditions tested in 20+ organisms
- 153 conditions perfectly standardized (same unit + concentration)
- Best: Carbon sources at 20mM (D-Glucose, D-Fructose, etc.)

TOP COMPOSITE CONDITIONS (Media_Condition):
1. LB_Cobalt chloride hexahydrate: 19 organisms
2. LB_Nickel (II) chloride hexahydrate: 18 organisms
3. LB_Sodium nitrite: 18 organisms
4. RCH2_defined_noCarbon_D-Glucose: 17 organisms

MEDIA CLUSTERS:
- LB + RCH2 defined: 24 organisms (Pseudomonas, E. coli, etc.)
- Marine media: 6 organisms
- Methanogen media: 6 organisms
- Shewanella media: 4 organisms

ORGANISM COMPARABILITY:
- Best: 5 P. fluorescens strains (60-86 shared conditions)
- Good: E. coli Keio bridges multiple clusters
- Warn: Media confounds organism effects

RECOMMENDED CONDITION ENCODING:
MVP: Label encode 1,220 Media_Condition combinations
V2: SMILES embeddings for chemicals + learned media embeddings

QUALITY FILTERING PIPELINE:
1. Filter cor12 >= 0.2 (removes 38% low-quality)
2. Filter |t| >= 2 for high-confidence fitness (keeps 11.5%)
3. Exclude condition_2 solvents from primary condition encoding

KEY FIELDS NOT TO MISS:
- temperature, pH: Physical stress modifiers
- aerobic: Critical for anaerobe experiments
- nGenerations: Growth duration (sparse)
- vessel: Microplate vs tube (batch effects)
- mutantLibrary: Control for library bias
- expGroup: Condition category

GENE DATA:
- 221,005 protein-coding genes
- Gene table has: locusId, scaffold, begin, end, strand, GC content
- aaseqs file: FASTA format protein sequences
- For sliding window: sort by scaffold + begin position

RECOMMENDED COMPOSITE STRING FORMAT:
{Media}_{Condition1}_{Concentration}{Units}_{Temperature}C_{Aerobic}
Example: "LB_CobaltChloride_0.2mM_30C_Aerobic"