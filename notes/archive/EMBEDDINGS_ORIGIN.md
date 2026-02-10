# Protein embeddings: origin and format

## File

- **Path:** `data/mvp/embeddings/protein_embeddings.pt`
- **Format:** PyTorch `.pt` file (saved via `torch.save()`).

## Contents (verified)

- **Type:** Python `dict`.
- **Keys:** Gene identifiers as `"orgId:locusId"` (e.g. `"ANA3:7022746"`). Exactly **138,518** keys, one per gene in the MVP gene table.
- **Values:** `torch.Tensor`, shape **`(1152,)`**, dtype `float32`. One vector per gene.

So the file is a mapping from MVP gene ID to a 1152-dimensional protein embedding vector.

## How they were generated

**There is no script or notebook in this repository that creates or references this file.** The following is inferred:

1. **Intended pipeline (from README and notes):** The project plan says to use **ESM-2** (e.g. `esm2_t33_650M_UR50D`) to compute protein embeddings from sequences and store them on disk. The README checklist includes "[MVP] Generate ESM-2 embeddings for this subset."

2. **Dependency:** `requirements.txt` includes **`esm>=3.2.1`** (the `fair-esm` package from Meta), which can encode sequences from a FASTA and produce embeddings.

3. **Likely process:** The embeddings were almost certainly produced by a **one-off script or notebook** (not committed to the repo) that:
   - Loaded MVP protein sequences (e.g. from `data/mvp/proteins.fasta` or per-organism FASTA under `data/mvp/organism_fastas/`),
   - Used the `esm` package to run an ESM-2 model over the sequences,
   - Collected the per-sequence embedding (e.g. mean over sequence tokens or [CLS]-style token),
   - Built a dict `{ "orgId:locusId": tensor, ... }` and called `torch.save(..., "data/mvp/embeddings/protein_embeddings.pt")`.

4. **Dimension 1152:** The README mentions ESM-2 650M with **1280** dimensions. The saved vectors are **1152**-d, so either a different ESM-2 variant/layer was used, or the 650M model’s representation was truncated/aggregated. The exact model and layer used to create this file are not recorded in the repo.

## Current use in code

**No code under `src/` loads or uses this file.** The pipeline currently uses only parquet tables and the media Excel file. To use these embeddings you would add a loader (e.g. in `src/data_io.py`) that calls `torch.load("data/mvp/embeddings/protein_embeddings.pt", map_location="cpu", weights_only=False)` and returns the dict (or a DataFrame/keyed structure).

## How to regenerate

To regenerate embeddings in the same style:

1. Install `esm` (already in `requirements.txt`).
2. Load MVP protein sequences (e.g. from `data/mvp/proteins.fasta` or the FASTA used originally), with headers `>orgId:locusId`.
3. Use the `esm` API to encode (see [fair-esm](https://github.com/facebookresearch/esm) and `scripts/extract.py` in that repo).
4. Build a dict `gene_id -> tensor` and `torch.save()` it to `data/mvp/embeddings/protein_embeddings.pt`.

If you need 1280-d vectors to match the README, use the 650M model (e.g. `esm2_t33_650M_UR50D`) and document the model name in this file or in the script that generates the embeddings.
