#!/usr/bin/env python3
"""
Generate ProteomeLM embedding vectors from a single-organism FASTA.

Uses the ProteomeLM package: ESM-C 600M for sequence encoding, then the
ProteomeLM transformer to produce per-gene contextualized embeddings.
Output .pt contains embeddings and group_labels (orgId:locusId) for joining
to genes.parquet / fitness / experiments.

Run from repo root with ProteomeLM on PYTHONPATH. Install ProteomeLM dependencies
(see ProteomeLM/requirements.txt), e.g. pip install -r ProteomeLM/requirements.txt.

  PYTHONPATH=ProteomeLM:$PYTHONPATH python scripts/generate_proteomelm_embeddings.py --input data/mvp/organism_fastas/Keio.fasta --output data/mvp/embeddings/Keio_proteomelm.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
proteomelm_root = project_root / "ProteomeLM"
if proteomelm_root.is_dir() and str(proteomelm_root) not in sys.path:
    sys.path.insert(0, str(proteomelm_root))

import torch

from proteomelm.utils import build_genome_esmc
from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ProteomeLM embeddings from a single-organism FASTA",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input FASTA path (one organism; headers should be orgId:locusId)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .pt path (embeddings + group_labels)",
    )
    parser.add_argument(
        "--proteomelm-model",
        type=str,
        default="Bitbol-Lab/ProteomeLM-S",
        help="ProteomeLM model: HuggingFace ID or local path (default: Bitbol-Lab/ProteomeLM-S)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for ProteomeLM and ESM-C (default: cuda:0)",
    )
    args = parser.parse_args()

    fasta_path = args.input.resolve()
    if not fasta_path.is_file():
        print(f"Error: input FASTA not found: {fasta_path}", file=sys.stderr)
        return 1

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU", file=sys.stderr)
        device = "cpu"

    print("Step 1: ESM-C 600M encoding...")
    data = build_genome_esmc(str(fasta_path), device=device)
    assert "inputs_embeds" in data and "group_embeds" in data and "group_labels" in data
    inputs_embeds = data["inputs_embeds"]
    group_embeds = data["group_embeds"]
    group_labels = data["group_labels"]
    if isinstance(inputs_embeds, torch.Tensor):
        pass
    else:
        inputs_embeds = torch.from_numpy(inputs_embeds)
        group_embeds = torch.from_numpy(group_embeds)
    n = inputs_embeds.shape[0]
    print(f"  Encoded {n} sequences.")

    print("Step 2: Loading ProteomeLM and running forward...")
    model = ProteomeLMForMaskedLM.from_pretrained(args.proteomelm_model)
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    with torch.no_grad():
        inp = inputs_embeds[None].to(device=device, dtype=torch.bfloat16)
        grp = group_embeds[None].to(device=device, dtype=torch.bfloat16)
        output = model(inputs_embeds=inp, group_embeds=grp)
    embeddings = output.logits.squeeze(0).cpu()
    assert embeddings.shape[0] == n

    print("Step 3: Saving .pt...")
    result = {
        "embeddings": embeddings,
        "group_labels": group_labels,
    }
    torch.save(result, out_path)
    print(f"  Saved embeddings shape {embeddings.shape} and {len(group_labels)} labels to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
