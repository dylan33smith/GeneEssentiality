#!/usr/bin/env python3
"""
Generate ESM-C sequence embeddings and save in the format expected by ProteomeLM.

Uses the official esm library (EvolutionaryScale) only. Does not import ProteomeLM.
Output: .pt file with keys inputs_embeds, group_embeds (both shape [N, D]),
and group_labels (list of N sequence IDs). ProteomeLM can load this via
torch.load(encoded_genome_file).

If ESMC.from_pretrained("esmc_300m") fails, try: --model esmc_600m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from Bio import SeqIO
from tqdm import tqdm

# ESM-C from official esm package (no ProteomeLM)
from esm.models.esmc import ESMC


def average_representation(
    output: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Mean-pool over non-padding positions. Same logic as ProteomeLM's
    average_representation; reimplemented here to keep the script standalone.

    Args:
        output: (batch_size, sequence_length, hidden_size)
        input_ids: (batch_size, sequence_length)
        pad_token_id: token id used for padding

    Returns:
        (batch_size, hidden_size)
    """
    mask = input_ids != pad_token_id
    out = output.clone()
    out[~mask] = 0.0
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return out.sum(dim=1) / valid_counts


def load_fasta(fasta_path: Path, truncate: int) -> tuple[list[str], list[str]]:
    """Load labels and sequences from FASTA; truncate each sequence to truncate residues."""
    labels: list[str] = []
    sequences: list[str] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        labels.append(record.id)
        sequences.append(str(record.seq)[:truncate])
    return labels, sequences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ESM-C embeddings for ProteomeLM (FASTA -> .pt)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input FASTA file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--model",
        default="esmc_300m",
        help="ESM-C checkpoint name (default: esmc_300m; try esmc_600m if 300m unavailable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Max residue count per batch (default: 16000)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for model (default: cuda:0)",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=4096,
        help="Max sequence length in residues (default: 4096)",
    )
    args = parser.parse_args()

    fasta_path = args.input.resolve()
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        return 1

    labels, sequences = load_fasta(fasta_path, args.truncate)
    if not sequences:
        print(f"Error: No sequences found in {fasta_path}", file=sys.stderr)
        return 1

    n_seqs = len(sequences)
    print(f"Loaded {n_seqs} sequences from {fasta_path}")

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("Warning: CUDA not available, using CPU", file=sys.stderr)
        device = "cpu"
    else:
        device = args.device

    try:
        model = ESMC.from_pretrained(args.model)
    except Exception as e:
        print(
            f"Error: Failed to load model '{args.model}'. Try --model esmc_600m. {e}",
            file=sys.stderr,
        )
        return 1

    model = model.eval().to(device)
    pad_token_id = model.tokenizer.pad_token_id

    all_embeddings: list[torch.Tensor] = []
    current_batch: list[str] = []
    current_tokens = 0

    for i in tqdm(range(n_seqs), desc="Encoding sequences"):
        if current_tokens + len(sequences[i]) > args.max_tokens and current_batch:
            input_ids = model._tokenize(current_batch).long()
            with torch.no_grad():
                output = model(input_ids.to(device))
            emb = average_representation(
                output.embeddings,
                input_ids,
                pad_token_id,
            ).detach().cpu()
            all_embeddings.append(emb)
            current_batch = []
            current_tokens = 0
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        current_batch.append(sequences[i])
        current_tokens += len(sequences[i])

    if current_batch:
        input_ids = model._tokenize(current_batch).long()
        with torch.no_grad():
            output = model(input_ids.to(device))
        emb = average_representation(
            output.embeddings,
            input_ids,
            pad_token_id,
        ).detach().cpu()
        all_embeddings.append(emb)

    embeddings = torch.cat(all_embeddings, dim=0)
    assert embeddings.shape[0] == n_seqs, (embeddings.shape[0], n_seqs)

    data = {
        "inputs_embeds": embeddings,
        "group_embeds": embeddings,
        "group_labels": labels,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")

    # Validation: reload and print shapes
    loaded = torch.load(args.output, map_location="cpu", weights_only=False)
    print("Validation (loaded file):")
    print(f"  inputs_embeds.shape = {loaded['inputs_embeds'].shape}")
    print(f"  group_embeds.shape  = {loaded['group_embeds'].shape}")
    print(f"  First example shape = {loaded['inputs_embeds'][0].shape}")
    print("  Format matches ProteomeLM expected keys and shapes (inputs_embeds, group_embeds (N, D)).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
