"""
Encode proteins from organism FASTA files using ESMC.

Standalone utility (not part of the main pipeline). Reads FASTA files from
organism_fastas/, encodes each protein with ESMC, and outputs a .pt file
mapping protein IDs to embedding vectors.

Usage:
    python scripts/encode_esmc.py [--input-dir PATH] [--output PATH] [--device DEVICE]

Example:
    python scripts/encode_esmc.py --input-dir data/mvp/organism_fastas --output data/mvp/embeddings/protein_embeddings.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "mvp" / "organism_fastas"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "mvp" / "embeddings" / "protein_embeddings.pt"

DEFAULT_MODEL = "esmc_600m"
MAX_SEQUENCE_LENGTH = 4096
DEFAULT_BATCH_TOKENS = 16000


def _average_representation(
    output: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """Average the representation over sequence length, ignoring padding tokens.

    Args:
        output: (batch_size, seq_len, hidden_size)
        input_ids: (batch_size, seq_len)
        pad_token_id: Padding token ID to exclude

    Returns:
        (batch_size, hidden_size)
    """
    mask = input_ids != pad_token_id
    output_masked = output.clone()
    output_masked[~mask] = 0.0
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return output_masked.sum(dim=1) / valid_counts


def load_model(checkpoint: str = DEFAULT_MODEL, device: str = "cuda") -> torch.nn.Module:
    """Load ESMC model for encoding."""
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(checkpoint)
    model.eval()
    model.to(device, dtype=torch.bfloat16)
    return model


def parse_fasta_file(fasta_path: Path) -> tuple[list[str], list[str]]:
    """Parse FASTA file and return (protein_ids, sequences).

    Protein ID is the full header without '>' (e.g. orgId:locusId).
    Sequences are truncated to MAX_SEQUENCE_LENGTH.
    """
    ids_list = []
    seq_list = []
    current_id = None
    current_seq_parts: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq_parts)[:MAX_SEQUENCE_LENGTH]
                    ids_list.append(current_id)
                    seq_list.append(seq)
                current_id = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    if current_id is not None:
        seq = "".join(current_seq_parts)[:MAX_SEQUENCE_LENGTH]
        ids_list.append(current_id)
        seq_list.append(seq)

    return ids_list, seq_list


def extract_embeddings_batch(
    model: torch.nn.Module,
    sequences: list[str],
    device: str,
    pad_token_id: int,
) -> torch.Tensor:
    """Encode a batch of sequences and return mean-pooled embeddings.

    Args:
        model: ESMC model
        sequences: List of protein sequences
        device: Device for inference
        pad_token_id: Padding token ID for masking

    Returns:
        Tensor of shape (batch_size, hidden_size)
    """
    input_ids = model._tokenize(sequences).long().to(device)
    with torch.no_grad():
        output = model(input_ids)

    if hasattr(output, "embeddings") and output.embeddings is not None:
        repr_tensor = output.embeddings
    else:
        repr_tensor = output.hidden_states[-1]

    embeddings = _average_representation(repr_tensor.float(), input_ids, pad_token_id)
    return embeddings.cpu()


def encode_fasta_file(
    model: torch.nn.Module,
    fasta_path: Path,
    device: str,
    batch_tokens: int,
) -> dict[str, torch.Tensor]:
    """Encode all proteins in a FASTA file.

    Returns dict mapping protein_id -> embedding tensor.
    """
    ids_list, seq_list = parse_fasta_file(fasta_path)
    if not ids_list:
        return {}

    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id or 0

    embeddings_dict: dict[str, torch.Tensor] = {}
    current_batch_ids: list[str] = []
    current_batch_seqs: list[str] = []
    current_num_tokens = 0

    for i in tqdm(range(len(seq_list)), desc=fasta_path.name, leave=False):
        seq = seq_list[i]
        pid = ids_list[i]

        if current_num_tokens + len(seq) > batch_tokens and current_batch_seqs:
            batch_embeddings = extract_embeddings_batch(
                model, current_batch_seqs, device, pad_token_id
            )
            for bid, emb in zip(current_batch_ids, batch_embeddings):
                embeddings_dict[bid] = emb

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_batch_ids = []
            current_batch_seqs = []
            current_num_tokens = 0

        current_batch_ids.append(pid)
        current_batch_seqs.append(seq)
        current_num_tokens += len(seq)

    if current_batch_seqs:
        batch_embeddings = extract_embeddings_batch(
            model, current_batch_seqs, device, pad_token_id
        )
        for bid, emb in zip(current_batch_ids, batch_embeddings):
            embeddings_dict[bid] = emb

    return embeddings_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode proteins with ESMC")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing organism FASTA files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=DEFAULT_BATCH_TOKENS,
        help="Max tokens per batch",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="ESMC model checkpoint name",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    fasta_files = sorted(input_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta files in {input_dir}")

    print("=" * 50)
    print("ESMC PROTEIN ENCODING")
    print("=" * 50)
    print(f"Input dir:  {input_dir}")
    print(f"Output:     {output_path}")
    print(f"Device:     {args.device}")
    print(f"Model:      {args.model}")
    print(f"FASTA files: {len(fasta_files)}")
    print()

    print("Loading model...")
    model = load_model(args.model, args.device)

    all_embeddings: dict[str, torch.Tensor] = {}
    for fasta_path in tqdm(fasta_files, desc="Organisms"):
        org_embeddings = encode_fasta_file(model, fasta_path, args.device, args.batch_tokens)
        for pid, emb in org_embeddings.items():
            if pid in all_embeddings:
                raise ValueError(f"Duplicate protein ID: {pid}")
            all_embeddings[pid] = emb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_embeddings, output_path)

    print()
    print("=" * 50)
    print(f"Saved {len(all_embeddings):,} protein embeddings to {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
