"""Step 4: Generate ProteomeLM embeddings from per-organism FASTA files.

Uses ESM-C 600M for sequence encoding, then the ProteomeLM transformer to
produce per-gene contextualized embeddings. Processes all organism FASTAs in
a single invocation.

Output per organism: a .pt dict with keys 'embeddings' (Tensor [N, D]) and
'group_labels' (list of N strings 'orgId:locusId').

Memory optimisations vs. the original script:
- torch.inference_mode() instead of torch.no_grad()
- torch.cuda.empty_cache() + gc.collect() after each organism
- Peak GPU memory logged per organism
- tqdm progress bar over organisms
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def _ensure_proteomelm_on_path(project_root: Path) -> None:
    """Add the local ProteomeLM checkout to sys.path if it exists."""
    proteomelm_root = project_root / "ProteomeLM"
    if proteomelm_root.is_dir() and str(proteomelm_root) not in sys.path:
        sys.path.insert(0, str(proteomelm_root))


def _resolve_device(requested: str) -> str:
    """Fall back to CPU if CUDA is requested but unavailable."""
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


def encode_single_organism(
    fasta_path: Path,
    out_path: Path,
    model_name: str,
    device: str,
) -> int:
    """Encode one organism FASTA and save the .pt file.

    Args:
        fasta_path: Path to organism FASTA.
        out_path: Destination .pt file.
        model_name: HuggingFace model ID or local path for ProteomeLM.
        device: Torch device string.

    Returns:
        Number of sequences encoded.
    """
    import torch
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
    from proteomelm.utils import build_genome_esmc

    logger.info("  ESM-C 600M encoding %s ...", fasta_path.name)
    data = build_genome_esmc(str(fasta_path), device=device)

    assert "inputs_embeds" in data and "group_embeds" in data and "group_labels" in data
    inputs_embeds = data["inputs_embeds"]
    group_embeds = data["group_embeds"]
    group_labels = data["group_labels"]

    if not isinstance(inputs_embeds, torch.Tensor):
        inputs_embeds = torch.from_numpy(inputs_embeds)
        group_embeds = torch.from_numpy(group_embeds)

    n = inputs_embeds.shape[0]
    logger.info("  Encoded %d sequences.", n)

    logger.info("  Loading ProteomeLM and running forward pass ...")
    model = ProteomeLMForMaskedLM.from_pretrained(model_name)
    model = model.to(dtype=torch.bfloat16, device=device).eval()

    with torch.inference_mode():
        inp = inputs_embeds[None].to(device=device, dtype=torch.bfloat16)
        grp = group_embeds[None].to(device=device, dtype=torch.bfloat16)
        output = model(inputs_embeds=inp, group_embeds=grp)

    embeddings = output.logits.squeeze(0).cpu()
    assert embeddings.shape[0] == n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "embeddings": embeddings,
        "group_labels": group_labels,
    }
    torch.save(result, out_path)
    logger.info(
        "  Saved embeddings shape %s and %d labels -> %s",
        list(embeddings.shape),
        len(group_labels),
        out_path,
    )

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        logger.info("  Peak GPU memory: %.0f MB", peak_mb)
        torch.cuda.reset_peak_memory_stats(device)

    del model, inp, grp, output, embeddings, inputs_embeds, group_embeds, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return n


class GenerateEmbeddingsStep:
    """Generate ProteomeLM embeddings for all organism FASTAs."""

    @property
    def name(self) -> str:
        return "generate_embeddings"

    def check_inputs(self, config: PipelineConfig) -> bool:
        input_dir = config.data.mvp_dir / config.embeddings.input_subdir
        if not input_dir.is_dir():
            logger.warning("FASTA input directory not found: %s", input_dir)
            return False
        fasta_files = sorted(input_dir.glob("*.fasta"))
        if not fasta_files:
            logger.warning("No .fasta files in %s", input_dir)
            return False
        return True

    def run(self, config: PipelineConfig) -> None:
        from tqdm import tqdm

        _ensure_proteomelm_on_path(config.project_root)

        input_dir = config.data.mvp_dir / config.embeddings.input_subdir
        output_dir = config.data.mvp_dir / config.embeddings.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        device = _resolve_device(config.embeddings.device)
        model_name = config.embeddings.proteomelm_model

        fasta_files = sorted(input_dir.glob("*.fasta"))
        if not fasta_files:
            raise FileNotFoundError(f"No .fasta files in {input_dir}")

        logger.info("Input dir:  %s", input_dir)
        logger.info("Output dir: %s", output_dir)
        logger.info("Device:     %s", device)
        logger.info("Model:      %s", model_name)
        logger.info("Organisms:  %d", len(fasta_files))

        total_seqs = 0
        for fasta_path in tqdm(fasta_files, desc="Organisms", unit="org"):
            out_path = output_dir / f"{fasta_path.stem}_proteomelm.pt"
            if out_path.exists():
                logger.info("  Skipping %s (output exists)", fasta_path.name)
                continue
            n = encode_single_organism(fasta_path, out_path, model_name, device)
            total_seqs += n

        logger.info("Embedding generation complete. Total new sequences: %d", total_seqs)
