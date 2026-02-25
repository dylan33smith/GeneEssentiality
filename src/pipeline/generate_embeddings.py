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
    proteomelm_root = project_root.parent / "ProteomeLM"
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
    hidden_layer: int = 8,
    proteomelm_model: object | None = None,
) -> int:
    """Encode one organism FASTA and save the .pt file.

    Args:
        fasta_path: Path to organism FASTA.
        out_path: Destination .pt file.
        model_name: HuggingFace model ID or local path for ProteomeLM.
        device: Torch device string.
        hidden_layer: Index of hidden state to extract (0=embedding, 1..N=layers).
            Use -1 for the last layer.
        proteomelm_model: Pre-loaded ProteomeLM model. If None, loads from model_name.

    Returns:
        Number of sequences encoded.
    """
    import torch
    from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
    from proteomelm.utils import build_genome_esmc

    logger.info("  ESM-C 600M encoding %s ...", fasta_path.name)
    data = build_genome_esmc(str(fasta_path), device=device)

    for key in ("inputs_embeds", "group_embeds", "group_labels"):
        if key not in data:
            raise ValueError(f"build_genome_esmc output missing required key '{key}'")
    inputs_embeds = data["inputs_embeds"]
    group_embeds = data["group_embeds"]
    group_labels = data["group_labels"]

    if not isinstance(inputs_embeds, torch.Tensor):
        inputs_embeds = torch.from_numpy(inputs_embeds)
        group_embeds = torch.from_numpy(group_embeds)

    n = inputs_embeds.shape[0]
    logger.info("  Encoded %d sequences.", n)

    owns_model = proteomelm_model is None
    if owns_model:
        logger.info("  Loading ProteomeLM ...")
        proteomelm_model = ProteomeLMForMaskedLM.from_pretrained(model_name)
        proteomelm_model = proteomelm_model.to(dtype=torch.bfloat16, device=device).eval()

    logger.info("  Running ProteomeLM forward pass ...")
    with torch.inference_mode():
        inp = inputs_embeds[None].to(device=device, dtype=torch.bfloat16)
        grp = group_embeds[None].to(device=device, dtype=torch.bfloat16)
        output = proteomelm_model(
            inputs_embeds=inp, group_embeds=grp, output_hidden_states=True
        )

    layer_idx = hidden_layer if hidden_layer >= 0 else -1
    embeddings = output.hidden_states[layer_idx].squeeze(0).cpu()
    if embeddings.shape[0] != n:
        raise ValueError(
            f"Embedding count ({embeddings.shape[0]}) != sequence count ({n})"
        )

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

    del inp, grp, output, embeddings, inputs_embeds, group_embeds, data
    if owns_model:
        del proteomelm_model
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
        import torch
        from tqdm import tqdm

        _ensure_proteomelm_on_path(config.project_root)

        input_dir = config.data.mvp_dir / config.embeddings.input_subdir
        hidden_layer = config.embeddings.hidden_layer
        layer_suffix = "last" if hidden_layer < 0 else str(hidden_layer)
        output_subdir_with_layer = f"{config.embeddings.output_subdir}_layer{layer_suffix}"
        output_dir = config.data.mvp_dir / output_subdir_with_layer
        output_dir.mkdir(parents=True, exist_ok=True)

        device = _resolve_device(config.embeddings.device)
        model_name = config.embeddings.proteomelm_model

        fasta_files = sorted(input_dir.glob("*.fasta"))
        if not fasta_files:
            raise FileNotFoundError(f"No .fasta files in {input_dir}")

        logger.info("Input dir:      %s", input_dir)
        logger.info("Output dir:     %s", output_dir)
        logger.info("Hidden layer:   %s", layer_suffix)
        logger.info("Device:         %s", device)
        logger.info("Model:          %s", model_name)
        logger.info("Organisms:      %d", len(fasta_files))

        files_to_process = [
            f for f in fasta_files
            if not (output_dir / f"{f.stem}_proteomelm.pt").exists()
        ]
        skipped = len(fasta_files) - len(files_to_process)
        if skipped:
            logger.info("Skipping %d organisms with existing outputs", skipped)

        if not files_to_process:
            logger.info("All organisms already processed.")
            return

        from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM

        logger.info("Loading ProteomeLM model (once for all organisms) ...")
        model = ProteomeLMForMaskedLM.from_pretrained(model_name)
        model = model.to(dtype=torch.bfloat16, device=device).eval()

        total_seqs = 0
        for fasta_path in tqdm(files_to_process, desc="Organisms", unit="org"):
            out_path = output_dir / f"{fasta_path.stem}_proteomelm.pt"
            n = encode_single_organism(
                fasta_path, out_path, model_name, device,
                hidden_layer=hidden_layer, proteomelm_model=model,
            )
            total_seqs += n

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Embedding generation complete. Total new sequences: %d", total_seqs)
