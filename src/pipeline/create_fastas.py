"""Step 3: Split combined proteins.fasta into per-organism FASTA files.

Reads proteins.fasta from processed/ and/or mvp/ directories and writes one
FASTA file per organism to the organism_fastas/ subdirectory.

FASTA header format: >orgId:locusId
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

ORGANISM_FASTAS_DIR = "organism_fastas"


def parse_fasta_header(header: str) -> tuple[str | None, str | None]:
    """Parse FASTA header to extract orgId and locusId.

    Format: >orgId:locusId
    Returns: (orgId, locusId)
    """
    header = header.lstrip(">")
    parts = header.split(":")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def split_fasta_by_organism(fasta_path: Path, output_dir: Path) -> dict[str, int]:
    """Split a combined FASTA into one file per organism.

    Streams through the FASTA file and writes one file per organism.

    Args:
        fasta_path: Path to the combined proteins.fasta.
        output_dir: Directory to write per-organism FASTA files.

    Returns:
        Dict mapping orgId to sequence count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    current_org: str | None = None
    current_records: list[str] = []

    def flush_organism(org_id: str | None) -> None:
        if org_id is None or len(current_records) == 0:
            return
        out_path = output_dir / f"{org_id}.fasta"
        with open(out_path, "w") as fh:
            fh.write("".join(current_records))
        record_count = sum(1 for line in current_records if line.startswith(">"))
        counts[org_id] = record_count

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                org_id, _ = parse_fasta_header(line.strip())
                if org_id != current_org:
                    flush_organism(current_org)
                    current_org = org_id
                    current_records = []
                current_records.append(line)
            else:
                current_records.append(line)

    flush_organism(current_org)

    return counts


class CreateFastasStep:
    """Split proteins.fasta into per-organism FASTA files."""

    @property
    def name(self) -> str:
        return "create_fastas"

    def check_inputs(self, config: PipelineConfig) -> bool:
        ok = True
        for subset in config.create_fastas.subsets:
            base_dir = (
                config.data.mvp_dir if subset == "mvp" else config.data.processed_dir
            )
            fasta = base_dir / "proteins.fasta"
            if not fasta.is_file():
                logger.warning("proteins.fasta not found for subset '%s': %s", subset, fasta)
                ok = False
        return ok

    def run(self, config: PipelineConfig) -> None:
        for subset in config.create_fastas.subsets:
            base_dir = (
                config.data.mvp_dir if subset == "mvp" else config.data.processed_dir
            )
            fasta_path = base_dir / "proteins.fasta"
            output_dir = base_dir / ORGANISM_FASTAS_DIR

            if not fasta_path.is_file():
                logger.warning(
                    "Skipping subset '%s': %s not found", subset, fasta_path
                )
                continue

            logger.info("Splitting %s ...", fasta_path)
            counts = split_fasta_by_organism(fasta_path, output_dir)
            total = sum(counts.values())
            logger.info(
                "  %s: created %d organism FASTA files (%s sequences) in %s",
                subset,
                len(counts),
                f"{total:,}",
                output_dir,
            )
