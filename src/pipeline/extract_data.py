"""Step 1: Extract data from FEBA SQLite database into processed Parquet files.

Extracts core tables (genes, experiments, fitness, organisms, domains,
orthologs) from data/raw/feba.db and saves them as Parquet files in
data/processed/. Also copies the raw amino-acid sequences file
(data/raw/aaseqs) to data/processed/proteins.fasta.

Refactored from:
    src/data/extract_data.py
    src/data/process_sequences.py (copy_all_sequences part)
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

EXPECTED_PARQUETS = [
    "genes.parquet",
    "experiments.parquet",
    "fitness.parquet",
    "organisms.parquet",
    "domains.parquet",
    "orthologs.parquet",
]
EXPECTED_FASTA = "proteins.fasta"


def extract_genes(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract protein-coding genes (type=1) with computed gene_length."""
    query = """
    SELECT
        orgId, locusId, sysName, scaffoldId, begin, end,
        (end - begin + 1) as gene_length,
        type, strand, gene, desc, GC
    FROM Gene
    WHERE type = 1
    ORDER BY orgId, scaffoldId, begin
    """
    df = pd.read_sql_query(query, conn)
    logger.info("  Extracted %s protein-coding genes", f"{len(df):,}")
    return df


def extract_experiments(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract experiments with numeric coercion on quality columns."""
    query = """
    SELECT
        orgId, expName, expDesc, expGroup, mutantLibrary,
        media, mediaStrength,
        condition_1, concentration_1, units_1,
        condition_2, concentration_2, units_2,
        condition_3, concentration_3, units_3,
        temperature, pH, aerobic, vessel, liquid, shaking,
        cor12, maxFit, nGenerations
    FROM Experiment
    ORDER BY orgId, expName
    """
    df = pd.read_sql_query(query, conn)
    for col in ["mediaStrength", "cor12", "maxFit", "nGenerations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("  Extracted %s experiments", f"{len(df):,}")
    return df


def extract_fitness(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract gene fitness scores (fit, t) from GeneFitness."""
    query = """
    SELECT orgId, locusId, expName, fit, t
    FROM GeneFitness
    ORDER BY orgId, locusId, expName
    """
    df = pd.read_sql_query(query, conn)
    logger.info("  Extracted %s fitness records", f"{len(df):,}")
    return df


def extract_organisms(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract organism metadata."""
    query = """
    SELECT orgId, division, genus, species, strain, taxonomyId
    FROM Organism
    ORDER BY orgId
    """
    df = pd.read_sql_query(query, conn)
    logger.info("  Extracted %s organisms", f"{len(df):,}")
    return df


def extract_domains(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract protein domain annotations (PFam / TIGRFam)."""
    query = """
    SELECT orgId, locusId, domainDb, domainId, domainName,
           begin, end, score, evalue
    FROM GeneDomain
    ORDER BY orgId, locusId, begin
    """
    df = pd.read_sql_query(query, conn)
    logger.info("  Extracted %s domain annotations", f"{len(df):,}")
    return df


def extract_orthologs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Extract ortholog pairs for cluster-based splits."""
    query = """
    SELECT orgId1, locusId1, orgId2, locusId2, ratio
    FROM Ortholog
    ORDER BY orgId1, locusId1, orgId2, locusId2
    """
    df = pd.read_sql_query(query, conn)
    logger.info("  Extracted %s ortholog pairs", f"{len(df):,}")
    return df


def save_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    """Save DataFrame as Parquet and log size."""
    filepath = path / f"{name}.parquet"
    df.to_parquet(filepath, index=False)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info("  Saved %s (%.1f MB)", filepath.name, size_mb)


class ExtractDataStep:
    """Extract raw FEBA database into processed Parquet tables + proteins.fasta."""

    @property
    def name(self) -> str:
        return "extract_data"

    def check_inputs(self, config: PipelineConfig) -> bool:
        raw_dir = config.data.raw_dir
        db_path = raw_dir / config.extract_data.db_filename
        seq_path = raw_dir / config.extract_data.raw_sequences_filename
        ok = True
        if not db_path.is_file():
            logger.warning("SQLite database not found: %s", db_path)
            ok = False
        if not seq_path.is_file():
            logger.warning("Raw sequences file not found: %s", seq_path)
            ok = False
        return ok

    def run(self, config: PipelineConfig) -> None:
        processed_dir = config.data.processed_dir
        raw_dir = config.data.raw_dir

        all_expected = EXPECTED_PARQUETS + [EXPECTED_FASTA]
        missing = [f for f in all_expected if not (processed_dir / f).exists()]
        if not missing:
            logger.info(
                "All processed outputs already exist in %s. Skipping.",
                processed_dir,
            )
            return

        db_path = raw_dir / config.extract_data.db_filename
        if not db_path.is_file():
            raise FileNotFoundError(f"Database not found: {db_path}")

        processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Connecting to database: %s", db_path)
        conn = sqlite3.connect(str(db_path))

        try:
            genes_df = extract_genes(conn)
            experiments_df = extract_experiments(conn)
            organisms_df = extract_organisms(conn)
            domains_df = extract_domains(conn)
            orthologs_df = extract_orthologs(conn)
            fitness_df = extract_fitness(conn)

            logger.info("Saving Parquet files ...")
            save_parquet(genes_df, processed_dir, "genes")
            save_parquet(experiments_df, processed_dir, "experiments")
            save_parquet(organisms_df, processed_dir, "organisms")
            save_parquet(domains_df, processed_dir, "domains")
            save_parquet(orthologs_df, processed_dir, "orthologs")
            save_parquet(fitness_df, processed_dir, "fitness")
        finally:
            conn.close()

        seq_src = raw_dir / config.extract_data.raw_sequences_filename
        seq_dst = processed_dir / EXPECTED_FASTA
        if seq_src.is_file():
            logger.info("Copying %s -> %s", seq_src, seq_dst)
            shutil.copy(seq_src, seq_dst)
            with open(seq_dst) as f:
                seq_count = sum(1 for line in f if line.startswith(">"))
            logger.info("  Copied %s sequences", f"{seq_count:,}")
        else:
            logger.warning("Raw sequences file not found: %s", seq_src)

        logger.info("Extraction complete. Output: %s", processed_dir)
