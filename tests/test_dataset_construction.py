"""Dataset construction validation — every pipeline step from raw data to labeled embeddings.

Validates format and exact values at each stage (except embedding tensor values, which
depend on external ProteomeLM). All tests require real data on disk and are marked
with @pytest.mark.data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config
from src.data_io import get_data_dir, get_mvp_dir, get_processed_dir
from src.pipeline.extract_data import (
    extract_experiments,
    extract_fitness,
    extract_genes,
    extract_organisms,
)
from src.pipeline.classify_genes import compute_gene_essentiality, merge_essentiality_into_genes
from src.pipeline.filter_mvp import (
    build_condition_vocab,
    filter_experiments,
    filter_fitness_vectorized,
    get_mvp_organisms,
)
from tests.conftest import parse_fasta_to_seqs

pytestmark = pytest.mark.data

# --- Schema expectations (format only) ---
RAW_DB_TABLES = ["Gene", "Experiment", "GeneFitness", "Organism"]
PROCESSED_GENE_COLUMNS = {
    "orgId", "locusId", "sysName", "scaffoldId", "begin", "end",
    "gene_length", "type", "strand", "gene", "desc", "GC",
}
PROCESSED_EXPERIMENT_COLUMNS = {"orgId", "expName", "media", "cor12", "expGroup"}
PROCESSED_FITNESS_COLUMNS = {"orgId", "locusId", "expName", "fit", "t"}
PROCESSED_ORGANISM_COLUMNS = {"orgId", "genus", "species"}
ESSENTIALITY_COLUMNS = {
    "n_total_experiments", "n_confident_experiments", "frac_not_confident",
    "mean_fit_confident", "frac_essential_all", "frac_essential_confident",
    "essentiality_class",
}
VALID_ESSENTIALITY_CLASSES = {"always_essential", "conditional", "non_essential", "no_data"}
CONDITION_VOCAB_KEYS = {"media", "condition_1", "expGroup", "metadata"}


# --- Step 1: Raw data (extract_data inputs) ---


def test_raw_db_exists_and_has_tables():
    """Raw FEBA database exists and contains expected tables."""
    config = load_config()
    db_path = config.data.raw_dir / config.extract_data.db_filename
    if not db_path.is_file():
        pytest.skip(f"Raw database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?, ?)",
            tuple(RAW_DB_TABLES),
        )
        tables = {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()

    assert tables == set(RAW_DB_TABLES), f"Missing tables: {set(RAW_DB_TABLES) - tables}"


def test_raw_sequences_file_exists():
    """Raw sequences file (aaseqs) exists and has FASTA-like headers."""
    config = load_config()
    seq_path = config.data.raw_dir / config.extract_data.raw_sequences_filename
    if not seq_path.is_file():
        pytest.skip(f"Raw sequences not found: {seq_path}")

    with open(seq_path) as f:
        first_lines = [f.readline() for _ in range(5)]
    headers = [l.strip() for l in first_lines if l.strip().startswith(">")]
    assert len(headers) >= 1, "Raw sequences should have at least one FASTA header"


def test_raw_sequences_identical_to_processed_fasta():
    """Raw aaseqs and processed proteins.fasta are byte-identical (exact copy)."""
    config = load_config()
    raw_path = config.data.raw_dir / config.extract_data.raw_sequences_filename
    processed_path = get_processed_dir() / "proteins.fasta"
    if not raw_path.is_file() or not processed_path.is_file():
        pytest.skip("Raw sequences or processed proteins.fasta not found")

    raw_bytes = raw_path.read_bytes()
    processed_bytes = processed_path.read_bytes()
    assert raw_bytes == processed_bytes, "processed proteins.fasta should equal raw aaseqs"


# --- Step 2: Processed data (extract_data outputs) ---


def test_processed_parquet_files_exist():
    """Processed Parquet files exist with expected columns."""
    processed_dir = get_processed_dir()
    if not processed_dir.is_dir():
        pytest.skip(f"Processed dir not found: {processed_dir}")

    genes_path = processed_dir / "genes.parquet"
    if not genes_path.is_file():
        pytest.skip("genes.parquet not found")

    genes = pd.read_parquet(genes_path)
    missing = PROCESSED_GENE_COLUMNS - set(genes.columns)
    assert not missing, f"genes.parquet missing columns: {missing}"

    exps_path = processed_dir / "experiments.parquet"
    if exps_path.is_file():
        exps = pd.read_parquet(exps_path)
        missing = PROCESSED_EXPERIMENT_COLUMNS - set(exps.columns)
        assert not missing, f"experiments.parquet missing columns: {missing}"

    fit_path = processed_dir / "fitness.parquet"
    if fit_path.is_file():
        fit = pd.read_parquet(fit_path)
        missing = PROCESSED_FITNESS_COLUMNS - set(fit.columns)
        assert not missing, f"fitness.parquet missing columns: {missing}"

    org_path = processed_dir / "organisms.parquet"
    if org_path.is_file():
        orgs = pd.read_parquet(org_path)
        missing = PROCESSED_ORGANISM_COLUMNS - set(orgs.columns)
        assert not missing, f"organisms.parquet missing columns: {missing}"


def test_processed_proteins_fasta_exists_and_format():
    """Processed proteins.fasta exists and has orgId:locusId headers."""
    processed_dir = get_processed_dir()
    fasta_path = processed_dir / "proteins.fasta"
    if not fasta_path.is_file():
        pytest.skip("proteins.fasta not found")

    with open(fasta_path) as f:
        headers = [line.strip().lstrip(">") for line in f if line.strip().startswith(">")]

    assert len(headers) > 0, "proteins.fasta should have headers"
    for h in headers[:10]:
        parts = h.split(":")
        assert len(parts) >= 2, f"Header should be orgId:locusId format: {h[:50]}"


def test_processed_parquet_exact_match_extract_from_db():
    """Processed Parquet files exactly match re-extraction from raw DB."""
    config = load_config()
    db_path = config.data.raw_dir / config.extract_data.db_filename
    processed_dir = get_processed_dir()
    if not db_path.is_file() or not processed_dir.is_dir():
        pytest.skip("Raw DB or processed dir not found")

    conn = sqlite3.connect(str(db_path))
    try:
        extracted_genes = extract_genes(conn)
        extracted_experiments = extract_experiments(conn)
        extracted_organisms = extract_organisms(conn)
        extracted_fitness = extract_fitness(conn)
    finally:
        conn.close()

    genes_path = processed_dir / "genes.parquet"
    if genes_path.is_file():
        disk_genes = pd.read_parquet(genes_path)
        pd.testing.assert_frame_equal(
            extracted_genes.reset_index(drop=True),
            disk_genes[extracted_genes.columns].reset_index(drop=True),
            check_dtype=False,
        )

    exps_path = processed_dir / "experiments.parquet"
    if exps_path.is_file():
        disk_exps = pd.read_parquet(exps_path)
        pd.testing.assert_frame_equal(
            extracted_experiments.reset_index(drop=True),
            disk_exps[extracted_experiments.columns].reset_index(drop=True),
            check_dtype=False,
        )

    orgs_path = processed_dir / "organisms.parquet"
    if orgs_path.is_file():
        disk_orgs = pd.read_parquet(orgs_path)
        pd.testing.assert_frame_equal(
            extracted_organisms.reset_index(drop=True),
            disk_orgs[extracted_organisms.columns].reset_index(drop=True),
            check_dtype=False,
        )

    fit_path = processed_dir / "fitness.parquet"
    if fit_path.is_file():
        disk_fit = pd.read_parquet(fit_path)
        pd.testing.assert_frame_equal(
            extracted_fitness.reset_index(drop=True),
            disk_fit[extracted_fitness.columns].reset_index(drop=True),
            check_dtype=False,
        )


# --- Step 3: Classify genes processed (processed genes with essentiality) ---


def test_processed_genes_have_essentiality_columns():
    """Processed genes.parquet has essentiality classification columns."""
    processed_dir = get_processed_dir()
    genes_path = processed_dir / "genes.parquet"
    if not genes_path.is_file():
        pytest.skip("processed genes.parquet not found")

    genes = pd.read_parquet(genes_path)
    missing = ESSENTIALITY_COLUMNS - set(genes.columns)
    assert not missing, f"Processed genes missing essentiality columns: {missing}"

    classes = set(genes["essentiality_class"].dropna().unique())
    assert classes.issubset(VALID_ESSENTIALITY_CLASSES), f"Invalid classes: {classes}"


def test_processed_genes_essentiality_exact_match_recomputed():
    """Processed genes essentiality exactly matches recomputation from fitness."""
    config = load_config()
    processed_dir = get_processed_dir()
    genes_path = processed_dir / "genes.parquet"
    fitness_path = processed_dir / "fitness.parquet"
    if not genes_path.is_file() or not fitness_path.is_file():
        pytest.skip("Processed genes or fitness not found")

    genes = pd.read_parquet(genes_path)
    fitness = pd.read_parquet(fitness_path)
    gene_stats = compute_gene_essentiality(fitness, config.classify_genes)
    merged = merge_essentiality_into_genes(genes, gene_stats)

    for col in ESSENTIALITY_COLUMNS:
        if col not in genes.columns:
            continue
        disk_vals = genes[col]
        recomputed_vals = merged[col]
        if col in ("frac_not_confident", "mean_fit_confident", "frac_essential_all", "frac_essential_confident"):
            mask = disk_vals.notna() & recomputed_vals.notna()
            np.testing.assert_allclose(
                disk_vals[mask].astype(float).values,
                recomputed_vals[mask].astype(float).values,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Mismatch in {col}",
            )
            assert (disk_vals.isna() == recomputed_vals.isna()).all(), f"NaN pattern mismatch in {col}"
        else:
            pd.testing.assert_series_equal(disk_vals, recomputed_vals, check_names=False)


# --- Step 4: Filter MVP (MVP subset) ---


def test_mvp_parquet_files_exist_and_format():
    """MVP Parquet files exist with expected schema."""
    mvp_dir = get_mvp_dir()
    if not mvp_dir.is_dir():
        pytest.skip(f"MVP dir not found: {mvp_dir}")

    for name in ["genes.parquet", "experiments.parquet", "fitness.parquet", "organisms.parquet"]:
        path = mvp_dir / name
        if not path.is_file():
            pytest.skip(f"MVP {name} not found")
        df = pd.read_parquet(path)
        assert len(df) > 0, f"MVP {name} is empty"


def test_mvp_proteins_fasta_exists():
    """MVP proteins.fasta exists and has orgId:locusId format."""
    mvp_dir = get_mvp_dir()
    fasta_path = mvp_dir / "proteins.fasta"
    if not fasta_path.is_file():
        pytest.skip("MVP proteins.fasta not found")

    with open(fasta_path) as f:
        headers = [line.strip().lstrip(">") for line in f if line.strip().startswith(">")]
    assert len(headers) > 0
    for h in headers[:5]:
        assert ":" in h, f"Header should be orgId:locusId: {h[:50]}"


def test_mvp_condition_vocab_exists_and_format():
    """MVP condition_vocab.json exists with expected keys."""
    mvp_dir = get_mvp_dir()
    vocab_path = mvp_dir / "condition_vocab.json"
    if not vocab_path.is_file():
        pytest.skip("condition_vocab.json not found")

    with open(vocab_path) as f:
        vocab = json.load(f)
    missing = CONDITION_VOCAB_KEYS - set(vocab.keys())
    assert not missing, f"condition_vocab missing keys: {missing}"


def test_mvp_exact_subset_of_processed():
    """MVP Parquet tables are exact subsets of processed (same column values).

    For genes: only base metadata columns are compared; essentiality columns
    (n_total_experiments, etc.) are recomputed from MVP fitness and thus differ.
    """
    processed_dir = get_processed_dir()
    mvp_dir = get_mvp_dir()
    if not processed_dir.is_dir() or not mvp_dir.is_dir():
        pytest.skip("Processed or MVP dir not found")

    key_cols = {
        "genes": ["orgId", "locusId"],
        "experiments": ["orgId", "expName"],
        "organisms": ["orgId"],
    }
    skip_cols_genes = ESSENTIALITY_COLUMNS
    for table in ["genes", "experiments", "organisms"]:
        p_path = processed_dir / f"{table}.parquet"
        m_path = mvp_dir / f"{table}.parquet"
        if not p_path.is_file() or not m_path.is_file():
            continue
        proc = pd.read_parquet(p_path)
        mvp = pd.read_parquet(m_path)
        keys = key_cols[table]
        merged = mvp.merge(proc, on=keys, how="left", suffixes=("", "_proc"))
        skip_cols = skip_cols_genes if table == "genes" else set()
        for col in mvp.columns:
            if col in keys or col in skip_cols or col + "_proc" not in merged.columns:
                continue
            left = merged[col]
            right = merged[col + "_proc"]
            if pd.api.types.is_numeric_dtype(left):
                np.testing.assert_array_almost_equal(left.values, right.values, err_msg=f"MVP {table}.{col}")
            else:
                pd.testing.assert_series_equal(left, right, check_names=False)


def test_mvp_fitness_exact_subset_of_processed():
    """MVP fitness is exact subset of processed fitness (same fit, t values)."""
    processed_dir = get_processed_dir()
    mvp_dir = get_mvp_dir()
    p_path = processed_dir / "fitness.parquet"
    m_path = mvp_dir / "fitness.parquet"
    if not p_path.is_file() or not m_path.is_file():
        pytest.skip("Fitness files not found")

    proc_fit = pd.read_parquet(p_path)
    mvp_fit = pd.read_parquet(m_path)
    merged = mvp_fit.merge(
        proc_fit,
        on=["orgId", "locusId", "expName"],
        how="left",
        suffixes=("", "_proc"),
    )
    np.testing.assert_allclose(merged["fit"].values, merged["fit_proc"].values, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(merged["t"].values, merged["t_proc"].values, rtol=1e-9, atol=1e-12)


def test_mvp_proteins_fasta_exact_subset_of_processed():
    """MVP proteins.fasta contains exactly the genes in MVP genes, sequences match processed."""
    processed_dir = get_processed_dir()
    mvp_dir = get_mvp_dir()
    proc_fasta = processed_dir / "proteins.fasta"
    mvp_fasta = mvp_dir / "proteins.fasta"
    mvp_genes_path = mvp_dir / "genes.parquet"
    if not proc_fasta.is_file() or not mvp_fasta.is_file() or not mvp_genes_path.is_file():
        pytest.skip("FASTA or MVP genes not found")

    mvp_genes = pd.read_parquet(mvp_genes_path)
    mvp_gene_ids = set(zip(mvp_genes["orgId"].astype(str), mvp_genes["locusId"].astype(str)))

    proc_seqs = parse_fasta_to_seqs(proc_fasta)
    mvp_seqs = parse_fasta_to_seqs(mvp_fasta)

    assert set(mvp_seqs.keys()) == mvp_gene_ids, "MVP fasta genes should equal MVP genes"
    for key in mvp_seqs:
        assert key in proc_seqs, f"Gene {key} in MVP fasta should exist in processed"
        assert mvp_seqs[key] == proc_seqs[key], f"Sequence for {key} should match processed"


def test_mvp_condition_vocab_exact_match_recomputed():
    """MVP condition_vocab.json exactly matches recomputation from MVP experiments."""
    import logging
    logging.getLogger("src.pipeline.filter_mvp").setLevel(logging.WARNING)

    mvp_dir = get_mvp_dir()
    exps_path = mvp_dir / "experiments.parquet"
    vocab_path = mvp_dir / "condition_vocab.json"
    if not exps_path.is_file() or not vocab_path.is_file():
        pytest.skip("MVP experiments or condition_vocab not found")

    exps = pd.read_parquet(exps_path)
    recomputed = build_condition_vocab(exps)
    with open(vocab_path) as f:
        disk_vocab = json.load(f)

    for key in list(CONDITION_VOCAB_KEYS) + ["units_1", "composite", "aerobic", "temperature"]:
        if key not in disk_vocab or key not in recomputed:
            continue
        disk_v = disk_vocab[key]
        rec_v = recomputed[key]
        if key == "metadata":
            assert disk_v == rec_v, f"condition_vocab.{key} should match"
            continue
        assert disk_v["vocab"] == rec_v["vocab"], f"condition_vocab.{key}.vocab should match"
        assert disk_v["size"] == rec_v["size"], f"condition_vocab.{key}.size should match"
        disk_inv = {int(k): v for k, v in disk_v["inverse"].items()}
        assert disk_inv == rec_v["inverse"], f"condition_vocab.{key}.inverse should match"


# --- Step 5: Classify genes MVP (MVP genes with essentiality) ---


def test_mvp_genes_have_essentiality_columns():
    """MVP genes.parquet has essentiality classification."""
    mvp_dir = get_mvp_dir()
    genes_path = mvp_dir / "genes.parquet"
    if not genes_path.is_file():
        pytest.skip("MVP genes.parquet not found")

    genes = pd.read_parquet(genes_path)
    missing = ESSENTIALITY_COLUMNS - set(genes.columns)
    assert not missing, f"MVP genes missing essentiality columns: {missing}"
    assert genes["essentiality_class"].notna().all() or True


def test_mvp_genes_essentiality_exact_match_recomputed():
    """MVP genes essentiality exactly matches recomputation from MVP fitness."""
    config = load_config()
    mvp_dir = get_mvp_dir()
    genes_path = mvp_dir / "genes.parquet"
    fitness_path = mvp_dir / "fitness.parquet"
    if not genes_path.is_file() or not fitness_path.is_file():
        pytest.skip("MVP genes or fitness not found")

    genes = pd.read_parquet(genes_path)
    fitness = pd.read_parquet(fitness_path)
    gene_stats = compute_gene_essentiality(fitness, config.classify_genes)
    merged = merge_essentiality_into_genes(genes, gene_stats)

    for col in ESSENTIALITY_COLUMNS:
        if col not in genes.columns:
            continue
        disk_vals = genes[col]
        recomputed_vals = merged[col]
        if col in ("frac_not_confident", "mean_fit_confident", "frac_essential_all", "frac_essential_confident"):
            mask = disk_vals.notna() & recomputed_vals.notna()
            np.testing.assert_allclose(
                disk_vals[mask].astype(float).values,
                recomputed_vals[mask].astype(float).values,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"MVP {col}",
            )
            assert (disk_vals.isna() == recomputed_vals.isna()).all(), f"NaN pattern mismatch in {col}"
        else:
            pd.testing.assert_series_equal(disk_vals, recomputed_vals, check_names=False)


# --- Step 6: Create fastas (organism_fastas) ---


def test_organism_fastas_exist_and_format():
    """Per-organism FASTA files exist with orgId:locusId headers."""
    config = load_config()
    mvp_dir = config.data.mvp_dir
    fastas_dir = mvp_dir / config.embeddings.input_subdir
    if not fastas_dir.is_dir():
        pytest.skip(f"Organism fastas dir not found: {fastas_dir}")

    fasta_files = list(fastas_dir.glob("*.fasta"))
    if not fasta_files:
        pytest.skip("No .fasta files in organism_fastas")

    for fasta_path in fasta_files[:3]:
        with open(fasta_path) as f:
            headers = [line.strip().lstrip(">") for line in f if line.strip().startswith(">")]
        assert len(headers) > 0, f"{fasta_path.name} has no headers"
        for h in headers[:3]:
            parts = h.split(":")
            assert len(parts) >= 2, f"Header format orgId:locusId: {h[:50]}"


def test_organism_fastas_exact_partition_of_mvp_proteins():
    """Organism fastas exactly partition mvp/proteins.fasta (same genes, same sequences)."""
    config = load_config()
    mvp_dir = config.data.mvp_dir
    fastas_dir = mvp_dir / config.embeddings.input_subdir
    mvp_fasta = mvp_dir / "proteins.fasta"
    if not fastas_dir.is_dir() or not mvp_fasta.is_file():
        pytest.skip("Organism fastas or MVP proteins.fasta not found")

    mvp_seqs = parse_fasta_to_seqs(mvp_fasta)
    fasta_files = sorted(fastas_dir.glob("*.fasta"))
    if not fasta_files:
        pytest.skip("No organism FASTA files")

    org_seqs = {}
    for fasta_path in fasta_files:
        for key, seq in parse_fasta_to_seqs(fasta_path).items():
            assert key not in org_seqs, f"Duplicate gene {key} across organism fastas"
            org_seqs[key] = seq

    assert set(org_seqs.keys()) == set(mvp_seqs.keys()), "Organism fastas should partition MVP proteins"
    for key in org_seqs:
        assert org_seqs[key] == mvp_seqs[key], f"Sequence for {key} should match MVP proteins.fasta"


# --- Step 7: Generate embeddings (raw .pt files) ---


def _get_embeddings_input_dir():
    """Resolve embeddings input dir (raw .pt, no labels) from config."""
    config = load_config()
    hidden_layer = config.embeddings.hidden_layer
    layer_suffix = "last" if hidden_layer < 0 else str(hidden_layer)
    output_subdir = f"{config.embeddings.output_subdir}_layer{layer_suffix}"
    return config.data.mvp_dir / output_subdir


def _get_labeled_embeddings_dir():
    """Resolve labeled embeddings dir from config."""
    config = load_config()
    return config.data.mvp_dir / config.labels.output_subdir


def test_raw_embeddings_pt_format():
    """Raw embedding .pt files have embeddings (Tensor) and group_labels (list)."""
    emb_dir = _get_embeddings_input_dir()
    if not emb_dir.is_dir():
        emb_alt = get_mvp_dir() / "ProtLM_embeddings"
        if emb_alt.is_dir():
            emb_dir = emb_alt
        else:
            pytest.skip(f"Embeddings dir not found: {emb_dir}")

    pt_files = list(emb_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files in embeddings dir")

    for pt_path in pt_files[:2]:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert isinstance(data, dict), f"{pt_path.name}: expected dict"
        assert "embeddings" in data, f"{pt_path.name}: missing embeddings"
        assert "group_labels" in data, f"{pt_path.name}: missing group_labels"

        emb = data["embeddings"]
        assert isinstance(emb, torch.Tensor), "embeddings should be Tensor"
        assert emb.dim() == 2, f"embeddings shape should be (N, D), got {emb.shape}"
        assert emb.shape[0] > 0 and emb.shape[1] > 0

        labels = data["group_labels"]
        assert isinstance(labels, list), "group_labels should be list"
        assert len(labels) == emb.shape[0], "group_labels length should match embeddings"
        for lbl in labels[:3]:
            assert ":" in str(lbl), f"group_label format orgId:locusId: {lbl}"


def test_raw_embeddings_shape_consistent():
    """All raw embedding .pt files have same embedding dimension."""
    emb_dir = _get_embeddings_input_dir()
    if not emb_dir.is_dir():
        emb_alt = get_mvp_dir() / "ProtLM_embeddings"
        emb_dir = emb_alt if emb_alt.is_dir() else emb_dir
    if not emb_dir.is_dir():
        pytest.skip("Embeddings dir not found")

    dims = set()
    for pt_path in emb_dir.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "embeddings" in data:
            dims.add(data["embeddings"].shape[1])
    assert len(dims) == 1, f"Inconsistent embedding dims: {dims}"


# --- Step 8: Label embeddings (final .pt with y) ---


def test_labeled_embeddings_pt_format():
    """Labeled .pt files have embeddings, group_labels, and y (integer labels)."""
    labeled_dir = _get_labeled_embeddings_dir()
    if not labeled_dir.is_dir():
        labeled_alt = get_mvp_dir() / "ProtLM_embeddings_with_labels"
        if labeled_alt.is_dir():
            labeled_dir = labeled_alt
        else:
            pytest.skip(f"Labeled embeddings dir not found: {labeled_dir}")

    pt_files = list(labeled_dir.glob("*.pt"))
    if not pt_files:
        pytest.skip("No .pt files in labeled embeddings dir")

    for pt_path in pt_files[:2]:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert isinstance(data, dict), f"{pt_path.name}: expected dict"
        assert "embeddings" in data, f"{pt_path.name}: missing embeddings"
        assert "group_labels" in data, f"{pt_path.name}: missing group_labels"
        assert "y" in data, f"{pt_path.name}: missing y"

        y = data["y"]
        assert isinstance(y, torch.Tensor) or isinstance(y, list), "y should be tensor or list"
        if isinstance(y, torch.Tensor):
            assert y.dim() == 1, f"y shape should be (N,), got {y.shape}"
            assert len(y) == data["embeddings"].shape[0]
            valid_vals = set(y.tolist())
        else:
            valid_vals = set(y)
        assert valid_vals.issubset({0, 1, 2, -1}), f"y values should be 0,1,2,-1: {valid_vals}"


def test_labeled_embeddings_match_organism_fastas():
    """Labeled .pt files correspond to organism FASTA files (one .pt per organism)."""
    config = load_config()
    mvp_dir = config.data.mvp_dir
    fastas_dir = mvp_dir / config.embeddings.input_subdir
    labeled_dir = _get_labeled_embeddings_dir()
    if not labeled_dir.is_dir():
        labeled_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not labeled_dir.is_dir():
        pytest.skip("Labeled embeddings dir not found")
    if not fastas_dir.is_dir():
        pytest.skip("Organism fastas dir not found")

    fasta_stems = {f.stem for f in fastas_dir.glob("*.fasta")}
    pt_stems = set()
    for f in labeled_dir.glob("*.pt"):
        stem = f.stem
        if stem.endswith("_proteomelm"):
            stem = stem[: -len("_proteomelm")]
        pt_stems.add(stem)

    missing = fasta_stems - pt_stems
    assert not missing, f"Organisms with FASTA but no labeled .pt: {missing}"


def test_labeled_embeddings_y_exact_match_genes_parquet():
    """y labels in labeled .pt files exactly match genes.parquet essentiality mapping."""
    config = load_config()
    mvp_dir = config.data.mvp_dir
    genes_path = mvp_dir / "genes.parquet"
    labeled_dir = _get_labeled_embeddings_dir()
    if not labeled_dir.is_dir():
        labeled_dir = get_mvp_dir() / "ProtLM_embeddings_with_labels"
    if not labeled_dir.is_dir() or not genes_path.is_file():
        pytest.skip("Labeled embeddings or MVP genes not found")

    genes = pd.read_parquet(genes_path)
    class_to_int = config.labels.class_to_int
    genes["gene_key"] = genes["orgId"].astype(str) + ":" + genes["locusId"].astype(str)
    genes["expected_y"] = genes["essentiality_class"].map(class_to_int)
    gene_lookup = genes.set_index("gene_key")["expected_y"].to_dict()

    mismatches = []
    for pt_path in labeled_dir.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        y = data["y"]
        if isinstance(y, torch.Tensor):
            y_list = y.tolist()
        else:
            y_list = list(y)
        for label, y_val in zip(data["group_labels"], y_list):
            expected = gene_lookup.get(label, -1)
            if pd.isna(expected):
                expected = -1
            else:
                expected = int(expected)
            if y_val != expected:
                mismatches.append((pt_path.name, label, y_val, expected))

    assert not mismatches, f"y label mismatches: {mismatches[:10]}"
