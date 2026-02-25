"""Unit tests for src/pipeline/extract_data.py using in-memory SQLite.

These tests don't require the real FEBA database on disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.pipeline.extract_data import (
    extract_experiments,
    extract_fitness,
    extract_genes,
    extract_organisms,
    save_parquet,
)


def _build_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the FEBA schema and sample rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE Gene (
            orgId TEXT, locusId TEXT, sysName TEXT, scaffoldId TEXT,
            "begin" INT, "end" INT, type INT, strand TEXT,
            gene TEXT, "desc" TEXT, GC REAL
        )
    """)
    conn.execute("""
        INSERT INTO Gene VALUES
        ('TestOrg', 'g1', 'sys1', 'scaffold1', 100, 400, 1, '+', 'geneA', 'desc A', 0.5),
        ('TestOrg', 'g2', 'sys2', 'scaffold1', 500, 800, 1, '-', 'geneB', 'desc B', 0.6),
        ('TestOrg', 'g3', 'sys3', 'scaffold1', 900, 1200, 2, '+', 'rRNA', 'ribosomal', 0.4)
    """)
    conn.execute("""
        CREATE TABLE Experiment (
            orgId TEXT, expName TEXT, expDesc TEXT, expGroup TEXT,
            mutantLibrary TEXT, media TEXT, mediaStrength TEXT,
            condition_1 TEXT, concentration_1 TEXT, units_1 TEXT,
            condition_2 TEXT, concentration_2 TEXT, units_2 TEXT,
            condition_3 TEXT, concentration_3 TEXT, units_3 TEXT,
            temperature TEXT, pH TEXT, aerobic TEXT, vessel TEXT,
            liquid TEXT, shaking TEXT, cor12 TEXT, maxFit TEXT, nGenerations TEXT
        )
    """)
    conn.execute("""
        INSERT INTO Experiment (orgId, expName, expDesc, expGroup, media, cor12, maxFit, nGenerations)
        VALUES ('TestOrg', 'exp1', 'LB control', 'group1', 'LB', '0.95', '3.5', '8')
    """)
    conn.execute("""
        CREATE TABLE GeneFitness (
            orgId TEXT, locusId TEXT, expName TEXT, fit REAL, t REAL
        )
    """)
    conn.execute("""
        INSERT INTO GeneFitness VALUES
        ('TestOrg', 'g1', 'exp1', -2.0, 5.0),
        ('TestOrg', 'g2', 'exp1', 0.3, 1.5)
    """)
    conn.execute("""
        CREATE TABLE Organism (
            orgId TEXT, division TEXT, genus TEXT, species TEXT,
            strain TEXT, taxonomyId INT
        )
    """)
    conn.execute("""
        INSERT INTO Organism VALUES ('TestOrg', 'Proteobacteria', 'Pseudo', 'aeruginosa', 'PA01', 12345)
    """)
    conn.commit()
    return conn


class TestExtractGenes:
    def test_extracts_protein_coding_only(self):
        conn = _build_test_db()
        genes = extract_genes(conn)
        conn.close()
        assert len(genes) == 2
        assert all(genes["type"] == 1)

    def test_has_gene_length_column(self):
        conn = _build_test_db()
        genes = extract_genes(conn)
        conn.close()
        assert "gene_length" in genes.columns
        assert genes.iloc[0]["gene_length"] == 301


class TestExtractExperiments:
    def test_coerces_numeric_columns(self):
        conn = _build_test_db()
        exps = extract_experiments(conn)
        conn.close()
        assert pd.api.types.is_numeric_dtype(exps["cor12"])
        assert pd.api.types.is_numeric_dtype(exps["maxFit"])


class TestExtractFitness:
    def test_returns_expected_rows(self):
        conn = _build_test_db()
        fitness = extract_fitness(conn)
        conn.close()
        assert len(fitness) == 2
        assert set(fitness.columns) >= {"orgId", "locusId", "expName", "fit", "t"}


class TestExtractOrganisms:
    def test_returns_expected_rows(self):
        conn = _build_test_db()
        orgs = extract_organisms(conn)
        conn.close()
        assert len(orgs) == 1
        assert orgs.iloc[0]["genus"] == "Pseudo"


class TestSaveParquet:
    def test_saves_and_is_readable(self, tmp_path: Path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        save_parquet(df, tmp_path, "test_table")
        result = pd.read_parquet(tmp_path / "test_table.parquet")
        pd.testing.assert_frame_equal(df, result)
