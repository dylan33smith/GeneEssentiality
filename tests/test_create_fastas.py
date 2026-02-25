"""Tests for src/pipeline/create_fastas.py — FASTA parsing and splitting."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.create_fastas import parse_fasta_header, split_fasta_by_organism


class TestParseFastaHeader:
    def test_standard(self):
        org, locus = parse_fasta_header(">ANA3:7022496 some description")
        assert org == "ANA3"
        assert locus == "7022496 some description"

    def test_no_desc(self):
        org, locus = parse_fasta_header(">ANA3:7022496")
        assert org == "ANA3"
        assert locus == "7022496"

    def test_malformed(self):
        org, locus = parse_fasta_header(">nodescription")
        assert org is None
        assert locus is None


def test_split_fasta_by_organism(tmp_path: Path):
    fasta_content = (
        ">OrgA:gene1\nACGT\nTGCA\n"
        ">OrgA:gene2\nAAAA\n"
        ">OrgB:gene3\nCCCC\n"
    )
    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(fasta_content)

    output_dir = tmp_path / "organism_fastas"
    counts = split_fasta_by_organism(fasta_path, output_dir)

    assert counts["OrgA"] == 2
    assert counts["OrgB"] == 1
    assert (output_dir / "OrgA.fasta").exists()
    assert (output_dir / "OrgB.fasta").exists()

    orga_text = (output_dir / "OrgA.fasta").read_text()
    assert ">OrgA:gene1" in orga_text
    assert ">OrgA:gene2" in orga_text
