"""Tests for src/model/dataset.py — EssentialityDataset and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model.dataset import (
    EssentialityDataset,
    _org_id_from_pt_filename,
    create_datasets_from_config,
    resolve_embeddings_dir,
)


class TestOrgIdFromPtFilename:
    def test_standard_suffix(self):
        assert _org_id_from_pt_filename("ANA3_proteomelm.pt") == "ANA3"

    def test_no_suffix(self):
        assert _org_id_from_pt_filename("ANA3.pt") == "ANA3"


class TestResolveEmbeddingsDir:
    def test_default(self):
        path = resolve_embeddings_dir(None)
        assert path.name == "ProtLM_embeddings_with_labels"

    def test_custom(self, tmp_path: Path):
        custom = str(tmp_path / "custom_dir")
        path = resolve_embeddings_dir(custom)
        assert path == Path(custom).resolve()


class TestEssentialityDataset:
    def test_loads_correct_organisms(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"])
        labels = ds.get_labels()
        assert all(l >= 0 for l in labels.tolist())

    def test_excludes_no_data(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"], exclude_no_data=True)
        labels = ds.get_labels()
        assert (labels == -1).sum() == 0
        assert len(ds) == 9

    def test_includes_no_data(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"], exclude_no_data=False)
        labels = ds.get_labels()
        assert (labels == -1).sum() == 1
        assert len(ds) == 10

    def test_len(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA", "OrgB"], exclude_no_data=True)
        assert len(ds) == 9 + 7

    def test_getitem_types(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"])
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_embedding_dim(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"])
        assert ds.embedding_dim == 8

    def test_get_labels_shape(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(sample_embeddings_dir, organism_ids=["OrgA"])
        labels = ds.get_labels()
        assert labels.dim() == 1
        assert len(labels) == len(ds)

    def test_empty_organism_raises(self, sample_embeddings_dir: Path):
        with pytest.raises(ValueError, match="No data loaded"):
            EssentialityDataset(sample_embeddings_dir, organism_ids=["NonExistent"])


class TestLabelMap:
    """Test binary label remapping (merging always_essential + conditional)."""

    BINARY_MAP = {0: 0, 1: 0, 2: 1}

    def test_labels_remapped(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
            label_map=self.BINARY_MAP,
        )
        labels = ds.get_labels()
        assert set(labels.tolist()).issubset({0, 1})

    def test_n_classes_is_two(self, sample_embeddings_dir: Path):
        ds = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
            label_map=self.BINARY_MAP,
        )
        assert ds.n_classes == 2

    def test_sample_count_unchanged(self, sample_embeddings_dir: Path):
        ds_3class = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
        )
        ds_binary = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
            label_map=self.BINARY_MAP,
        )
        assert len(ds_binary) == len(ds_3class)

    def test_essential_count_is_sum_of_ae_and_cond(self, sample_embeddings_dir: Path):
        ds_3class = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
        )
        ds_binary = EssentialityDataset(
            sample_embeddings_dir, organism_ids=["OrgA"],
            label_map=self.BINARY_MAP,
        )
        original = ds_3class.get_labels()
        n_ae = (original == 0).sum().item()
        n_cond = (original == 1).sum().item()
        binary = ds_binary.get_labels()
        n_essential = (binary == 0).sum().item()
        assert n_essential == n_ae + n_cond


class TestCreateDatasetsFromConfig:
    def test_returns_three_datasets(self, sample_model_config_yaml: Path):
        train_ds, val_ds, test_ds = create_datasets_from_config(sample_model_config_yaml)
        assert isinstance(train_ds, EssentialityDataset)
        assert isinstance(val_ds, EssentialityDataset)
        assert isinstance(test_ds, EssentialityDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_binary_config_with_label_map(
        self, sample_embeddings_dir: Path, tmp_path: Path,
    ):
        config = {
            "data": {"input_dir": str(sample_embeddings_dir)},
            "classification": {
                "label_map": {0: 0, 1: 0, 2: 1},
            },
            "split": {
                "train_organisms": ["OrgA"],
                "val_organisms": ["OrgB"],
                "test_organisms": ["OrgC"],
            },
        }
        config_path = tmp_path / "model_binary.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        train_ds, val_ds, test_ds = create_datasets_from_config(config_path)
        assert train_ds.n_classes == 2
        assert set(train_ds.get_labels().tolist()).issubset({0, 1})
