"""PyTorch Dataset for essentiality classification from ProteomeLM embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import yaml
from torch.utils.data import Dataset

from src.data_io import get_data_dir, get_mvp_dir


def resolve_embeddings_dir(config_dir: str | Path | None = None) -> Path:
    """Resolve embeddings directory from config path or default.

    Args:
        config_dir: Path from config (e.g. 'data/mvp/ProtLM_embeddings_with_labels/').
            If None, uses default relative to project root.

    Returns:
        Absolute path to the embeddings directory.
    """
    if config_dir is None:
        return get_mvp_dir() / "ProtLM_embeddings_with_labels"
    root = get_data_dir().parent
    path = Path(config_dir)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _org_id_from_pt_filename(filename: str) -> str:
    """Extract organism ID from .pt filename (e.g. ANA3_proteomelm.pt -> ANA3)."""
    stem = Path(filename).stem
    if stem.endswith("_proteomelm"):
        return stem[: -len("_proteomelm")]
    return stem


class EssentialityDataset(Dataset):
    """Dataset of (embedding, label) pairs for essentiality classification.

    Loads per-organism .pt files from a directory, concatenates them, filters
    out no_data (y == -1), and subsets by organism IDs for train/val/test splits.

    Each .pt file must have keys: embeddings (Tensor [N, D]), group_labels (list),
    y (Tensor [N] with 0=always_essential, 1=conditional, 2=non_essential, -1=no_data).

    An optional ``label_map`` dict remaps integer labels after loading (e.g.
    ``{0: 0, 1: 0, 2: 1}`` to merge always_essential + conditional into class 0).
    """

    def __init__(
        self,
        embeddings_dir: Path | str,
        organism_ids: Iterable[str],
        exclude_no_data: bool = True,
        label_map: dict[int, int] | None = None,
    ) -> None:
        """Build dataset from .pt files for the given organisms.

        Args:
            embeddings_dir: Directory containing *_proteomelm.pt files.
            organism_ids: Set or list of organism IDs to include (e.g. train/val/test).
            exclude_no_data: If True, filter out samples with y == -1 (no_data).
            label_map: Optional mapping from original label int to new label int.
                Applied after no_data filtering. Use to merge classes (e.g. binary).
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.organism_ids = set(organism_ids)
        self.exclude_no_data = exclude_no_data
        self.label_map = label_map

        self._embeddings: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and concatenate .pt files for the requested organisms."""
        pt_files = sorted(self.embeddings_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt files found in {self.embeddings_dir}"
            )

        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for pt_path in pt_files:
            org_id = _org_id_from_pt_filename(pt_path.name)
            if org_id not in self.organism_ids:
                continue

            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not isinstance(data, dict):
                continue
            if "embeddings" not in data or "y" not in data:
                continue

            emb = data["embeddings"]
            y = data["y"]
            if isinstance(y, list):
                y = torch.tensor(y, dtype=torch.long)

            if self.exclude_no_data:
                mask = y >= 0
                emb = emb[mask]
                y = y[mask]

            all_embeddings.append(emb)
            all_labels.append(y)

        if not all_embeddings:
            raise ValueError(
                f"No data loaded for organisms {self.organism_ids}. "
                f"Check that .pt files exist and organism IDs match filenames."
            )

        self._embeddings = torch.cat(all_embeddings, dim=0).float()
        self._labels = torch.cat(all_labels, dim=0)

        if self.label_map is not None:
            remapped = self._labels.clone()
            for src, dst in self.label_map.items():
                remapped[self._labels == src] = dst
            self._labels = remapped

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embeddings[idx], self._labels[idx]

    def get_labels(self) -> torch.Tensor:
        """Return all labels as a 1D tensor of shape (N,)."""
        return self._labels

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (D)."""
        return self._embeddings.shape[1]

    @property
    def n_classes(self) -> int:
        """Number of distinct classes present after any label remapping."""
        return int(self._labels.max().item()) + 1


def create_datasets_from_config(
    config_path: Path | str | None = None,
) -> tuple[EssentialityDataset, EssentialityDataset, EssentialityDataset]:
    """Create train, val, and test datasets from a model YAML config.

    The config may contain an optional ``classification`` section with a
    ``label_map`` dict to remap integer labels (e.g. for binary
    essential-vs-non_essential).

    Args:
        config_path: Path to model config. Defaults to config/model.yaml.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    root = get_data_dir().parent
    if config_path is None:
        config_path = root / "config" / "model.yaml"
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    classification_cfg = cfg.get("classification", {})

    embeddings_dir = resolve_embeddings_dir(data_cfg.get("input_dir"))
    train_orgs = set(split_cfg.get("train_organisms", []))
    val_orgs = set(split_cfg.get("val_organisms", []))
    test_orgs = set(split_cfg.get("test_organisms", []))

    label_map_raw = classification_cfg.get("label_map")
    label_map = (
        {int(k): int(v) for k, v in label_map_raw.items()}
        if label_map_raw
        else None
    )

    ds_kwargs = dict(exclude_no_data=True, label_map=label_map)
    train_ds = EssentialityDataset(
        embeddings_dir=embeddings_dir, organism_ids=train_orgs, **ds_kwargs,
    )
    val_ds = EssentialityDataset(
        embeddings_dir=embeddings_dir, organism_ids=val_orgs, **ds_kwargs,
    )
    test_ds = EssentialityDataset(
        embeddings_dir=embeddings_dir, organism_ids=test_orgs, **ds_kwargs,
    )

    return train_ds, val_ds, test_ds
