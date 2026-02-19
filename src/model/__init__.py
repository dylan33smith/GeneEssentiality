"""Model package: dataset, network, training for essentiality classification."""

from src.model.dataset import (
    EssentialityDataset,
    create_datasets_from_config,
    resolve_embeddings_dir,
)

__all__ = [
    "EssentialityDataset",
    "create_datasets_from_config",
    "resolve_embeddings_dir",
]
