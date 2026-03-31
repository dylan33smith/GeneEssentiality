"""Shared models and loss functions for Phase 2 experiments.

Imported by individual experiment scripts to avoid duplicating architecture
and ranking-loss code across 11 files.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden1: int = 2048, hidden2: int = 512, dropout: float = 0.3) -> nn.Module:
    """Standard 2-hidden-layer MLP used by flat-MLP experiments."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden2, 1),
    )


class ResidualModel(nn.Module):
    """gene_mean(embedding) + offset(embedding, condition).

    Decomposes prediction into a gene-level baseline fitness (from the
    ProteomeLM embedding alone) plus a condition-dependent offset.
    """

    def __init__(self, gene_embed_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gene_embed_dim = gene_embed_dim
        self.gene_head = nn.Sequential(
            nn.Linear(gene_embed_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 1),
        )
        full_dim = gene_embed_dim + condition_dim
        self.offset_head = nn.Sequential(
            nn.Linear(full_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gene_emb = x[:, :self.gene_embed_dim]
        gene_mean = self.gene_head(gene_emb)
        offset = self.offset_head(x)
        return gene_mean + offset


class WideOffsetResidualModel(nn.Module):
    """Like ResidualModel but with a wider/deeper offset head.

    gene_head is unchanged; offset_head gets more capacity (2048->1024->256->1)
    to better model gene x condition interactions.
    """

    def __init__(self, gene_embed_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gene_embed_dim = gene_embed_dim
        self.gene_head = nn.Sequential(
            nn.Linear(gene_embed_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 1),
        )
        full_dim = gene_embed_dim + condition_dim
        self.offset_head = nn.Sequential(
            nn.Linear(full_dim, 2048), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gene_emb = x[:, :self.gene_embed_dim]
        gene_mean = self.gene_head(gene_emb)
        offset = self.offset_head(x)
        return gene_mean + offset


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def listwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gene_idx: torch.Tensor,
) -> torch.Tensor:
    """KL-div between softmax(target) and softmax(pred) within each gene group.

    Pre-filters to genes with >= 2 rows. Returns scalar 0 if no gene has >= 2.
    """
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    unique_genes, _inverse, counts = torch.unique(gene_idx, return_inverse=True, return_counts=True)
    multi_mask = counts >= 2
    if not multi_mask.any():
        return torch.tensor(0.0, device=pred.device)

    multi_genes = unique_genes[multi_mask]
    total_loss = torch.tensor(0.0, device=pred.device)
    n_groups = 0
    for g in multi_genes:
        mask = gene_idx == g
        p = pred[mask]
        t = target[mask]
        log_q = F.log_softmax(p, dim=0)
        p_target = F.softmax(t, dim=0)
        total_loss = total_loss + F.kl_div(log_q, p_target, reduction="batchmean")
        n_groups += 1
    return total_loss / n_groups


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gene_idx: torch.Tensor,
    margin: float = 0.1,
    max_pairs_per_gene: int = 0,
    normalize_per_gene: bool = False,
) -> torch.Tensor:
    """Margin ranking loss over within-gene pairs.

    Args:
        max_pairs_per_gene: Cap on ordered pairs sampled per gene (0 = unlimited).
            Defaults to env ``PAIR_MAX_PAIRS`` if set, else 512.
        normalize_per_gene: When True, average loss per gene first, then
            average across genes (prevents high-condition genes from dominating).
    """
    if max_pairs_per_gene <= 0:
        max_pairs_per_gene = int(os.environ.get("PAIR_MAX_PAIRS", "512"))

    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    unique_genes, _inverse, counts = torch.unique(gene_idx, return_inverse=True, return_counts=True)
    multi_mask = counts >= 2
    if not multi_mask.any():
        return torch.tensor(0.0, device=pred.device)

    multi_genes = unique_genes[multi_mask]

    if normalize_per_gene:
        gene_losses: list[torch.Tensor] = []
        for g in multi_genes:
            mask = gene_idx == g
            p = pred[mask]
            t = target[mask]
            pi, pj = _ordered_pairs(p, t, max_pairs_per_gene)
            if pi is None:
                continue
            labels = torch.ones_like(pi)
            gene_losses.append(F.margin_ranking_loss(pi, pj, labels, margin=margin))
        if not gene_losses:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(gene_losses).mean()

    all_p_i: list[torch.Tensor] = []
    all_p_j: list[torch.Tensor] = []
    for g in multi_genes:
        mask = gene_idx == g
        p = pred[mask]
        t = target[mask]
        pi, pj = _ordered_pairs(p, t, max_pairs_per_gene)
        if pi is None:
            continue
        all_p_i.append(pi)
        all_p_j.append(pj)

    if not all_p_i:
        return torch.tensor(0.0, device=pred.device)

    p_i = torch.cat(all_p_i)
    p_j = torch.cat(all_p_j)
    labels = torch.ones_like(p_i)
    return F.margin_ranking_loss(p_i, p_j, labels, margin=margin)


def _ordered_pairs(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build (pred_i, pred_j) for strict-order target pairs within one gene."""
    n = len(target)
    idx_i = torch.arange(n, device=pred.device).unsqueeze(1).expand(n, n).reshape(-1)
    idx_j = torch.arange(n, device=pred.device).unsqueeze(0).expand(n, n).reshape(-1)
    keep = target[idx_i] > target[idx_j]
    if keep.sum() == 0:
        return None, None
    valid_i = idx_i[keep]
    valid_j = idx_j[keep]
    if max_pairs > 0 and len(valid_i) > max_pairs:
        sel = torch.randperm(len(valid_i), device=pred.device)[:max_pairs]
        valid_i = valid_i[sel]
        valid_j = valid_j[sel]
    return pred[valid_i], pred[valid_j]
