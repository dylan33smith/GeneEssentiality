"""Model architecture wrappers.

Usage:
    from wrappers.models import mlp, residual
    
    # Baseline MLP
    model = mlp.GeneEssentialityMLP(input_dim=1377, hidden1=2048, hidden2=512)
    
    # Residual model (gene_mean + offset)
    model = residual.ResidualGeneModel(input_dim=1377, hidden1=2048, hidden2=512)
"""

from .mlp import GeneEssentialityMLP

__all__ = ["GeneEssentialityMLP"]
