"""Loss function wrappers.

Usage:
    from wrappers.losses import mse, pairwise, combined
    
    # Standard MSE
    loss_fn = mse.mse_loss_fn
    
    # Pairwise ranking loss
    loss_fn = pairwise.PairwiseRankingLoss()
"""

import torch.nn as nn

mse_loss_fn = nn.MSELoss()

__all__ = ["mse_loss_fn"]
