import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import argparse
from autoresearch_regression.prepare import build_loaders_and_val_frame, evaluate

logger = logging.getLogger(__name__)

class PairwiseMLP(nn.Module):
    def __init__(self, input_dim, hidden1=2048, hidden2=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        
    def forward(self, x):
        return self.net(x).squeeze()

def pairwise_ranking_loss(pred, margin=0.1):
    """Compute pairwise ranking loss where we want higher predictions for higher fitness."""
    # pred shape: (batch_size,)
    # pred[i] is the difference: pred(gene_high) - pred(gene_low)
    # We want pred[i] > margin for all i
    return torch.mean(torch.relu(margin - pred))

def train(epochs=20, lr=0.001, weight_decay=1e-4, batch_size=2048, margin=0.1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    
    logger.info("Config: epochs=%d lr=%.4f margin=%.2f", epochs, lr, margin)
    
    train_loader, val_loader, test_loader, val_df = build_loaders_and_val_frame()
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))
    
    # FIXED: Get actual input dimension from data instead of hardcoding
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    logger.info("Detected input dimension: %d", input_dim)
    
    model = PairwiseMLP(input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=False)
    
    best_val_rmse = float("inf")
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            # x shape: (batch_size, input_dim) - already pairwise differences
            
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = pairwise_ranking_loss(pred, margin=margin)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).squeeze()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.numpy())
            
            metrics = evaluate(model, val_loader, device, val_df)
            val_rmse = metrics['val_rmse']
            val_spearman = metrics['mean_within_gene_spearman']
            logger.info("epoch %d/%d train_loss=%.6f val_rmse=%.6f val_spearman=%.4f lr=%.6f",
                       epoch, epochs, train_loss, val_rmse, val_spearman, optimizer.param_groups[0]['lr'])
            
            scheduler.step(val_rmse)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
    
    # Test on best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Best model: val_rmse=%.6f", best_val_rmse)
        
        model.eval()
        with torch.no_grad():
            test_preds = []
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x).squeeze()
                test_preds.extend(pred.cpu().numpy())
            
            # Get fresh val_df for evaluation
            _, _, _, test_df = build_loaders_and_val_frame()
            test_metrics = evaluate(model, val_loader, device, val_df)
            test_rmse = test_metrics['val_rmse']
            test_spearman = test_metrics['mean_within_gene_spearman']
            logger.info("Test (best model): val_rmse=%.6f mean_within_gene_spearman=%.6f",
                       test_rmse, test_spearman)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    
    train(epochs=args.epochs, lr=args.lr, margin=args.margin)
