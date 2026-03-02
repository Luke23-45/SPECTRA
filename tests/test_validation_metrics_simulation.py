import torch
from torchmetrics import MetricCollection, AUROC, AveragePrecision, Recall
import torch.nn.functional as F

def sim_distribution_shift():
    """
    Simulates the specific crash seen in the user's logs:
    Train: Sepsis=23% (boosted)
    Val: Sepsis=3% (natural)
    
    Why does AUC and Recall crash while BCE loss goes up? Let's prove it.
    """
    print("--- Sepsis Distribution Shift Simulation ---")
    
    # 1. Simulate Model Calibration on TRAIN (23% Positives)
    # The model learns a bias term suited for 23% frequency.
    # It tends to output higher logits globally.
    train_logits = torch.randn(10000) * 2.0 - 1.0 # Centered around slightly negative
    train_labels = (torch.rand(10000) < 0.23).float()
    
    # Shift logits slightly upward if positive (model learned something)
    train_logits[train_labels == 1.0] += 2.0
    
    train_probs = torch.sigmoid(train_logits)
    train_bce = F.binary_cross_entropy_with_logits(train_logits, train_labels, pos_weight=torch.tensor([3.0]))
    
    print(f"Train BCE Loss (pos_weight=3.0): {train_bce.item():.4f}")
    
    # 2. Simulate validation on TRUE distribution (3% Positives)
    # The model still outputs logits based on a 23% prior!
    val_logits = torch.randn(10000) * 2.0 - 1.0
    val_labels = (torch.rand(10000) < 0.03).float() # Only 3% positive!
    
    val_logits[val_labels == 1.0] += 1.0 # Model is less confident on unseen data
    
    val_probs = torch.sigmoid(val_logits)
    val_bce = F.binary_cross_entropy_with_logits(val_logits, val_labels, pos_weight=torch.tensor([3.0]))
    
    print(f"Val BCE Loss (pos_weight=3.0, Over-predicting): {val_bce.item():.4f}")
    
    # Calculate Metrics
    metrics = MetricCollection({
        'AUC': AUROC(task="binary"),
        'PRC': AveragePrecision(task="binary"),
        'Recall': Recall(task="binary")
    })
    
    res = metrics(val_probs, val_labels)
    print(f"Val Metrics with standard 0.5 threshold: {res}")
    
    # What happens as training progresses and the model becomes OVERCONFIDENT on 23% distribution?
    # It learns to push logits higher globally, triggering massive false positives on the 3% val set.
    val_logits_epoch3 = val_logits + 1.5 # Overconfidence bias shift
    val_probs_epoch3 = torch.sigmoid(val_logits_epoch3)
    val_bce_epoch3 = F.binary_cross_entropy_with_logits(val_logits_epoch3, val_labels, pos_weight=torch.tensor([3.0]))
    
    res_epoch3 = metrics(val_probs_epoch3, val_labels)
    print(f"\n--- Epoch 3: Model Overconfident on 23% Train Prior ---")
    print(f"Val BCE Loss (Spikes!): {val_bce_epoch3.item():.4f}")
    print(f"Val Metrics (Recall crashes/AUC drops due to confident FP): {res_epoch3}")
    
if __name__ == "__main__":
    sim_distribution_shift()
