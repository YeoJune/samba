"""
Auxiliary Loss
Cross-entropy loss on auxiliary decoder predictions
Ensures the dense memory vector contains meaningful semantic information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxLoss(nn.Module):
    """
    Cross-entropy loss on auxiliary logits from decoder
    
    This ensures the dense vector aggregated from sparse hidden states
    contains meaningful semantic information.
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, aux_logits, targets):
        """
        Args:
            aux_logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            
        Returns:
            loss: scalar
        """
        batch, seq_len, vocab_size = aux_logits.shape
        
        # Reshape for cross entropy
        aux_logits_flat = aux_logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        loss = self.ce_loss(aux_logits_flat, targets_flat)
        
        return loss


class AuxLossWithMetrics(AuxLoss):
    """Auxiliary loss with additional metrics"""
    
    def forward(self, aux_logits, targets):
        """
        Returns:
            loss: scalar
            metrics: dict with accuracy and perplexity
        """
        loss = super().forward(aux_logits, targets)
        
        # Calculate accuracy
        batch, seq_len, vocab_size = aux_logits.shape
        predictions = aux_logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        metrics = {
            'aux_accuracy': accuracy.item(),
            'aux_perplexity': perplexity.item()
        }
        
        return loss, metrics


# Keep old names for backward compatibility
ReadoutLoss = AuxLoss
ReadoutLossWithMetrics = AuxLossWithMetrics
