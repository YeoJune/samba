"""
Readout Loss
Auxiliary loss to ensure the dense readout vector predicts the correct output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadoutLoss(nn.Module):
    """
    Cross-entropy loss on readout logits
    
    This ensures the dense vector aggregated from sparse hidden states
    contains meaningful semantic information.
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, readout_logits, targets):
        """
        Args:
            readout_logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            
        Returns:
            loss: scalar
        """
        batch, seq_len, vocab_size = readout_logits.shape
        
        # Reshape for cross entropy
        readout_logits_flat = readout_logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        loss = self.ce_loss(readout_logits_flat, targets_flat)
        
        return loss


class ReadoutLossWithMetrics(ReadoutLoss):
    """Readout loss with additional metrics"""
    
    def forward(self, readout_logits, targets):
        """
        Returns:
            loss: scalar
            metrics: dict with accuracy and perplexity
        """
        loss = super().forward(readout_logits, targets)
        
        # Calculate accuracy
        batch, seq_len, vocab_size = readout_logits.shape
        predictions = readout_logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        metrics = {
            'readout_accuracy': accuracy.item(),
            'readout_perplexity': perplexity.item()
        }
        
        return loss, metrics
