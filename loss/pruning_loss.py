"""
L1 Loss (Pruning Loss)
L1 regularization on ALL layer outputs to induce sparsity
"""

import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """
    L1 regularization on all layer outputs
    
    Encourages the model to use sparse representations in all layer outputs,
    mimicking the sparse coding observed in biological neural systems.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, all_layer_outputs):
        """
        Args:
            all_layer_outputs: list of (batch, seq_len, d_model) tensors (24 items)
            
        Returns:
            loss: scalar L1 norm
        """
        total_l1 = 0.0
        num_layers = len(all_layer_outputs)
        
        for layer_output in all_layer_outputs:
            # L1 norm: sum of absolute values
            l1 = torch.abs(layer_output).mean()
            total_l1 += l1
        
        # Average over layers
        avg_l1 = total_l1 / num_layers
        
        return avg_l1


class L1LossWithMetrics(L1Loss):
    """L1 loss with sparsity metrics - on all 24 layers"""
    
    def forward(self, all_layer_outputs, threshold=1e-3):
        """
        Args:
            all_layer_outputs: list of layer outputs from all layers
                               [(batch, seq_len, d_model), ...] (24 items)
        Returns:
            loss: scalar L1 norm
            metrics: dict with sparsity statistics
        """
        total_l1 = 0.0
        total_near_zero = 0.0
        total_l0 = 0.0
        num_layers = len(all_layer_outputs)
        
        for layer_output in all_layer_outputs:
            # L1 loss
            l1 = torch.abs(layer_output).mean()
            total_l1 += l1
            
            # Near-zero ratio
            near_zero = (layer_output.abs() < threshold).float().mean()
            total_near_zero += near_zero
            
            # L0 norm
            l0 = (layer_output == 0).float().mean()
            total_l0 += l0
        
        loss = total_l1 / num_layers
        
        metrics = {
            'avg_near_zero_ratio': (total_near_zero / num_layers).item(),
            'avg_l0_sparsity': (total_l0 / num_layers).item(),
            'l1_loss': loss.item()
        }
        
        return loss, metrics


# Keep old names for backward compatibility
PruningLoss = L1Loss
PruningLossWithMetrics = L1LossWithMetrics


class AdaptivePruningLoss(nn.Module):
    """
    Adaptive pruning loss with target sparsity
    
    Adjusts L1 penalty based on current sparsity level to reach target.
    """
    
    def __init__(self, target_sparsity=0.5, adaptation_rate=0.01):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.register_buffer('lambda_', torch.tensor(1.0))
    
    def forward(self, sampled_layer_outputs, threshold=1e-3):
        """
        Args:
            sampled_layer_outputs: list of layer outputs from sampled layers
        Returns:
            loss: scalar adaptive L1 norm
            metrics: dict with sparsity statistics and lambda
        """
        # Calculate current sparsity
        total_near_zero = 0.0
        num_layers = len(sampled_layer_outputs)
        
        for layer_output in sampled_layer_outputs:
            near_zero = (layer_output.abs() < threshold).float().mean()
            total_near_zero += near_zero
        
        current_sparsity = total_near_zero / num_layers
        
        # Adapt lambda based on sparsity gap
        sparsity_gap = self.target_sparsity - current_sparsity
        self.lambda_ = self.lambda_ + self.adaptation_rate * sparsity_gap
        self.lambda_ = torch.clamp(self.lambda_, min=0.1, max=10.0)
        
        # Calculate L1 loss
        total_l1 = 0.0
        for layer_output in sampled_layer_outputs:
            l1 = torch.abs(layer_output).mean()
            total_l1 += l1
        
        avg_l1 = total_l1 / num_layers
        loss = self.lambda_ * avg_l1
        
        metrics = {
            'current_sparsity': current_sparsity.item(),
            'target_sparsity': self.target_sparsity,
            'lambda': self.lambda_.item(),
            'l1_loss': avg_l1.item()
        }
        
        return loss, metrics
