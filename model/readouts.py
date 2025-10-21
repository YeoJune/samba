"""
Improved Readout implementations closer to LSM philosophy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReadout(nn.Module):
    """
    Pure linear readout (most LSM-like)
    Uses only sampled layers
    
    Note: Receives pre-sampled hidden states from Samba
    """
    
    def __init__(self, d_inner, d_state, vocab_size):
        super().__init__()
        input_dim = d_inner * d_state
        self.readout = nn.Linear(input_dim, vocab_size)
        
    def forward(self, sampled_hidden_states):
        # Use only last sampled layer
        last_hidden = sampled_hidden_states[-1]  # (batch, seq, d_inner, d_state)
        batch, seq_len = last_hidden.shape[:2]
        
        # Flatten
        hidden_flat = last_hidden.reshape(batch, seq_len, -1)
        
        # Pure linear projection
        return self.readout(hidden_flat)


class SelectedLayersReadout(nn.Module):
    """
    Linear readout from selected sampled layers
    
    Note: Receives pre-sampled hidden states from Samba
    Can optionally sub-sample further if needed
    """
    
    def __init__(self, d_inner, d_state, vocab_size, use_all_sampled=True):
        super().__init__()
        self.use_all_sampled = use_all_sampled
        
        # If not using all, will use first, middle, last from sampled
        input_dim = d_inner * d_state * (1 if not use_all_sampled else 3)
        
        self.readout = nn.Linear(input_dim, vocab_size)
        
    def forward(self, sampled_hidden_states):
        batch, seq_len = sampled_hidden_states[0].shape[:2]
        
        if self.use_all_sampled:
            # Concatenate all sampled layers
            outputs = []
            for t in range(seq_len):
                hidden_t = [
                    h[:, t, :, :].reshape(batch, -1)
                    for h in sampled_hidden_states
                ]
                hidden_concat = torch.cat(hidden_t, dim=-1)
                output_t = self.readout(hidden_concat)
                outputs.append(output_t)
            return torch.stack(outputs, dim=1)
        else:
            # Use first, middle, last from sampled
            n = len(sampled_hidden_states)
            selected = [sampled_hidden_states[0], 
                       sampled_hidden_states[n//2], 
                       sampled_hidden_states[-1]]
            
            outputs = []
            for t in range(seq_len):
                hidden_t = [h[:, t, :, :].reshape(batch, -1) for h in selected]
                hidden_concat = torch.cat(hidden_t, dim=-1)
                output_t = self.readout(hidden_concat)
                outputs.append(output_t)
            return torch.stack(outputs, dim=1)


class BottleneckReadout(nn.Module):
    """
    Readout with dimensionality reduction bottleneck
    
    Note: Receives pre-sampled hidden states from Samba
    """
    
    def __init__(self, d_inner, d_state, vocab_size, bottleneck_dim=512):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        # Shared projection to bottleneck (applied to each layer)
        self.layer_projection = nn.Linear(input_dim, bottleneck_dim)
        
        # Final readout from summed bottleneck
        self.final_readout = nn.Linear(bottleneck_dim, vocab_size)
        
    def forward(self, sampled_hidden_states):
        batch, seq_len = sampled_hidden_states[0].shape[:2]
        
        # Project each sampled layer to bottleneck
        bottlenecks = []
        for hidden_states in sampled_hidden_states:
            hidden_flat = hidden_states.reshape(batch, seq_len, -1)
            bottleneck = self.layer_projection(hidden_flat)
            bottlenecks.append(bottleneck)
        
        # Sum across layers
        aggregated = torch.stack(bottlenecks, dim=0).sum(dim=0)
        
        # Final readout
        return self.final_readout(aggregated)


class WeightedLayersReadout(nn.Module):
    """
    Per-layer linear readout with learnable weighted sum
    
    Note: Receives pre-sampled hidden states from Samba
    Learns importance of each sampled layer
    """
    
    def __init__(self, d_inner, d_state, vocab_size):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        # Shared readout projection (applied to each layer)
        self.layer_readout = nn.Linear(input_dim, vocab_size)
        
        # Learnable weights (will be sized dynamically based on input)
        # Initialized in first forward pass
        self.layer_weights = None
        
    def forward(self, sampled_hidden_states):
        batch, seq_len = sampled_hidden_states[0].shape[:2]
        n_sampled = len(sampled_hidden_states)
        
        # Initialize weights on first forward pass
        if self.layer_weights is None:
            self.layer_weights = nn.Parameter(torch.ones(n_sampled, device=sampled_hidden_states[0].device))
        
        # Process each sampled layer independently
        layer_outputs = []
        for hidden_states in sampled_hidden_states:
            hidden_flat = hidden_states.reshape(batch, seq_len, -1)
            layer_out = self.layer_readout(hidden_flat)
            layer_outputs.append(layer_out)
        
        # Weighted sum
        stacked = torch.stack(layer_outputs, dim=0)  # (n_sampled, batch, seq, vocab)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        
        return (stacked * weights).sum(dim=0)


class EfficientMeanPoolReadout(nn.Module):
    """
    Mean pooling + 2-layer MLP
    
    Note: Receives pre-sampled hidden states from Samba
    """
    
    def __init__(self, d_inner, d_state, vocab_size, bottleneck_dim=1024):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        self.readout = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, vocab_size)
        )
        
    def forward(self, sampled_hidden_states):
        batch, seq_len = sampled_hidden_states[0].shape[:2]
        
        # Mean pool across sampled layers
        stacked = torch.stack(sampled_hidden_states, dim=0)
        pooled = stacked.mean(dim=0)
        pooled_flat = pooled.reshape(batch, seq_len, -1)
        
        return self.readout(pooled_flat)


# Recommendation table
READOUT_OPTIONS = {
    'linear': {
        'class': LinearReadout,
        'description': 'Pure linear readout from last sampled layer',
        'params': '~1.24B',
        'lsm_philosophy': 'Excellent',
        'information': 'Medium (last sampled layer only)',
        'recommended_for': 'Most LSM-like, simple readout'
    },
    'selected_layers': {
        'class': SelectedLayersReadout,
        'description': 'Linear readout from sampled layers',
        'params': 'Varies with sampled layers',
        'lsm_philosophy': 'Good',
        'information': 'Good (uses sampled layers)',
        'recommended_for': 'Balance of philosophy and information'
    },
    'bottleneck': {
        'class': BottleneckReadout,
        'description': 'Bottleneck readout from sampled layers',
        'params': '~330M / n_layers * n_sampled',
        'lsm_philosophy': 'Fair',
        'information': 'Good (all sampled layers)',
        'recommended_for': 'Information preservation with reasonable size'
    },
    'weighted': {
        'class': WeightedLayersReadout,
        'description': 'Weighted readout learning sampled layer importance',
        'params': 'Shared weights across layers',
        'lsm_philosophy': 'Good',
        'information': 'Excellent (learned weights)',
        'recommended_for': 'Learn layer importance efficiently'
    },
    'efficient_mean': {
        'class': EfficientMeanPoolReadout,
        'description': 'Mean pooling over sampled layers',
        'params': '~77M',
        'lsm_philosophy': 'Fair',
        'information': 'Fair (mean pooling)',
        'recommended_for': 'Quick experiments, proof of concept'
    }
}


def get_readout(readout_type, d_inner, d_state, vocab_size, **kwargs):
    """
    Factory function to get readout layer
    
    Note: All readouts now receive pre-sampled hidden states from Samba.forward
    
    Args:
        readout_type: One of ['linear', 'selected_layers', 'bottleneck', 
                              'weighted', 'efficient_mean']
        d_inner: Inner dimension
        d_state: State dimension
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for specific readout types
        
    Returns:
        Readout module
    """
    if readout_type == 'linear':
        return LinearReadout(d_inner, d_state, vocab_size)
    elif readout_type == 'selected_layers':
        return SelectedLayersReadout(d_inner, d_state, vocab_size, **kwargs)
    elif readout_type == 'bottleneck':
        return BottleneckReadout(d_inner, d_state, vocab_size, **kwargs)
    elif readout_type == 'weighted':
        return WeightedLayersReadout(d_inner, d_state, vocab_size)
    elif readout_type == 'efficient_mean':
        return EfficientMeanPoolReadout(d_inner, d_state, vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown readout type: {readout_type}")
