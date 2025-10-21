"""
Improved Readout implementations closer to LSM philosophy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearReadout(nn.Module):
    """
    Pure linear readout (most LSM-like)
    Uses last layer only
    
    Parameters: ~1.24B
    """
    
    def __init__(self, d_inner, d_state, vocab_size):
        super().__init__()
        input_dim = d_inner * d_state
        self.readout = nn.Linear(input_dim, vocab_size)
        
    def forward(self, all_hidden_states):
        # Use only last layer
        last_hidden = all_hidden_states[-1]  # (batch, seq, d_inner, d_state)
        batch, seq_len = last_hidden.shape[:2]
        
        # Flatten
        hidden_flat = last_hidden.reshape(batch, seq_len, -1)
        
        # Pure linear projection
        return self.readout(hidden_flat)


class SelectedLayersReadout(nn.Module):
    """
    Linear readout from selected layers
    Balance between information and parameters
    
    Parameters: ~5B (for 4 layers)
    """
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size,
                 selected_layers=None):
        super().__init__()
        
        # Default: select evenly spaced layers
        if selected_layers is None:
            n_select = min(4, n_layers)  # Select 4 layers
            step = n_layers // n_select
            selected_layers = [i * step for i in range(n_select)]
            if selected_layers[-1] != n_layers - 1:
                selected_layers[-1] = n_layers - 1  # Always include last
        
        self.selected_layers = selected_layers
        input_dim = d_inner * d_state * len(selected_layers)
        
        self.readout = nn.Linear(input_dim, vocab_size)
        
    def forward(self, all_hidden_states):
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Select and concatenate layers at each timestep
        outputs = []
        for t in range(seq_len):
            hidden_t = [
                all_hidden_states[i][:, t, :, :].reshape(batch, -1)
                for i in self.selected_layers
            ]
            hidden_concat = torch.cat(hidden_t, dim=-1)
            output_t = self.readout(hidden_concat)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=1)


class BottleneckReadout(nn.Module):
    """
    Readout with dimensionality reduction bottleneck
    More efficient while preserving information
    
    Parameters: ~330M
    """
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size,
                 bottleneck_dim=512):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        # Per-layer projection to bottleneck
        self.layer_projections = nn.ModuleList([
            nn.Linear(input_dim, bottleneck_dim)
            for _ in range(n_layers)
        ])
        
        # Final readout from summed bottleneck
        self.final_readout = nn.Linear(bottleneck_dim, vocab_size)
        
    def forward(self, all_hidden_states):
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Project each layer to bottleneck
        bottlenecks = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            hidden_flat = hidden_states.reshape(batch, seq_len, -1)
            bottleneck = self.layer_projections[layer_idx](hidden_flat)
            bottlenecks.append(bottleneck)
        
        # Sum across layers
        aggregated = torch.stack(bottlenecks, dim=0).sum(dim=0)
        
        # Final readout
        return self.final_readout(aggregated)


class WeightedLayersReadout(nn.Module):
    """
    Per-layer linear readout with learnable weighted sum
    Allows model to learn which layers are important
    
    Parameters: ~30B (but most efficient training)
    """
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        # Each layer gets its own readout
        self.layer_readouts = nn.ModuleList([
            nn.Linear(input_dim, vocab_size)
            for _ in range(n_layers)
        ])
        
        # Learnable weights for combining layers
        self.layer_weights = nn.Parameter(torch.ones(n_layers))
        
    def forward(self, all_hidden_states):
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Process each layer independently
        layer_outputs = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            hidden_flat = hidden_states.reshape(batch, seq_len, -1)
            layer_out = self.layer_readouts[layer_idx](hidden_flat)
            layer_outputs.append(layer_out)
        
        # Weighted sum
        stacked = torch.stack(layer_outputs, dim=0)  # (n_layers, batch, seq, vocab)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        
        return (stacked * weights).sum(dim=0)


class EfficientMeanPoolReadout(nn.Module):
    """
    Current implementation but optimized
    Mean pooling + 2-layer MLP (simpler than before)
    
    Parameters: ~77M
    """
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size,
                 bottleneck_dim=1024):
        super().__init__()
        
        input_dim = d_inner * d_state
        
        self.readout = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, vocab_size)
        )
        
    def forward(self, all_hidden_states):
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Mean pool across layers
        stacked = torch.stack(all_hidden_states, dim=0)
        pooled = stacked.mean(dim=0)
        pooled_flat = pooled.reshape(batch, seq_len, -1)
        
        return self.readout(pooled_flat)


# Recommendation table
READOUT_OPTIONS = {
    'linear': {
        'class': LinearReadout,
        'params': '1.24B',
        'lsm_philosophy': 'Excellent',
        'information': 'Medium (last layer only)',
        'training': 'Slow (large linear layer)',
        'recommended_for': 'Most LSM-like, if you have memory'
    },
    'selected_layers': {
        'class': SelectedLayersReadout,
        'params': '~5B',
        'lsm_philosophy': 'Good',
        'information': 'Good (4 layers)',
        'training': 'Slow',
        'recommended_for': 'Good balance of philosophy and information'
    },
    'bottleneck': {
        'class': BottleneckReadout,
        'params': '330M',
        'lsm_philosophy': 'Fair',
        'information': 'Good (all layers)',
        'training': 'Medium',
        'recommended_for': 'Information preservation with reasonable size'
    },
    'weighted': {
        'class': WeightedLayersReadout,
        'params': '30B',
        'lsm_philosophy': 'Good',
        'information': 'Excellent (all layers, learned weights)',
        'training': 'Slow',
        'recommended_for': 'Learn layer importance, if you have memory'
    },
    'efficient_mean': {
        'class': EfficientMeanPoolReadout,
        'params': '77M',
        'lsm_philosophy': 'Fair',
        'information': 'Fair (mean pooling)',
        'training': 'Fast',
        'recommended_for': 'Quick experiments, proof of concept'
    }
}


def get_readout(readout_type, d_inner, d_state, n_layers, vocab_size, **kwargs):
    """
    Factory function to get readout layer
    
    Args:
        readout_type: One of ['linear', 'selected_layers', 'bottleneck', 
                              'weighted', 'efficient_mean']
        d_inner: Inner dimension
        d_state: State dimension
        n_layers: Number of layers
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for specific readout types
        
    Returns:
        Readout module
    """
    if readout_type == 'linear':
        return LinearReadout(d_inner, d_state, vocab_size)
    elif readout_type == 'selected_layers':
        return SelectedLayersReadout(d_inner, d_state, n_layers, vocab_size, **kwargs)
    elif readout_type == 'bottleneck':
        return BottleneckReadout(d_inner, d_state, n_layers, vocab_size, **kwargs)
    elif readout_type == 'weighted':
        return WeightedLayersReadout(d_inner, d_state, n_layers, vocab_size)
    elif readout_type == 'efficient_mean':
        return EfficientMeanPoolReadout(d_inner, d_state, n_layers, vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown readout type: {readout_type}")
