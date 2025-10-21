"""
Samba: Sparse Mamba (130M parameters)
Mamba with MLP readout for sparse representation learning
"""

import torch
import torch.nn as nn
from .mamba import Mamba


class SambaReadout(nn.Module):
    """MLP readout layer to aggregate hidden states into dense representation"""
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size, hidden_dim=512):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_layers = n_layers
        
        # Calculate input dimension: all hidden states from all layers
        # hidden_states shape per layer: (batch, seq_len, d_inner, d_state)
        input_dim = d_inner * d_state * n_layers
        
        # MLP for readout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
    def forward(self, all_hidden_states):
        """
        all_hidden_states: list of (batch, seq_len, d_inner, d_state) for each layer
        Returns: (batch, seq_len, vocab_size)
        """
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Flatten all hidden states
        # For each timestep, concatenate all hidden states from all layers
        outputs = []
        for t in range(seq_len):
            hidden_t = []
            for layer_hidden in all_hidden_states:
                # Extract hidden state at timestep t
                h_t = layer_hidden[:, t, :, :]  # (batch, d_inner, d_state)
                h_t_flat = h_t.reshape(batch, -1)  # (batch, d_inner * d_state)
                hidden_t.append(h_t_flat)
            
            # Concatenate all layers
            hidden_t_all = torch.cat(hidden_t, dim=-1)  # (batch, d_inner * d_state * n_layers)
            
            # MLP readout
            output_t = self.mlp(hidden_t_all)  # (batch, vocab_size)
            outputs.append(output_t)
        
        # Stack over sequence length
        readout_logits = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        
        return readout_logits


class Samba(nn.Module):
    """
    Samba: Sparse Mamba (130M parameters, HuggingFace compatible)
    
    Combines Mamba with MLP readout to learn sparse representations.
    """
    
    def __init__(
        self, 
        vocab_size=50280, 
        d_model=768, 
        n_layers=24, 
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dt_rank="auto",
        readout_hidden_dim=512
    ):
        super().__init__()
        
        # Base Mamba model
        self.mamba = Mamba(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dt_rank=dt_rank
        )
        
        # MLP readout
        d_inner = d_model * expand_factor
        self.readout = SambaReadout(
            d_inner=d_inner,
            d_state=d_state,
            n_layers=n_layers,
            vocab_size=vocab_size,
            hidden_dim=readout_hidden_dim
        )
        
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.d_state = d_state
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        Returns: 
            - main_logits: (batch, seq_len, vocab_size) from original Mamba
            - readout_logits: (batch, seq_len, vocab_size) from MLP readout
            - all_hidden_states: list of hidden states for sparsity analysis
        """
        # Forward through Mamba
        main_logits, all_hidden_states = self.mamba(input_ids)
        
        # Readout from hidden states
        readout_logits = self.readout(all_hidden_states)
        
        return main_logits, readout_logits, all_hidden_states
    
    def get_sparsity_stats(self, all_hidden_states, threshold=1e-3):
        """
        Calculate sparsity statistics of hidden states
        
        Args:
            all_hidden_states: list of (batch, seq_len, d_inner, d_state)
            threshold: values below this are considered zero
            
        Returns:
            dict with sparsity metrics
        """
        stats = {}
        
        for i, hidden_states in enumerate(all_hidden_states):
            # Count near-zero values
            near_zero = (hidden_states.abs() < threshold).float().mean()
            
            # L0 norm (actual zeros)
            l0 = (hidden_states == 0).float().mean()
            
            # L1 norm
            l1 = hidden_states.abs().mean()
            
            stats[f'layer_{i}'] = {
                'near_zero_ratio': near_zero.item(),
                'l0_sparsity': l0.item(),
                'l1_norm': l1.item()
            }
        
        # Average across layers
        stats['avg_near_zero_ratio'] = sum(s['near_zero_ratio'] for s in stats.values() if isinstance(s, dict)) / self.n_layers
        stats['avg_l1_norm'] = sum(s['l1_norm'] for s in stats.values() if isinstance(s, dict)) / self.n_layers
        
        return stats
