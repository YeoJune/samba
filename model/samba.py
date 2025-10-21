"""
Samba: Sparse Mamba (130M parameters)
Mamba with MLP readout for sparse representation learning
"""

import torch
import torch.nn as nn
from .mamba import Mamba


class SambaReadout(nn.Module):
    """Attention-based readout with stride sampling"""
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size, hidden_dim=512, stride=4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_layers = n_layers
        self.stride = stride
        
        state_dim = d_inner * d_state
        
        self.query_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.key_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value_net = nn.Linear(state_dim, hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.scale = 128 ** -0.5
        
    def forward(self, all_hidden_states):
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Stack and flatten
        hidden_flat = torch.stack([h.reshape(batch, seq_len, -1) for h in all_hidden_states], dim=0)
        
        # Sample timesteps with stride
        sampled_indices = list(range(0, seq_len, self.stride))
        
        outputs = []
        for t in sampled_indices:
            h_t = hidden_flat[:, :, t, :]
            
            query = self.query_net(h_t.mean(dim=0, keepdim=True))
            keys = self.key_net(h_t)
            scores = torch.einsum('qbd,lbd->lbq', query, keys) * self.scale
            attn_weights = torch.softmax(scores, dim=0)
            
            values = self.value_net(h_t)
            aggregated = (attn_weights * values).sum(dim=0)
            output_t = self.output_proj(aggregated)
            outputs.append(output_t)
        
        readout_sampled = torch.stack(outputs, dim=1)
        
        # Interpolate to full length
        if self.stride > 1:
            readout_logits = torch.nn.functional.interpolate(
                readout_sampled.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            readout_logits = readout_sampled
        
        return readout_logits, sampled_indices


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
        readout_hidden_dim=512,
        readout_stride=4  # Sample every N timesteps for readout
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
        
        # MLP readout with stride
        d_inner = d_model * expand_factor
        self.readout = SambaReadout(
            d_inner=d_inner,
            d_state=d_state,
            n_layers=n_layers,
            vocab_size=vocab_size,
            hidden_dim=readout_hidden_dim,
            stride=readout_stride
        )
        
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.d_state = d_state
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        Returns: 
            - main_logits: (batch, seq_len, vocab_size)
            - readout_logits: (batch, seq_len, vocab_size)
            - all_hidden_states: list of hidden states
            - sampled_indices: timesteps used for readout (for L1 loss)
        """
        main_logits, all_hidden_states = self.mamba(input_ids)
        readout_logits, sampled_indices = self.readout(all_hidden_states)
        
        return main_logits, readout_logits, all_hidden_states, sampled_indices
    
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
