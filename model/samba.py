"""
Samba: Sparse Mamba (130M parameters)
Mamba with MLP readout for sparse representation learning
"""

import torch
import torch.nn as nn
from .mamba import Mamba


class SambaReadout(nn.Module):
    """Attention-based readout (receives pre-sampled hidden states)"""
    
    def __init__(self, d_inner, d_state, vocab_size, hidden_dim=512):
        super().__init__()
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
        
    def forward(self, sampled_hidden_states):
        """
        Args:
            sampled_hidden_states: list of already sampled layer hidden states
        """
        batch, seq_len = sampled_hidden_states[0].shape[:2]
        
        # Stack and flatten sampled layers
        hidden_flat = torch.stack([h.reshape(batch, seq_len, -1) for h in sampled_hidden_states], dim=0)
        
        outputs = []
        for t in range(seq_len):
            h_t = hidden_flat[:, :, t, :]
            
            query = self.query_net(h_t.mean(dim=0, keepdim=True))
            keys = self.key_net(h_t)
            scores = torch.einsum('qbd,lbd->lbq', query, keys) * self.scale
            attn_weights = torch.softmax(scores, dim=0)
            
            values = self.value_net(h_t)
            aggregated = (attn_weights * values).sum(dim=0)
            output_t = self.output_proj(aggregated)
            outputs.append(output_t)
        
        readout_logits = torch.stack(outputs, dim=1)
        
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
        
        # Readout
        d_inner = d_model * expand_factor
        self.readout = SambaReadout(
            d_inner=d_inner,
            d_state=d_state,
            vocab_size=vocab_size,
            hidden_dim=readout_hidden_dim
        )
        
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.d_state = d_state
        self.readout_stride = readout_stride
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        Returns: 
            - main_logits: (batch, seq_len, vocab_size)
            - readout_logits: (batch, seq_len, vocab_size)
            - sampled_hidden_states: list of sampled layer hidden states (for loss)
            - sampled_layer_indices: layer indices used for readout and loss
        """
        # Embedding
        x = self.mamba.embedding(input_ids)
        
        # Layer-wise forward with stride sampling (VRAM optimization)
        sampled_hidden_states = []
        sampled_layer_indices = list(range(0, self.n_layers, self.readout_stride))
        
        for i, layer in enumerate(self.mamba.layers):
            x, hidden_states = layer(x)
            
            # Only store hidden states for sampled layers
            if i in sampled_layer_indices:
                sampled_hidden_states.append(hidden_states)
        
        # Main output
        x = self.mamba.norm_f(x)
        main_logits = self.mamba.lm_head(x)
        
        # Readout output (uses only sampled hidden states)
        readout_logits = self.readout(sampled_hidden_states)
        
        return main_logits, readout_logits, sampled_hidden_states, sampled_layer_indices
    
    def get_sparsity_stats(self, sampled_hidden_states, threshold=1e-3):
        """
        Calculate sparsity statistics from sampled hidden states
        
        Args:
            sampled_hidden_states: list of sampled layer hidden states
            threshold: values below this are considered zero
            
        Returns:
            dict with sparsity metrics
        """
        total_near_zero = 0.0
        total_l0 = 0.0
        total_l1 = 0.0
        num_sampled = len(sampled_hidden_states)
        
        for hidden_states in sampled_hidden_states:
            total_near_zero += (hidden_states.abs() < threshold).float().mean()
            total_l0 += (hidden_states == 0).float().mean()
            total_l1 += hidden_states.abs().mean()
        
        return {
            'avg_near_zero_ratio': (total_near_zero / num_sampled).item(),
            'avg_l0_sparsity': (total_l0 / num_sampled).item(),
            'avg_l1_norm': (total_l1 / num_sampled).item()
        }
