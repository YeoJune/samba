"""
Samba: Sparse Mamba (130M parameters)
Mamba with MLP readout for sparse representation learning
"""

import torch
import torch.nn as nn
from .mamba import Mamba


class SambaReadout(nn.Module):
    """
    Efficient readout with layer-wise attention
    
    Design principles:
    1. Minimize information bottleneck: Use attention to selectively aggregate
    2. Flexible: Different timesteps can attend to different layers
    3. Lightweight: Small attention mechanism, efficient projection
    """
    
    def __init__(self, d_inner, d_state, n_layers, vocab_size, hidden_dim=512):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_layers = n_layers
        
        # Hidden state dimension per layer
        state_dim = d_inner * d_state
        
        # Lightweight layer-wise attention
        # Query: what info do we need? (learnable per position)
        self.query_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Key: what info does each layer have?
        self.key_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Value: transform layer representation
        self.value_net = nn.Linear(state_dim, hidden_dim)
        
        # Final projection to vocabulary
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.scale = 128 ** -0.5
        
    def forward(self, all_hidden_states):
        """
        all_hidden_states: list of (batch, seq_len, d_inner, d_state) for each layer
        Returns: (batch, seq_len, vocab_size)
        """
        batch, seq_len = all_hidden_states[0].shape[:2]
        
        # Stack and flatten hidden states
        # (n_layers, batch, seq_len, d_inner, d_state) → (n_layers, batch, seq_len, state_dim)
        hidden_flat = []
        for h in all_hidden_states:
            h_flat = h.reshape(batch, seq_len, -1)  # (batch, seq_len, d_inner * d_state)
            hidden_flat.append(h_flat)
        
        hidden_flat = torch.stack(hidden_flat, dim=0)  # (n_layers, batch, seq_len, state_dim)
        
        # For each timestep, compute attention over layers
        outputs = []
        for t in range(seq_len):
            # Get hidden states at timestep t from all layers
            h_t = hidden_flat[:, :, t, :]  # (n_layers, batch, state_dim)
            
            # Compute attention
            # Use mean across layers as query (what we need overall)
            query = self.query_net(h_t.mean(dim=0, keepdim=True))  # (1, batch, 128)
            
            # Each layer provides a key (what it has)
            keys = self.key_net(h_t)  # (n_layers, batch, 128)
            
            # Attention scores: which layers are most relevant?
            scores = torch.einsum('qbd,lbd->lbq', query, keys) * self.scale  # (n_layers, batch, 1)
            attn_weights = torch.softmax(scores, dim=0)  # (n_layers, batch, 1)
            
            # Transform each layer's hidden state
            values = self.value_net(h_t)  # (n_layers, batch, hidden_dim)
            
            # Weighted sum: aggregate based on attention
            aggregated = (attn_weights * values).sum(dim=0)  # (batch, hidden_dim)
            
            # Project to vocabulary
            output_t = self.output_proj(aggregated)  # (batch, vocab_size)
            outputs.append(output_t)
        
        # Stack over sequence length
        readout_logits = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        
        return readout_logits, attn_weights  # Return attention for analysis
    
    def get_attention_stats(self, all_hidden_states):
        """
        Analyze which layers are being attended to
        Useful for understanding what the model learns
        """
        with torch.no_grad():
            batch, seq_len = all_hidden_states[0].shape[:2]
            
            # Get attention weights for middle of sequence
            hidden_flat = []
            for h in all_hidden_states:
                h_flat = h.reshape(batch, seq_len, -1)
                hidden_flat.append(h_flat)
            hidden_flat = torch.stack(hidden_flat, dim=0)
            
            t = seq_len // 2  # Middle timestep
            h_t = hidden_flat[:, :, t, :]
            
            query = self.query_net(h_t.mean(dim=0, keepdim=True))
            keys = self.key_net(h_t)
            scores = torch.einsum('qbd,lbd->lbq', query, keys) * self.scale
            attn_weights = torch.softmax(scores, dim=0)
            
            # Average over batch
            avg_attn = attn_weights.mean(dim=1).squeeze()  # (n_layers,)
            
            return {
                'layer_attention': avg_attn.cpu().numpy(),
                'max_layer': int(avg_attn.argmax()),
                'entropy': -(avg_attn * torch.log(avg_attn + 1e-10)).sum().item()
            }


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
            - layer_attention: (n_layers, batch, 1) attention weights (optional)
        """
        # Forward through Mamba
        main_logits, all_hidden_states = self.mamba(input_ids)
        
        # Readout from hidden states with attention
        readout_output = self.readout(all_hidden_states)
        
        # Handle both old (single output) and new (tuple output) readout
        if isinstance(readout_output, tuple):
            readout_logits, layer_attention = readout_output
            return main_logits, readout_logits, all_hidden_states, layer_attention
        else:
            readout_logits = readout_output
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
