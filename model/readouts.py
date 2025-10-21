"""
Attention-based Readout for Samba
Works with layer outputs (d_model = 768) from mamba-ssm chunks
"""

import torch
import torch.nn as nn


class SambaReadout(nn.Module):
    """
    Attention-based readout (receives pre-sampled layer outputs from chunks)
    
    Compatible with new chunked mamba-ssm architecture:
    - Input: list of layer outputs (d_model = 768)
    - Each chunk output: (batch, seq_len, d_model)
    """
    
    def __init__(self, d_model, vocab_size, hidden_dim=512):
        super().__init__()
        
        self.query_net = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU())
        self.key_net = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU())
        self.value_net = nn.Linear(d_model, hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.scale = 128 ** -0.5
        
    def forward(self, sampled_layer_outputs):
        """
        Vectorized forward (no seq_len loop for efficiency)
        
        Args:
            sampled_layer_outputs: list of sampled layer outputs
                                   [(batch, seq_len, d_model), ...]
                                   
        Returns:
            readout_logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = sampled_layer_outputs[0].shape[:2]
        
        # Stack: (num_layers, batch, seq_len, d_model)
        h_stacked = torch.stack(sampled_layer_outputs, dim=0)
        
        # Permute to (seq_len, num_layers, batch, d_model)
        h = h_stacked.permute(2, 0, 1, 3)
        
        # Query: mean over layers -> (seq_len, 1, batch, 128)
        query_input = h.mean(dim=1)
        query = self.query_net(query_input).unsqueeze(1)
        
        # Keys: (seq_len, num_layers, batch, 128)
        keys = self.key_net(h)
        
        # Attention scores: (seq_len, num_layers, batch, 1)
        scores = torch.einsum('sqbd,slbd->slbq', query, keys) * self.scale
        attn_weights = torch.softmax(scores, dim=1)
        
        # Values: (seq_len, num_layers, batch, hidden_dim)
        values = self.value_net(h)
        
        # Aggregate: (seq_len, batch, hidden_dim)
        aggregated = (attn_weights * values).sum(dim=1).squeeze(-1)
        
        # Permute back to (batch, seq_len, hidden_dim)
        aggregated = aggregated.permute(1, 0, 2)
        
        # Output projection: (batch, seq_len, vocab_size)
        readout_logits = self.output_proj(aggregated)
        
        return readout_logits
