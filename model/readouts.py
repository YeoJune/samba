"""
Hybrid Readout for 3-Loss Architecture
LSM-style linear mixing + Windowed decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder


class Readout(nn.Module):
    """
    3-Loss Hybrid Readout
    1. LSM-style linear mixing of all 24 layer outputs
    2. Windowed decoder (GPT-2 based) with cross-attention
    3. Auxiliary prediction head
    """
    
    def __init__(self, vocab_size, d_model, n_layers=24, 
                 decoder_n_layers=6, decoder_n_heads=12, 
                 decoder_window_size=32, dropout=0.1):
        super().__init__()
        
        # 1. LSM-style learnable weights for linear mixing
        self.layer_weights = nn.Parameter(torch.ones(n_layers))
        
        # 2. Windowed decoder (GPT-2 based)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=decoder_n_layers,
            n_heads=decoder_n_heads,
            window_size=decoder_window_size,
            dropout=dropout
        )
        
        # 3. Auxiliary prediction head
        self.aux_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Store reference to parent embedding (will be set by Samba)
        self.parent_embedding = None
    
    def forward(self, all_layer_outputs, targets):
        """
        Args:
            all_layer_outputs: list of 24 layer outputs [(B, S, D), ...]
            targets: (B, S) - for shift_right
        Returns:
            aux_logits: (B, S, vocab_size)
        """
        # Step 1: LSM-style linear mixing
        # Stack: (24, B, S, D)
        y_stacked = torch.stack(all_layer_outputs, dim=0)
        
        # Softmax weights: (24,)
        weights = F.softmax(self.layer_weights, dim=0)
        
        # Weighted sum: (B, S, D)
        memory = torch.einsum('l,lbsd->bsd', weights, y_stacked)
        
        # Step 2: Shift targets for autoregressive input
        decoder_input = self.decoder.shift_right(targets)
        
        # Step 3: Decoder (windowed self-attn + cross-attn to memory)
        # Use parent embedding for weight sharing
        decoder_output = self.decoder(decoder_input, memory, embedding_layer=self.parent_embedding)
        
        # Step 4: Auxiliary prediction
        aux_logits = self.aux_head(decoder_output)
        
        return aux_logits
