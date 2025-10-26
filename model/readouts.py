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
                 decoder_window_size=32, dropout=0.1,
                 readout_mode="post"):  # NEW
        super().__init__()
        self.readout_mode = readout_mode  # NEW

        # Adjust n_layers for pre-residual mode (includes initial embedding)
        actual_n_layers = n_layers + 1 if readout_mode == "pre" else n_layers

        # 1. LSM-style learnable weights for linear mixing
        self.layer_weights = nn.Parameter(torch.ones(actual_n_layers))
        
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
        
        IMPORTANT: When decoder generates token at position t, it should only see
        memory up to position t-1 (not position t!). This prevents data leakage.
        """
        # Step 1: LSM-style linear mixing
        # Stack: (24, B, S, D)
        y_stacked = torch.stack(all_layer_outputs, dim=0)
        
        # Softmax weights: (24,)
        weights = F.softmax(self.layer_weights, dim=0)
        
        # Weighted sum: (B, S, D)
        memory = torch.einsum('l,lbsd->bsd', weights, y_stacked)
        
        # Step 2: Shift memory right by 1
        # Decoder position t should only see memory[0:t], not memory[t]
        # memory[t] contains info from input[t], which is used to predict target[t]
        # So we shift: decoder position t sees memory[t-1]
        B, S, D = memory.shape
        memory_shifted = torch.zeros_like(memory)
        memory_shifted[:, 1:, :] = memory[:, :-1, :].clone()
        # memory_shifted[:, 0, :] remains zeros (no previous context)
        
        # Step 3: Shift targets for autoregressive input
        decoder_input = self.decoder.shift_right(targets)
        
        # Step 4: Decoder (windowed self-attn + cross-attn to shifted memory)
        # Use parent embedding for weight sharing
        decoder_output = self.decoder(decoder_input, memory_shifted, embedding_layer=self.parent_embedding)
        
        # Step 5: Auxiliary prediction
        aux_logits = self.aux_head(decoder_output)
        
        return aux_logits
