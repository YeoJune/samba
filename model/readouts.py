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
                 readout_mode="post", pad_token_id=None):
        super().__init__()
        self.readout_mode = readout_mode
        self.pad_token_id = pad_token_id if pad_token_id is not None else 50256

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
    
    def forward(self, all_layer_outputs, targets=None):
        """
        Args:
            all_layer_outputs: list of 24 layer outputs [(B, S, D), ...]
            targets: (B, S) - for training (teacher forcing). None for inference.
        Returns:
            aux_logits: (B, S, vocab_size)
        
        Training mode (targets provided):
            - Uses teacher forcing: decoder_input[t] = targets[t-1]
            - Parallel processing of all positions
        
        Inference mode (targets=None):
            - Uses auto-regressive generation: decoder_input[t] = predicted[t-1]
            - Sequential generation position by position
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
        
        # Step 3: Prepare decoder input
        if targets is not None:
            # Training mode: Teacher Forcing
            # decoder_input[t] = targets[t-1] to predict targets[t]
            decoder_input = self.decoder.shift_right(targets, pad_token_id=self.pad_token_id)
            
            # Step 4: Decoder forward (parallel)
            decoder_output = self.decoder(decoder_input, memory_shifted, embedding_layer=self.parent_embedding)
            
            # Step 5: Auxiliary prediction
            aux_logits = self.aux_head(decoder_output)
            
        else:
            # Inference mode: Auto-regressive generation
            # Generate position by position, using previous predictions
            decoder_input = torch.full((B, S), self.pad_token_id, dtype=torch.long, device=memory.device)
            aux_logits = torch.zeros(B, S, self.aux_head.out_features, device=memory.device)
            
            for t in range(S):
                # Decoder input up to position t
                current_input = decoder_input[:, :t+1]
                current_memory = memory_shifted[:, :t+1, :]
                
                # Forward through decoder
                decoder_output = self.decoder(current_input, current_memory, embedding_layer=self.parent_embedding)
                
                # Get logits for position t
                logits_t = self.aux_head(decoder_output[:, -1, :])  # (B, vocab_size)
                aux_logits[:, t, :] = logits_t
                
                # Sample/argmax for next position input
                if t < S - 1:
                    next_token = logits_t.argmax(dim=-1)  # Greedy decoding
                    decoder_input[:, t+1] = next_token
        
        return aux_logits
