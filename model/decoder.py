"""
Windowed Transformer Decoder (GPT-2 based)
Used for auxiliary prediction task in 3-loss hybrid architecture
"""

import torch
import torch.nn as nn
import math


class WindowedAttn(nn.Module):
    """Windowed causal self-attention (GPT-2 style)"""
    
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = self.d_head ** -0.5
        
        # Q, K, V projections (GPT-2 style: combined then split)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (windowed)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(window_size, window_size)).view(1, 1, window_size, window_size)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, S, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # Each (B, S, D)
        
        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, S, D_h)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        
        # Create causal + windowed mask
        # Each token can only attend to previous tokens within window_size
        if S <= self.window_size:
            # Use pre-registered mask for small sequences
            mask = self.causal_mask[:, :, :S, :S]
        else:
            # Create windowed causal mask on-the-fly
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=x.dtype))
            
            # Apply window constraint: each position can only see last window_size tokens
            for i in range(S):
                if i >= self.window_size:
                    mask[i, :i - self.window_size] = 0
            
            mask = mask.view(1, 1, S, S)
        
        # Apply mask
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, S, D_h)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class DecoderLayer(nn.Module):
    """Single decoder layer: windowed self-attn + cross-attn + FFN"""
    
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        
        # Windowed self-attention
        self.self_attn = WindowedAttn(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (to Mamba memory)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN (GPT-2 style: 4x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, memory):
        """
        x: (batch, seq_len, d_model) - decoder input
        memory: (batch, seq_len, d_model) - Mamba hidden states
        """
        # Windowed self-attention (GPT-2 style: pre-norm)
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention to Mamba memory
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(x_norm, memory, memory)
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x


class Decoder(nn.Module):
    """
    Windowed Transformer Decoder (GPT-2 based)
    - Loads pretrained GPT-2 weights
    - Windowed self-attention (k=32)
    - Cross-attention to Mamba memory
    """
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, window_size, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding (shared with Mamba)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max seq len 1024
        self.dropout = nn.Dropout(dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, window_size, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm (GPT-2 style)
        self.norm_f = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, memory, embedding_layer=None):
        """
        input_ids: (batch, seq_len) - shifted targets
        memory: (batch, seq_len, d_model) - Mamba hidden states
        embedding_layer: Optional external embedding layer (for weight sharing)
        Returns: (batch, seq_len, d_model)
        """
        B, S = input_ids.shape
        
        # Embeddings (use external if provided, else use own)
        if embedding_layer is not None:
            token_emb = embedding_layer(input_ids)
        else:
            token_emb = self.embedding(input_ids)
        
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)  # (1, S)
        x = token_emb + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Decoder layers
        for layer in self.layers:
            x = layer(x, memory)
        
        # Final norm
        x = self.norm_f(x)
        
        return x
    
    @staticmethod
    def shift_right(input_ids, pad_token_id=0):
        """
        Shift input ids right for autoregressive decoding
        input_ids: (batch, seq_len)
        Returns: (batch, seq_len)
        """
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = pad_token_id
        return shifted
