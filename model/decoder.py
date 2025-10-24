"""
Windowed Transformer Decoder - ULTRA OPTIMIZED
Only caches window_size × window_size masks!
"""

import torch
import torch.nn as nn


class WindowedAttn(nn.Module):
    """Windowed causal self-attention - ULTRA OPTIMIZED"""
    
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = self.d_head ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute ONLY window_size × window_size mask
        w = self.window_size
        indices = torch.arange(w)
        mask = indices.unsqueeze(1) <= indices.unsqueeze(0)  # Causal
        self.register_buffer('window_mask', ~mask, persistent=False)
    
    def forward(self, x):
        B, S, D = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if S <= self.window_size:
            # Small: use pre-computed mask
            mask = self.window_mask[:S, :S]
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Large: vectorized mask (still no Python loops!)
            idx = torch.arange(S, device=x.device)
            causal = idx.unsqueeze(1) <= idx.unsqueeze(0)
            window = idx.unsqueeze(1) >= (idx.unsqueeze(0) - self.window_size + 1)
            mask = ~(causal & window)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class WindowedCrossAttn(nn.Module):
    """Windowed cross-attention - ULTRA OPTIMIZED"""
    
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout
        
        # Pre-compute window_size × window_size mask
        w = self.window_size
        q_idx = torch.arange(w).unsqueeze(1)
        k_idx = torch.arange(w).unsqueeze(0)
        causal = k_idx <= q_idx
        window = k_idx >= (q_idx - w + 1)
        self.register_buffer('window_mask', ~(causal & window), persistent=False)
    
    def forward(self, query, key, value):
        B, S_q, D = query.shape
        S_kv = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(B, S_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S_kv, self.n_heads, self.d_head).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if S_q <= self.window_size and S_kv <= self.window_size:
            # Small: use pre-computed mask
            mask = self.window_mask[:S_q, :S_kv]
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Large: vectorized mask
            q_idx = torch.arange(S_q, device=query.device).unsqueeze(1)
            k_idx = torch.arange(S_kv, device=key.device).unsqueeze(0)
            causal = k_idx <= q_idx
            window = k_idx >= (q_idx - self.window_size + 1)
            mask = ~(causal & window)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, S_q, D)
        out = self.out_proj(out)
        
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        self.self_attn = WindowedAttn(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = WindowedCrossAttn(d_model, n_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, memory):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), memory, memory)
        x = x + self.ffn(self.norm3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, window_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, window_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, memory, embedding_layer=None):
        B, S = input_ids.shape
        token_emb = embedding_layer(input_ids) if embedding_layer is not None else self.embedding(input_ids)
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = token_emb + self.pos_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm_f(x)
    
    @staticmethod
    def shift_right(input_ids, pad_token_id=0):
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = pad_token_id
        return shifted
