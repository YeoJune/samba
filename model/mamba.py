"""
Minimal Mamba Implementation
Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class S6(nn.Module):
    """Simplified S6 (Selective State Space) layer"""
    
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, 
            self.d_inner, 
            kernel_size=3, 
            padding=1, 
            groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize A (diagonal state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model), hidden_states
        """
        batch, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        x = F.silu(x)
        
        # SSM
        y, hidden_states = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output, hidden_states
    
    def ssm(self, x):
        """
        Selective State Space Model
        Returns output and hidden states for sparsity analysis
        """
        batch, seq_len, d_inner = x.shape
        
        # Compute ∆, B, C
        x_proj_out = self.x_proj(x)  # (batch, seq_len, d_state * 2)
        B, C = x_proj_out.chunk(2, dim=-1)  # Each (batch, seq_len, d_state)
        
        delta = F.softplus(self.dt_proj(x))  # (batch, seq_len, d_inner)
        
        # Discretize (simplified)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Prepare for scan
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)
        
        # State space scan
        x_expanded = x.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        hidden_states_list = []
        outputs = []
        
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x_expanded[:, t]
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t])
            
            hidden_states_list.append(h.clone())
            outputs.append(y_t)
        
        hidden_states = torch.stack(hidden_states_list, dim=1)  # (batch, seq_len, d_inner, d_state)
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y, hidden_states


class MambaBlock(nn.Module):
    """Single Mamba block with normalization"""
    
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = S6(d_model, d_state, expand_factor)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: output, hidden_states
        """
        residual = x
        x = self.norm(x)
        x, hidden_states = self.mamba(x)
        return x + residual, hidden_states


class Mamba(nn.Module):
    """Full Mamba model"""
    
    def __init__(
        self, 
        vocab_size, 
        d_model=256, 
        n_layers=4, 
        d_state=16,
        expand_factor=2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor) 
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        Returns: logits, all_hidden_states
        """
        x = self.embedding(input_ids)
        
        all_hidden_states = []
        for layer in self.layers:
            x, hidden_states = layer(x)
            all_hidden_states.append(hidden_states)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits, all_hidden_states
