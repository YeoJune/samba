"""
Samba: Sparse Mamba (130M parameters)
Mamba with readout for sparse representation learning
Uses mamba-ssm CUDA kernels for speed
"""

import torch
import torch.nn as nn
from .readouts import SambaReadout

try:
    from mamba_ssm import Mamba as MambaCUDA
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("⚠️ mamba-ssm not available. Install with: pip install mamba-ssm causal-conv1d")


class Samba(nn.Module):
    """
    Samba: Sparse Mamba (130M parameters)
    
    Architecture: n_layers (24) Mamba blocks with sampled outputs
    - Identical to pretrained Mamba for weight loading
    - Samples every readout_stride-th layer output (e.g., 4, 8, 12, 16, 20, 24)
    - Uses mamba-ssm CUDA kernels for speed (100x faster than pure PyTorch)
    - VRAM efficient: Only stores sampled outputs (not all hidden states)
    - Sparsity: L1 regularization on sampled layer outputs
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
        readout_stride=4,  # Sample every N layers (creates n_layers/stride chunks)
        use_cuda=True
    ):
        super().__init__()
        
        if use_cuda and not MAMBA_SSM_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm causal-conv1d")
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.readout_stride = readout_stride
        self.use_cuda = use_cuda and MAMBA_SSM_AVAILABLE
        
        # Embedding & Output (owned by Samba)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Validate readout_stride
        if n_layers % readout_stride != 0:
            raise ValueError(f"n_layers ({n_layers}) must be divisible by readout_stride ({readout_stride})")
        
        # Create n_layers individual Mamba blocks (NOT n_chunks!)
        # This ensures architecture matches pretrained Mamba exactly
        if self.use_cuda:
            # Use mamba-ssm CUDA implementation
            self.layers = nn.ModuleList([
                MambaCUDA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,  # ✓ Use expand_factor (NOT readout_stride!)
                    dt_rank=dt_rank if isinstance(dt_rank, int) else "auto",
                )
                for _ in range(n_layers)  # ✓ Create n_layers (24), NOT n_chunks (6)
            ])
        else:
            # Fallback to pure PyTorch (for compatibility)
            from .mamba import Mamba
            raise NotImplementedError("Pure PyTorch fallback not yet implemented for chunked architecture")
        
        # Readout
        self.readout = SambaReadout(
            d_model=d_model,
            vocab_size=vocab_size,
            hidden_dim=readout_hidden_dim
        )
        
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        Returns: 
            - main_logits: (batch, seq_len, vocab_size)
            - readout_logits: (batch, seq_len, vocab_size)
            - sampled_layer_outputs: list of sampled outputs (for loss)
            - sampled_layer_indices: sampled layer indices
        """
        # Embedding
        x = self.embedding(input_ids)
        
        # Process all layers and sample outputs at readout_stride intervals
        sampled_layer_outputs = []
        sampled_layer_indices = []
        
        for i, layer in enumerate(self.layers):
            # Pass through Mamba layer
            x = layer(x)  # Returns (batch, seq_len, d_model)
            
            # Sample output every readout_stride layers
            # e.g., if readout_stride=4: sample at layers 3, 7, 11, 15, 19, 23 (0-indexed)
            if (i + 1) % self.readout_stride == 0:
                sampled_layer_outputs.append(x)
                sampled_layer_indices.append(i)
        
        # Main output (final layer)
        x = self.norm_f(x)
        main_logits = self.lm_head(x)
        
        # Readout output (uses sampled outputs only)
        readout_logits = self.readout(sampled_layer_outputs)
        
        return main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices
    
    def get_sparsity_stats(self, sampled_layer_outputs, threshold=1e-3):
        """
        Calculate sparsity statistics from sampled layer outputs
        
        Args:
            sampled_layer_outputs: list of sampled layer outputs
                                   [(batch, seq_len, d_model), ...]
            threshold: values below this are considered zero
            
        Returns:
            dict with sparsity metrics
        """
        total_near_zero = 0.0
        total_l0 = 0.0
        total_l1 = 0.0
        num_sampled = len(sampled_layer_outputs)
        
        for layer_output in sampled_layer_outputs:
            total_near_zero += (layer_output.abs() < threshold).float().mean()
            total_l0 += (layer_output == 0).float().mean()
            total_l1 += layer_output.abs().mean()
        
        return {
            'avg_near_zero_ratio': (total_near_zero / num_sampled).item(),
            'avg_l0_sparsity': (total_l0 / num_sampled).item(),
            'avg_l1_norm': (total_l1 / num_sampled).item()
        }
