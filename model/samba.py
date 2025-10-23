"""
Samba: Sparse Mamba (130M parameters)
Mamba with readout for sparse representation learning
Uses mamba-ssm CUDA kernels for speed
"""

import torch
import torch.nn as nn
from .readouts import SambaReadout

from mamba_ssm import Mamba as MambaCUDA
MAMBA_SSM_AVAILABLE = True
try:
    from mamba_ssm import Mamba as MambaCUDA
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("⚠️ mamba-ssm not available. Install with: pip install mamba-ssm causal-conv1d")


class Samba(nn.Module):
    """
    Samba: Sparse Mamba (130M parameters)
    
    Uses mamba-ssm CUDA kernels with layer chunking for:
    - Speed: CUDA acceleration (100x faster than pure PyTorch)
    - VRAM: Only stores sampled chunk outputs (not all hidden states)
    - Sparsity: L1 regularization on layer outputs
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
        
        # Mamba chunks (each chunk has readout_stride layers)
        if n_layers % readout_stride != 0:
            raise ValueError(f"n_layers ({n_layers}) must be divisible by readout_stride ({readout_stride})")
        
        n_chunks = n_layers // readout_stride
        
        if self.use_cuda:
            # Use mamba-ssm CUDA implementation
            self.chunks = nn.ModuleList([
                MambaCUDA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand_factor,
                    dt_rank=dt_rank if isinstance(dt_rank, int) else "auto",
                    n_layers=readout_stride  # Each chunk has 'stride' layers
                )
                for _ in range(n_chunks)
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
            - sampled_layer_outputs: list of chunk outputs (for loss)
            - sampled_layer_indices: chunk indices
        """
        # Embedding
        x = self.embedding(input_ids)
        
        # Process each chunk and collect outputs
        sampled_layer_outputs = []
        
        for i, chunk in enumerate(self.chunks):
            # Each chunk processes readout_stride layers with CUDA acceleration
            x = chunk(x)  # Returns (batch, seq_len, d_model)
            
            # Store chunk output for readout and pruning loss
            sampled_layer_outputs.append(x)
        
        # Main output
        x = self.norm_f(x)
        main_logits = self.lm_head(x)
        
        # Readout output (uses all chunk outputs)
        readout_logits = self.readout(sampled_layer_outputs)
        
        # Chunk indices (0, 1, 2, ..., n_chunks-1)
        sampled_layer_indices = list(range(len(self.chunks)))
        
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
