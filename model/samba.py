"""
Samba: 3-Loss Hybrid Architecture
Mamba backbone + LSM mixing + Windowed decoder
- Stores all 24 layer outputs (not sampled)
- L1 loss on all outputs for sparsity
- Auxiliary decoder loss for semantic meaning
"""

import torch
import torch.nn as nn
from .readouts import Readout

try:
    from mamba_ssm import Mamba as MambaCUDA
    # Use mamba-ssm CUDA implementation
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("⚠️ mamba-ssm not available. Install with: pip install mamba-ssm causal-conv1d")


class Samba(nn.Module):
    """
    Samba: 3-Loss Hybrid Architecture (130M backbone + decoder)
    
    Architecture: 24 Mamba blocks + LSM mixing + Windowed decoder
    - Stores ALL 24 layer outputs (not sampled)
    - L1 loss on all outputs for sparsity
    - Auxiliary decoder loss for semantic meaning
    - Uses mamba-ssm CUDA kernels for speed
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
        decoder_n_layers=6,
        decoder_n_heads=12,
        decoder_window_size=32,
        decoder_dropout=0.1,
        use_cuda=True,
        readout_mode="post",
        pad_token_id=None
    ):
        super().__init__()
        
        if use_cuda and not MAMBA_SSM_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm causal-conv1d")
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.use_cuda = use_cuda and MAMBA_SSM_AVAILABLE
        self.readout_mode = readout_mode
        self.pad_token_id = pad_token_id if pad_token_id is not None else 50256
        
        # Embedding & Output (owned by Samba)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm_f = RMSNorm(d_model)  # Use RMSNorm to match HF Mamba
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Create n_layers individual Mamba blocks with RMSNorm (matches HF structure!)
        # HF structure: norm -> mixer -> residual
        if self.use_cuda:
            self.layers = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            
            for _ in range(n_layers):
                # RMSNorm for each layer (like HF)
                self.layer_norms.append(RMSNorm(d_model))
                
                # Mamba layer
                self.layers.append(
                    MambaCUDA(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand_factor,  # ✓ Use expand_factor (NOT readout_stride!)
                        dt_rank=dt_rank if isinstance(dt_rank, int) else "auto",
                    )
                )
        else:
            # Fallback to pure PyTorch (for compatibility)
            from .mamba import Mamba
            raise NotImplementedError("Pure PyTorch fallback not yet implemented for chunked architecture")
        
        # Readout (LSM mixing + Windowed decoder)
        self.readout = Readout(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            decoder_n_layers=decoder_n_layers,
            decoder_n_heads=decoder_n_heads,
            decoder_window_size=decoder_window_size,
            dropout=decoder_dropout,
            readout_mode=readout_mode,
            pad_token_id=pad_token_id
        )
        
        # Share embeddings: Decoder uses same embedding as Samba backbone
        self.readout.parent_embedding = self.embedding
        
        # Share aux_head with lm_head (optional but saves memory)
        self.readout.aux_head.weight = self.lm_head.weight
        
    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) - required for auxiliary loss
        Returns: 
            - main_logits: (batch, seq_len, vocab_size)
            - aux_logits: (batch, seq_len, vocab_size) or None
            - all_layer_outputs: list of all 24 layer outputs
        """
        # Embedding
        x = self.embedding(input_ids)
        # Process all layers and store outputs according to readout_mode
        all_layer_outputs = []

        # Optional: include initial embedding for pre-residual readout
        if self.readout_mode == "pre":
            all_layer_outputs.append(x)

        for i, layer in enumerate(self.layers):
            # Residual block structure (matches HF Mamba)
            residual = x
            x_norm = self.layer_norms[i](x)  # RMSNorm
            x_block = layer(x_norm)          # Mamba layer output (before residual)
            x = residual + x_block           # Residual connection

            # Store outputs based on mode
            if self.readout_mode == "pre":
                # Pre-residual: store block output only (pure transformation)
                all_layer_outputs.append(x_block)
            else:  # "post"
                # Post-residual: store accumulated output (with residual)
                all_layer_outputs.append(x)
        
        # Main output (final layer)
        x = self.norm_f(x)
        main_logits = self.lm_head(x)

        # Auxiliary output (uses all layer outputs)
        if targets is not None:
            aux_logits = self.readout(all_layer_outputs, input_ids, targets)
        else:
            aux_logits = None

        return main_logits, aux_logits, all_layer_outputs
    
    def get_sparsity_stats(self, all_layer_outputs, threshold=1e-3):
        """
        Calculate sparsity statistics from all layer outputs
        
        Args:
            all_layer_outputs: list of all layer outputs
                               [(batch, seq_len, d_model), ...] (24 items)
            threshold: values below this are considered zero
            
        Returns:
            dict with sparsity metrics
        """
        total_near_zero = 0.0
        total_l0 = 0.0
        total_l1 = 0.0
        num_layers = len(all_layer_outputs)
        
        for layer_output in all_layer_outputs:
            total_near_zero += (layer_output.abs() < threshold).float().mean()
            total_l0 += (layer_output == 0).float().mean()
            total_l1 += layer_output.abs().mean()
        
        return {
            'avg_near_zero_ratio': (total_near_zero / num_layers).item(),
            'avg_l0_sparsity': (total_l0 / num_layers).item(),
            'avg_l1_norm': (total_l1 / num_layers).item()
        }
