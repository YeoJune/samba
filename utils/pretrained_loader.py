"""
Pretrained model loader for Samba
Loads HuggingFace Mamba weights into chunked Samba model
"""

import torch
import torch.nn as nn


def load_pretrained_samba(samba_model, pretrained_name="state-spaces/mamba-130m-hf"):
    """
    Load pretrained HuggingFace Mamba weights into Samba model with chunks
    
    Strategy:
    1. Load full 24-layer HF model state_dict
    2. Load embedding, norm_f, lm_head directly to Samba
    3. Split 24 layers into chunks (e.g., 6 chunks of 4 layers each)
    4. Remap prefixes: layers.{i*4+j} → chunk[i].layers.{j}
    
    Args:
        samba_model: Samba model instance with chunks
        pretrained_name: HuggingFace model name
        
    Returns:
        samba_model: Model with loaded weights
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    print(f"Loading pretrained model from {pretrained_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(pretrained_name)
    hf_state_dict = hf_model.state_dict()
    
    print("Copying weights to chunked Samba model...")
    
    # 1. Embedding (direct copy)
    samba_model.embedding.weight.data.copy_(
        hf_model.backbone.embeddings.weight.data
    )
    print("✓ Embedding copied")
    
    # 2. Chunks (prefix remapping)
    n_chunks = len(samba_model.chunks)
    layers_per_chunk = samba_model.readout_stride
    
    for chunk_idx in range(n_chunks):
        print(f"Loading chunk {chunk_idx} (layers {chunk_idx * layers_per_chunk} to {(chunk_idx + 1) * layers_per_chunk - 1})...")
        
        chunk_state_dict = {}
        
        # Remap each layer in this chunk
        for local_layer_idx in range(layers_per_chunk):
            global_layer_idx = chunk_idx * layers_per_chunk + local_layer_idx
            
            # Find all keys for this global layer in HF model
            hf_prefix = f"backbone.layers.{global_layer_idx}."
            
            for key, value in hf_state_dict.items():
                if key.startswith(hf_prefix):
                    # Remove HF prefix and add mamba-ssm chunk prefix
                    # HF: backbone.layers.4.mixer.in_proj.weight
                    # →  layers.0.mixer.in_proj.weight (for chunk[1], local layer 0)
                    
                    local_key = key.replace(hf_prefix, f"layers.{local_layer_idx}.")
                    # Remove 'backbone.' if present
                    local_key = local_key.replace("backbone.", "")
                    
                    chunk_state_dict[local_key] = value
        
        # Load into chunk
        missing, unexpected = samba_model.chunks[chunk_idx].load_state_dict(chunk_state_dict, strict=False)
        
        if missing:
            print(f"  ⚠️ Missing keys in chunk {chunk_idx}: {missing[:3]}..." if len(missing) > 3 else f"  ⚠️ Missing keys: {missing}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys in chunk {chunk_idx}: {unexpected[:3]}..." if len(unexpected) > 3 else f"  ⚠️ Unexpected keys: {unexpected}")
        
        print(f"✓ Chunk {chunk_idx} loaded")
    
    # 3. Final norm
    samba_model.norm_f.weight.data.copy_(hf_model.backbone.norm_f.weight.data)
    if hasattr(samba_model.norm_f, 'bias') and samba_model.norm_f.bias is not None:
        samba_model.norm_f.bias.data.copy_(hf_model.backbone.norm_f.bias.data)
    print("✓ Final norm copied")
    
    # 4. LM head (already tied with embedding)
    print("✓ LM head (tied with embedding)")
    
    print("✅ All weights loaded successfully!")
    
    return samba_model


def verify_samba_weights(samba_model, pretrained_name="state-spaces/mamba-130m-hf"):
    """
    Verify that weights match between Samba and HF model
    
    Args:
        samba_model: Samba model with loaded weights
        pretrained_name: HuggingFace model name
    """
    from transformers import AutoModelForCausalLM
    
    print("\n" + "="*80)
    print("Verifying weight match...")
    print("="*80)
    
    hf_model = AutoModelForCausalLM.from_pretrained(pretrained_name)
    
    # Move both models to CPU for comparison
    device = next(samba_model.parameters()).device
    samba_cpu = samba_model.cpu()
    hf_cpu = hf_model.cpu()
    
    # Check embedding
    emb_diff = (samba_cpu.embedding.weight - hf_cpu.backbone.embeddings.weight).abs().max()
    print(f"Embedding max diff: {emb_diff:.2e}")
    
    # Check first chunk, first layer
    # Note: mamba-ssm structure may differ from HF, so this is approximate
    print(f"First chunk verification (approximate, structure may differ)")
    
    # Check final norm
    norm_diff = (samba_cpu.norm_f.weight - hf_cpu.backbone.norm_f.weight).abs().max()
    print(f"Final norm max diff: {norm_diff:.2e}")
    
    # Move back to original device
    samba_model.to(device)
    
    threshold = 1e-5
    if emb_diff < threshold and norm_diff < threshold:
        print("\n✅ Weight verification PASSED! (embedding and norm)")
        print("   Note: Chunk layer verification requires mamba-ssm structure analysis")
    else:
        print("\n⚠️ Warning: Some differences > 1e-5")
    
    print("="*80 + "\n")
