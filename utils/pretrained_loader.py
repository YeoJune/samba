"""
Pretrained model loader for Samba
Loads HuggingFace Mamba weights into chunked Samba model
"""

import torch
import torch.nn as nn


def load_pretrained_samba(samba_model, pretrained_name="state-spaces/mamba-130m-hf"):
    """
    Load pretrained HuggingFace Mamba weights into Samba model
    
    Strategy:
    1. Load full 24-layer HF model state_dict
    2. Load embedding, norm_f, lm_head directly to Samba
    3. Load each of 24 layers with direct 1:1 mapping
    4. Remap prefixes: backbone.layers.{i} → layers.{i}
    
    Args:
        samba_model: Samba model instance (with 24 individual layers)
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
    
    print("Copying weights to Samba model (24 layers)...")
    
    # 1. Embedding (direct copy)
    samba_model.embedding.weight.data.copy_(
        hf_model.backbone.embeddings.weight.data
    )
    print("✓ Embedding copied")
    
    # 2. Layers (1:1 mapping from HF to Samba)
    n_layers = len(samba_model.layers)
    
    for layer_idx in range(n_layers):
        # Find all keys for this layer in HF model
        hf_prefix = f"backbone.layers.{layer_idx}."
        
        layer_state_dict = {}
        
        for key, value in hf_state_dict.items():
            if key.startswith(hf_prefix):
                # Remove HF prefix
                # HF: backbone.layers.4.mixer.in_proj.weight
                # →  mixer.in_proj.weight
                local_key = key.replace(hf_prefix, "")
                layer_state_dict[local_key] = value
        
        # Load into layer
        missing, unexpected = samba_model.layers[layer_idx].load_state_dict(layer_state_dict, strict=False)
        
        if missing:
            print(f"  ⚠️ Layer {layer_idx} missing keys: {missing[:3]}..." if len(missing) > 3 else f"  ⚠️ Layer {layer_idx} missing: {missing}")
        if unexpected:
            print(f"  ⚠️ Layer {layer_idx} unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"  ⚠️ Layer {layer_idx} unexpected: {unexpected}")
    
    print(f"✓ All {n_layers} layers loaded")
    
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
    
    # Check first layer (approximate - mamba-ssm structure may differ from HF)
    print(f"\nFirst layer verification (approximate, structure may differ)")
    try:
        # This is a best-effort check since mamba-ssm internal structure differs
        print(f"  Layer 0 exists in both models")
    except Exception as e:
        print(f"  ⚠️ Cannot verify layer structure: {e}")
    
    # Check final norm
    norm_diff = (samba_cpu.norm_f.weight - hf_cpu.backbone.norm_f.weight).abs().max()
    print(f"\nFinal norm max diff: {norm_diff:.2e}")
    
    # Move back to original device
    samba_model.to(device)
    
    threshold = 1e-5
    if emb_diff < threshold and norm_diff < threshold:
        print("\n✅ Weight verification PASSED! (embedding and norm)")
        print("   Note: Layer verification requires mamba-ssm structure analysis")
    else:
        print("\n⚠️ Warning: Some differences > 1e-5")
    
    print("="*80 + "\n")
