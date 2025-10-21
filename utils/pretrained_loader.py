"""
Pretrained model loader for Samba
Loads HuggingFace Mamba weights into our model
"""

import torch
import torch.nn as nn


def load_pretrained_mamba(our_model, pretrained_name="state-spaces/mamba-130m-hf"):
    """
    Load pretrained HuggingFace Mamba weights into our model
    
    Args:
        our_model: Our Mamba model instance
        pretrained_name: HuggingFace model name
        
    Returns:
        our_model: Model with loaded weights
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    print(f"Loading pretrained model from {pretrained_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(pretrained_name)
    
    print("Copying weights...")
    
    # 1. Embedding
    our_model.embedding.weight.data.copy_(
        hf_model.backbone.embeddings.weight.data
    )
    print("✓ Embedding copied")
    
    # 2. Layers
    for i, (our_layer, hf_layer) in enumerate(zip(our_model.layers, hf_model.backbone.layers)):
        # Norm
        our_layer.norm.weight.data.copy_(hf_layer.norm.weight.data)
        
        # Mamba/Mixer
        our_mamba = our_layer.mamba
        hf_mixer = hf_layer.mixer
        
        # in_proj
        our_mamba.in_proj.weight.data.copy_(hf_mixer.in_proj.weight.data)
        
        # conv1d
        our_mamba.conv1d.weight.data.copy_(hf_mixer.conv1d.weight.data)
        our_mamba.conv1d.bias.data.copy_(hf_mixer.conv1d.bias.data)
        
        # x_proj
        our_mamba.x_proj.weight.data.copy_(hf_mixer.x_proj.weight.data)
        
        # dt_proj
        our_mamba.dt_proj.weight.data.copy_(hf_mixer.dt_proj.weight.data)
        our_mamba.dt_proj.bias.data.copy_(hf_mixer.dt_proj.bias.data)
        
        # A_log
        our_mamba.A_log.data.copy_(hf_mixer.A_log.data)
        
        # D
        our_mamba.D.data.copy_(hf_mixer.D.data)
        
        # out_proj
        our_mamba.out_proj.weight.data.copy_(hf_mixer.out_proj.weight.data)
        
        if (i + 1) % 5 == 0:
            print(f"✓ Layers {i-4}-{i} copied")
    
    # 3. Final norm
    our_model.norm_f.weight.data.copy_(hf_model.backbone.norm_f.weight.data)
    print("✓ Final norm copied")
    
    # 4. LM head (already tied with embedding)
    print("✓ LM head (tied with embedding)")
    
    print("✅ All weights loaded successfully!")
    
    return our_model


def verify_weight_match(our_model, pretrained_name="state-spaces/mamba-130m-hf"):
    """
    Verify that weights match between our model and HF model
    
    Args:
        our_model: Our Mamba model with loaded weights
        pretrained_name: HuggingFace model name
    """
    from transformers import AutoModelForCausalLM
    
    print("\n" + "="*80)
    print("Verifying weight match...")
    print("="*80)
    
    hf_model = AutoModelForCausalLM.from_pretrained(pretrained_name)
    
    # Check embedding
    emb_diff = (our_model.embedding.weight - hf_model.backbone.embeddings.weight).abs().max()
    print(f"Embedding max diff: {emb_diff:.2e}")
    
    # Check first layer
    our_layer = our_model.layers[0].mamba
    hf_layer = hf_model.backbone.layers[0].mixer
    
    in_proj_diff = (our_layer.in_proj.weight - hf_layer.in_proj.weight).abs().max()
    A_log_diff = (our_layer.A_log - hf_layer.A_log).abs().max()
    
    print(f"Layer 0 in_proj max diff: {in_proj_diff:.2e}")
    print(f"Layer 0 A_log max diff: {A_log_diff:.2e}")
    
    # Check final norm
    norm_diff = (our_model.norm_f.weight - hf_model.backbone.norm_f.weight).abs().max()
    print(f"Final norm max diff: {norm_diff:.2e}")
    
    threshold = 1e-5
    if emb_diff < threshold and in_proj_diff < threshold and A_log_diff < threshold and norm_diff < threshold:
        print("\n✅ Weight verification PASSED! All differences < 1e-5")
    else:
        print("\n⚠️ Warning: Some differences > 1e-5")
    
    print("="*80 + "\n")
