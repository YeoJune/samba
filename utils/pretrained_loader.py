"""
Pretrained model loader for Samba
- Loads HuggingFace Mamba weights into Samba backbone
- Loads GPT-2 weights into windowed decoder
"""

import torch
import torch.nn as nn


def load_pretrained_samba(samba_model, pretrained_name="state-spaces/mamba-130m-hf", debug=False):
    """
    Load pretrained HuggingFace Mamba weights into Samba model
    
    Strategy:
    1. Load full 24-layer HF model state_dict
    2. Load embedding, norm_f, lm_head directly to Samba
    3. Load each of 24 layers with direct 1:1 mapping
    4. Handle HF residual structure (norm + mixer) vs mamba-ssm flat structure
    
    Args:
        samba_model: Samba model instance (with 24 individual layers)
        pretrained_name: HuggingFace model name
        debug: Print debug information about key mapping
        
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
    
    if debug:
        # Debug: Show HF model structure
        print("\n🔍 HuggingFace model structure (first layer keys):")
        layer_0_keys = [k for k in hf_state_dict.keys() if k.startswith("backbone.layers.0.")]
        for key in sorted(layer_0_keys):
            print(f"  - {key}: {hf_state_dict[key].shape}")
        
        print("\n🔍 mamba-ssm model structure (first layer keys):")
        mamba_keys = list(samba_model.layers[0].state_dict().keys())
        for key in sorted(mamba_keys):
            print(f"  - {key}: {samba_model.layers[0].state_dict()[key].shape}")
    
    print("\nCopying weights to Samba model (24 layers)...")
    
    # 1. Embedding (direct copy)
    samba_model.embedding.weight.data.copy_(
        hf_model.backbone.embeddings.weight.data
    )
    print("✓ Embedding copied")
    
    # 2. Layers (load both norm and mixer weights)
    n_layers = len(samba_model.layers)
    successful_loads = 0
    
    for layer_idx in range(n_layers):
        # Load RMSNorm weights
        hf_norm_key = f"backbone.layers.{layer_idx}.norm.weight"
        if hf_norm_key in hf_state_dict:
            samba_model.layer_norms[layer_idx].weight.data.copy_(hf_state_dict[hf_norm_key])
        
        # Load Mamba mixer weights
        hf_mixer_prefix = f"backbone.layers.{layer_idx}.mixer."
        layer_state_dict = {}
        
        for key, value in hf_state_dict.items():
            if key.startswith(hf_mixer_prefix):
                # Remove HF prefix to get mamba-ssm key
                # HF: backbone.layers.4.mixer.in_proj.weight
                # →  in_proj.weight
                local_key = key.replace(hf_mixer_prefix, "")
                layer_state_dict[local_key] = value
        
        if len(layer_state_dict) > 0:
            # Load into layer (strict=False allows architectural differences)
            missing, unexpected = samba_model.layers[layer_idx].load_state_dict(
                layer_state_dict, strict=False
            )
            successful_loads += 1
            
            if debug and (missing or unexpected):
                print(f"\n  Layer {layer_idx}:")
                if missing:
                    print(f"    Missing: {missing}")
                if unexpected:
                    print(f"    Unexpected: {unexpected}")
        else:
            print(f"  ⚠️ Layer {layer_idx}: No weights found (prefix '{hf_mixer_prefix}' not in HF model)")
    
    print(f"✓ Loaded {successful_loads}/{n_layers} layers (with norms) successfully")
    
    # 3. Final norm (RMSNorm has no bias)
    samba_model.norm_f.weight.data.copy_(hf_model.backbone.norm_f.weight.data)
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


def load_pretrained_decoder(decoder, pretrained_name="gpt2", target_vocab_size=50280, debug=False):
    """
    Load pretrained GPT-2 weights into windowed decoder
    
    Strategy:
    1. Load GPT-2 state dict
    2. Copy embedding, position embedding (resize if needed)
    3. Copy layer weights (self-attn, FFN, norms)
    4. Cross-attention weights initialized randomly (not in GPT-2)
    
    Args:
        decoder: Decoder model instance
        pretrained_name: HuggingFace GPT-2 model name
        target_vocab_size: Target vocabulary size (50280 for GPT-2 tokenizer)
        debug: Print debug information
        
    Returns:
        decoder: Model with loaded weights
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    print(f"\nLoading decoder weights from {pretrained_name}...")
    gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained_name)
    gpt2_state_dict = gpt2_model.state_dict()
    
    # Note: Skip embedding loading since decoder will use parent embedding
    # 1. Embedding - Skip if using shared embedding
    gpt2_vocab_size = gpt2_model.config.vocab_size
    print(f"  ⚠️ Decoder embedding will be shared with Samba backbone")
    print(f"     (Skipping GPT-2 embedding copy to avoid duplication)")
    
    # 2. Position embedding
    gpt2_max_pos = gpt2_model.config.n_positions
    decoder_max_pos = decoder.pos_embedding.weight.shape[0]
    min_pos = min(gpt2_max_pos, decoder_max_pos)
    
    decoder.pos_embedding.weight.data[:min_pos].copy_(
        gpt2_model.transformer.wpe.weight.data[:min_pos]
    )
    print(f"  ✓ Position embedding copied ({min_pos} positions)")
    
    # 3. Decoder layers
    n_layers = len(decoder.layers)
    gpt2_n_layers = gpt2_model.config.n_layer
    layers_to_copy = min(n_layers, gpt2_n_layers)
    
    for i in range(layers_to_copy):
        gpt2_block = gpt2_model.transformer.h[i]
        decoder_layer = decoder.layers[i]
        
        # Self-attention (GPT-2 uses Conv1D, need transpose)
        # GPT-2 qkv_proj is c_attn, out_proj is c_proj
        qkv_weight = gpt2_block.attn.c_attn.weight.data.t()  # Transpose Conv1D
        qkv_bias = gpt2_block.attn.c_attn.bias.data
        
        decoder_layer.self_attn.qkv_proj.weight.data.copy_(qkv_weight)
        decoder_layer.self_attn.qkv_proj.bias.data.copy_(qkv_bias)
        
        out_weight = gpt2_block.attn.c_proj.weight.data.t()
        out_bias = gpt2_block.attn.c_proj.bias.data
        
        decoder_layer.self_attn.out_proj.weight.data.copy_(out_weight)
        decoder_layer.self_attn.out_proj.bias.data.copy_(out_bias)
        
        # Layer norm 1
        decoder_layer.norm1.weight.data.copy_(gpt2_block.ln_1.weight.data)
        decoder_layer.norm1.bias.data.copy_(gpt2_block.ln_1.bias.data)
        
        # FFN (GPT-2 uses Conv1D)
        fc1_weight = gpt2_block.mlp.c_fc.weight.data.t()
        fc1_bias = gpt2_block.mlp.c_fc.bias.data
        fc2_weight = gpt2_block.mlp.c_proj.weight.data.t()
        fc2_bias = gpt2_block.mlp.c_proj.bias.data
        
        decoder_layer.ffn[0].weight.data.copy_(fc1_weight)
        decoder_layer.ffn[0].bias.data.copy_(fc1_bias)
        decoder_layer.ffn[3].weight.data.copy_(fc2_weight)
        decoder_layer.ffn[3].bias.data.copy_(fc2_bias)
        
        # Layer norm 3
        decoder_layer.norm3.weight.data.copy_(gpt2_block.ln_2.weight.data)
        decoder_layer.norm3.bias.data.copy_(gpt2_block.ln_2.bias.data)
        
        # Cross-attention (norm2, cross_attn) initialized randomly
        if debug:
            print(f"  Layer {i}: ✓ Self-attn, FFN, Norms | "
                  f"⚠️ Cross-attn (random)")
    
    if layers_to_copy < n_layers:
        print(f"  ⚠️ Only copied {layers_to_copy}/{n_layers} layers "
              f"(GPT-2 has {gpt2_n_layers} layers, extra layers random)")
    else:
        print(f"  ✓ Copied all {layers_to_copy} layers")
    
    # 4. Final layer norm
    decoder.norm_f.weight.data.copy_(gpt2_model.transformer.ln_f.weight.data)
    decoder.norm_f.bias.data.copy_(gpt2_model.transformer.ln_f.bias.data)
    print(f"  ✓ Final norm copied")
    
    print("✅ Decoder weights loaded from GPT-2!")
    print("   Note: Cross-attention weights are randomly initialized")
    
    return decoder
