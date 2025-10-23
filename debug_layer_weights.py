"""
Debug: Check if layer weights are actually loaded
"""

import torch
from transformers import AutoModelForCausalLM
from model.samba import Samba
from utils.pretrained_loader import load_pretrained_samba

print("="*80)
print("Layer Weight Loading Verification")
print("="*80)

# Load HF model
print("\n1. Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
hf_state = hf_model.state_dict()

# Create and load Samba model
print("\n2. Creating Samba model...")
samba_model = Samba(
    vocab_size=50280,
    d_model=768,
    n_layers=24,
    d_state=16,
    d_conv=4,
    expand_factor=2,
    readout_stride=4
)

print("\n3. BEFORE loading - Layer 0 weights:")
layer_0_before = samba_model.layers[0].state_dict()
print(f"   in_proj.weight: mean={layer_0_before['in_proj.weight'].mean():.6f}, "
      f"std={layer_0_before['in_proj.weight'].std():.6f}")
print(f"   A_log: mean={layer_0_before['A_log'].mean():.6f}, "
      f"std={layer_0_before['A_log'].std():.6f}")
print(f"   D: mean={layer_0_before['D'].mean():.6f}, "
      f"std={layer_0_before['D'].std():.6f}")

print("\n4. Loading pretrained weights...")
samba_model = load_pretrained_samba(samba_model, "state-spaces/mamba-130m-hf", debug=False)

print("\n5. AFTER loading - Layer 0 weights:")
layer_0_after = samba_model.layers[0].state_dict()
print(f"   in_proj.weight: mean={layer_0_after['in_proj.weight'].mean():.6f}, "
      f"std={layer_0_after['in_proj.weight'].std():.6f}")
print(f"   A_log: mean={layer_0_after['A_log'].mean():.6f}, "
      f"std={layer_0_after['A_log'].std():.6f}")
print(f"   D: mean={layer_0_after['D'].mean():.6f}, "
      f"std={layer_0_after['D'].std():.6f}")

print("\n6. HuggingFace model - Layer 0 weights:")
hf_in_proj = hf_state['backbone.layers.0.mixer.in_proj.weight']
hf_A_log = hf_state['backbone.layers.0.mixer.A_log']
hf_D = hf_state['backbone.layers.0.mixer.D']

print(f"   in_proj.weight: mean={hf_in_proj.mean():.6f}, "
      f"std={hf_in_proj.std():.6f}")
print(f"   A_log: mean={hf_A_log.mean():.6f}, "
      f"std={hf_A_log.std():.6f}")
print(f"   D: mean={hf_D.mean():.6f}, "
      f"std={hf_D.std():.6f}")

print("\n7. Comparing weights (should be near 0):")
diff_in_proj = (hf_in_proj - layer_0_after['in_proj.weight'].cpu()).abs().max()
diff_A_log = (hf_A_log - layer_0_after['A_log'].cpu()).abs().max()
diff_D = (hf_D - layer_0_after['D'].cpu()).abs().max()

print(f"   in_proj.weight max diff: {diff_in_proj:.8f}")
print(f"   A_log max diff: {diff_A_log:.8f}")
print(f"   D max diff: {diff_D:.8f}")

if diff_in_proj > 1e-5 or diff_A_log > 1e-5 or diff_D > 1e-5:
    print("\n   ❌ WEIGHTS NOT LOADED CORRECTLY!")
else:
    print("\n   ✓ Weights loaded correctly")

# 8. Test forward with residual block (proper comparison)
print("\n8. Testing layer 0 with FULL residual block...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# Move to device
hf_model = hf_model.to(device)
samba_model = samba_model.to(device)

# Test input
input_ids = torch.randint(0, 50280, (2, 16)).to(device)

# HF embedding
with torch.no_grad():
    hf_emb = hf_model.backbone.embeddings(input_ids)
    print(f"   HF embedding: mean={hf_emb.mean():.6f}, std={hf_emb.std():.6f}, "
          f"has_nan={torch.isnan(hf_emb).any()}")
    
    # Samba embedding
    samba_emb = samba_model.embedding(input_ids)
    print(f"   Samba embedding: mean={samba_emb.mean():.6f}, std={samba_emb.std():.6f}, "
          f"has_nan={torch.isnan(samba_emb).any()}")
    
    # HF layer 0 (full residual block: norm -> mixer -> residual)
    hf_layer_0_out = hf_model.backbone.layers[0](hf_emb)
    print(f"\n   HF layer 0 FULL block output: mean={hf_layer_0_out.mean():.6f}, std={hf_layer_0_out.std():.6f}, "
          f"has_nan={torch.isnan(hf_layer_0_out).any()}")
    
    # Samba layer 0 (with residual block)
    try:
        # Replicate the residual block structure
        residual = samba_emb
        x = samba_model.layer_norms[0](samba_emb)
        print(f"   After norm: mean={x.mean():.6f}, std={x.std():.6f}")
        
        x = samba_model.layers[0](x)
        print(f"   After mixer: mean={x.mean():.6f}, std={x.std():.6f}, has_nan={torch.isnan(x).any()}")
        
        samba_layer_0_out = residual + x
        print(f"   Samba layer 0 FULL block output: mean={samba_layer_0_out.mean():.6f}, std={samba_layer_0_out.std():.6f}, "
              f"has_nan={torch.isnan(samba_layer_0_out).any()}")
        
        # Compare
        diff = (hf_layer_0_out - samba_layer_0_out).abs().max()
        print(f"\n   Layer 0 output max diff: {diff:.6f}")
        
        if diff > 1.0:
            print(f"   ❌ Outputs don't match!")
            
            # Check norm weights
            hf_norm = hf_model.backbone.layers[0].norm.weight
            samba_norm = samba_model.layer_norms[0].weight
            norm_diff = (hf_norm - samba_norm.cpu()).abs().max()
            print(f"   Norm weight diff: {norm_diff:.6f}")
        else:
            print(f"   ✓ Outputs match!")
        
    except Exception as e:
        print(f"   ❌ Samba layer 0 forward failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Verification complete!")
print("="*80)
