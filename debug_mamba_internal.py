"""
Debug: Check mamba-ssm internal state after weight loading
"""

import torch
from transformers import AutoModelForCausalLM
from mamba_ssm import Mamba as MambaCUDA

print("="*80)
print("mamba-ssm Internal State Investigation")
print("="*80)

# 1. Load HF model
print("\n1. Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
hf_layer_0 = hf_model.backbone.layers[0].mixer

print(f"   HF Layer 0 type: {type(hf_layer_0)}")
print(f"   HF Layer 0 class: {hf_layer_0.__class__.__name__}")

# 2. Create mamba-ssm layer
print("\n2. Creating mamba-ssm Mamba layer...")
mamba_layer = MambaCUDA(
    d_model=768,
    d_state=16,
    d_conv=4,
    expand=2,
    dt_rank="auto"
)

print(f"   Mamba layer type: {type(mamba_layer)}")
print(f"   Mamba layer class: {mamba_layer.__class__.__name__}")

# 3. Check internal attributes
print("\n3. Checking internal attributes...")

print("   HF Layer 0 attributes:")
hf_attrs = [attr for attr in dir(hf_layer_0) if not attr.startswith('_')]
for attr in sorted(hf_attrs)[:15]:
    try:
        val = getattr(hf_layer_0, attr)
        if isinstance(val, (int, float, bool, str)):
            print(f"     {attr}: {val}")
    except:
        pass

print("\n   mamba-ssm Layer attributes:")
mamba_attrs = [attr for attr in dir(mamba_layer) if not attr.startswith('_')]
for attr in sorted(mamba_attrs)[:15]:
    try:
        val = getattr(mamba_layer, attr)
        if isinstance(val, (int, float, bool, str)):
            print(f"     {attr}: {val}")
    except:
        pass

# 4. Compare specific attributes
print("\n4. Comparing key attributes...")

# Check if they have the same structure
hf_params = set(dict(hf_layer_0.named_parameters()).keys())
mamba_params = set(dict(mamba_layer.named_parameters()).keys())

print(f"   HF parameters: {sorted(hf_params)}")
print(f"   Mamba parameters: {sorted(mamba_params)}")
print(f"   Difference: {hf_params.symmetric_difference(mamba_params)}")

# 5. Check if HF model is also mamba-ssm
print("\n5. Checking if HF uses mamba-ssm internally...")
print(f"   HF layer type string: {str(type(hf_layer_0))}")
print(f"   Is mamba-ssm Mamba? {'mamba_ssm' in str(type(hf_layer_0))}")

# 6. Load weights and test
print("\n6. Testing weight loading and forward pass...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hf_model = hf_model.to(device)
mamba_layer = mamba_layer.to(device)

# Get HF layer 0 state dict
hf_state = {}
for key, value in hf_model.backbone.layers[0].mixer.state_dict().items():
    hf_state[key] = value

print(f"   HF state dict keys: {list(hf_state.keys())}")

# Load into mamba layer
missing, unexpected = mamba_layer.load_state_dict(hf_state, strict=False)
print(f"   Missing keys: {missing}")
print(f"   Unexpected keys: {unexpected}")

# Test forward
input_ids = torch.randint(0, 50280, (2, 16)).to(device)
x = torch.randn(2, 16, 768).to(device)

with torch.no_grad():
    # HF forward
    hf_out = hf_model.backbone.layers[0].mixer(x)
    print(f"\n   HF output: mean={hf_out.mean():.6f}, std={hf_out.std():.6f}, has_nan={torch.isnan(hf_out).any()}")
    
    # Mamba forward
    mamba_out = mamba_layer(x)
    print(f"   Mamba output: mean={mamba_out.mean():.6f}, std={mamba_out.std():.6f}, has_nan={torch.isnan(mamba_out).any()}")
    
    # Difference
    diff = (hf_out - mamba_out).abs().max()
    print(f"   Max difference: {diff:.6f}")
    
    if diff > 1.0:
        print(f"\n   ❌ OUTPUTS DON'T MATCH!")
        print(f"   This confirms the issue is with mamba-ssm layer behavior after load_state_dict")
    else:
        print(f"\n   ✓ Outputs match!")

print("\n" + "="*80)
