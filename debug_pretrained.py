"""
Debug script to verify pretrained weight loading
"""

import torch
from model.samba import Samba
from utils.pretrained_loader import load_pretrained_samba

print("="*80)
print("Pretrained Weight Loading Verification")
print("="*80)

# 1. Create Samba model
print("\n1. Creating Samba model...")
model = Samba(
    vocab_size=50280,
    d_model=768,
    n_layers=24,
    d_state=16,
    d_conv=4,
    expand_factor=2,
    readout_stride=4
)
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

# 2. Check initial state (before loading)
print("\n2. Before loading pretrained weights:")
print(f"   Embedding weight: mean={model.embedding.weight.mean():.6f}, std={model.embedding.weight.std():.6f}")
print(f"   Layer 0 param sample: mean={list(model.layers[0].parameters())[0].mean():.6f}")
print(f"   norm_f weight: mean={model.norm_f.weight.mean():.6f}, std={model.norm_f.weight.std():.6f}")
print(f"   Readout param sample: mean={list(model.readout.parameters())[0].mean():.6f}")
print(f"   Weight tying OK? {model.embedding.weight is model.lm_head.weight}")

# 3. Load pretrained weights
print("\n3. Loading pretrained weights...")
model = load_pretrained_samba(model, "state-spaces/mamba-130m-hf", debug=True)

# 4. Check after loading
print("\n4. After loading pretrained weights:")
print(f"   Embedding weight: mean={model.embedding.weight.mean():.6f}, std={model.embedding.weight.std():.6f}")
print(f"   Layer 0 param sample: mean={list(model.layers[0].parameters())[0].mean():.6f}")
print(f"   norm_f weight: mean={model.norm_f.weight.mean():.6f}, std={model.norm_f.weight.std():.6f}")
print(f"   Readout param sample: mean={list(model.readout.parameters())[0].mean():.6f}")
print(f"   Weight tying OK? {model.embedding.weight is model.lm_head.weight}")

# 5. Check for NaN/Inf in loaded weights
print("\n5. Checking for NaN/Inf in loaded weights:")
has_nan = False
has_inf = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"   ⚠️ NaN found in: {name}")
        has_nan = True
    if torch.isinf(param).any():
        print(f"   ⚠️ Inf found in: {name}")
        has_inf = True

if not has_nan and not has_inf:
    print("   ✓ No NaN or Inf in weights")

# 6. Test forward pass with dummy input
print("\n6. Testing forward pass with dummy input...")
input_ids = torch.randint(0, 50280, (2, 16))
print(f"   Input shape: {input_ids.shape}")

try:
    with torch.no_grad():
        main_logits, readout_logits, sampled_outputs, sampled_indices = model(input_ids)
    
    print(f"   ✓ Forward pass successful")
    print(f"\n   Main logits:")
    print(f"     - Shape: {main_logits.shape}")
    print(f"     - Mean: {main_logits.mean():.4f}, Std: {main_logits.std():.4f}")
    print(f"     - Min: {main_logits.min():.4f}, Max: {main_logits.max():.4f}")
    print(f"     - Has NaN? {torch.isnan(main_logits).any()}")
    print(f"     - Has Inf? {torch.isinf(main_logits).any()}")
    
    print(f"\n   Readout logits:")
    print(f"     - Shape: {readout_logits.shape}")
    print(f"     - Mean: {readout_logits.mean():.4f}, Std: {readout_logits.std():.4f}")
    print(f"     - Min: {readout_logits.min():.4f}, Max: {readout_logits.max():.4f}")
    print(f"     - Has NaN? {torch.isnan(readout_logits).any()}")
    print(f"     - Has Inf? {torch.isinf(readout_logits).any()}")
    
    print(f"\n   Sampled layer outputs ({len(sampled_outputs)} layers):")
    for i, output in enumerate(sampled_outputs):
        print(f"     Layer {sampled_indices[i]}: mean={output.mean():.4f}, std={output.std():.4f}, "
              f"min={output.min():.4f}, max={output.max():.4f}, "
              f"nan={torch.isnan(output).any()}, inf={torch.isinf(output).any()}")
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# 7. Test loss computation
print("\n7. Testing loss computation...")
targets = torch.randint(0, 50280, (2, 16))

try:
    with torch.no_grad():
        main_logits, readout_logits, sampled_outputs, sampled_indices = model(input_ids)
        
        # Main loss
        main_loss = torch.nn.functional.cross_entropy(
            main_logits.reshape(-1, 50280),
            targets.reshape(-1)
        )
        print(f"   Main loss: {main_loss.item():.4f} (NaN? {torch.isnan(main_loss)})")
        
        # Readout loss
        readout_loss = torch.nn.functional.cross_entropy(
            readout_logits.reshape(-1, 50280),
            targets.reshape(-1)
        )
        print(f"   Readout loss: {readout_loss.item():.4f} (NaN? {torch.isnan(readout_loss)})")
        
        # Pruning loss
        pruning_loss = sum(h.abs().mean() for h in sampled_outputs) / len(sampled_outputs)
        print(f"   Pruning loss: {pruning_loss.item():.4f} (NaN? {torch.isnan(pruning_loss)})")
        
        # Combined
        total_loss = main_loss + 0.5 * readout_loss + 0.05 * pruning_loss
        print(f"   Total loss: {total_loss.item():.4f} (NaN? {torch.isnan(total_loss)})")
        
except Exception as e:
    print(f"   ❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()

# 8. Compare with HuggingFace model
print("\n8. Comparing with original HuggingFace model...")
try:
    from transformers import AutoModelForCausalLM
    
    hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    
    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits
    
    print(f"   HF logits:")
    print(f"     - Shape: {hf_logits.shape}")
    print(f"     - Mean: {hf_logits.mean():.4f}, Std: {hf_logits.std():.4f}")
    print(f"     - Min: {hf_logits.min():.4f}, Max: {hf_logits.max():.4f}")
    
    # Compare embedding outputs
    hf_emb = hf_model.backbone.embeddings(input_ids)
    samba_emb = model.embedding(input_ids)
    emb_diff = (hf_emb - samba_emb).abs().max()
    print(f"\n   Embedding output max diff: {emb_diff:.6f}")
    
    # Compare final logits
    logit_diff = (hf_logits - main_logits).abs().max()
    print(f"   Final logits max diff: {logit_diff:.6f}")
    
    if logit_diff > 1e-3:
        print(f"   ⚠️ WARNING: Large difference in logits! Weight loading may be incorrect.")
    else:
        print(f"   ✓ Logits match well!")
    
except Exception as e:
    print(f"   ⚠️ Could not compare with HF model: {e}")

print("\n" + "="*80)
print("Verification complete!")
print("="*80)
