"""
Final test: Full model forward pass comparison
"""

import torch
from transformers import AutoModelForCausalLM
from model.samba import Samba
from utils.pretrained_loader import load_pretrained_samba

print("="*80)
print("Full Model Forward Pass Test")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# 1. Load HF model
print("1. Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to(device)
print("   ✓ HF model loaded\n")

# 2. Create and load Samba model
print("2. Creating Samba model...")
samba_model = Samba(
    vocab_size=50280,
    d_model=768,
    n_layers=24,
    d_state=16,
    d_conv=4,
    expand_factor=2,
    readout_stride=4
).to(device)

print("3. Loading pretrained weights...")
samba_model = load_pretrained_samba(samba_model, "state-spaces/mamba-130m-hf", debug=False)
print()

# 4. Test input
print("4. Testing full forward pass...")
input_ids = torch.randint(0, 50280, (2, 16)).to(device)
print(f"   Input shape: {input_ids.shape}\n")

with torch.no_grad():
    # HF forward
    hf_outputs = hf_model(input_ids)
    hf_logits = hf_outputs.logits
    
    print("   HF model outputs:")
    print(f"     Logits shape: {hf_logits.shape}")
    print(f"     Mean: {hf_logits.mean():.6f}, Std: {hf_logits.std():.6f}")
    print(f"     Min: {hf_logits.min():.6f}, Max: {hf_logits.max():.6f}")
    print(f"     Has NaN: {torch.isnan(hf_logits).any()}")
    print(f"     Has Inf: {torch.isinf(hf_logits).any()}\n")
    
    # Samba forward
    try:
        main_logits, readout_logits, sampled_outputs, sampled_indices = samba_model(input_ids)
        
        print("   Samba model outputs:")
        print(f"     Main logits shape: {main_logits.shape}")
        print(f"     Mean: {main_logits.mean():.6f}, Std: {main_logits.std():.6f}")
        print(f"     Min: {main_logits.min():.6f}, Max: {main_logits.max():.6f}")
        print(f"     Has NaN: {torch.isnan(main_logits).any()}")
        print(f"     Has Inf: {torch.isinf(main_logits).any()}\n")
        
        print(f"     Readout logits shape: {readout_logits.shape}")
        print(f"     Readout Mean: {readout_logits.mean():.6f}, Std: {readout_logits.std():.6f}")
        print(f"     Readout Has NaN: {torch.isnan(readout_logits).any()}\n")
        
        print(f"     Sampled layers: {len(sampled_outputs)} (indices: {sampled_indices})")
        for i, output in enumerate(sampled_outputs):
            print(f"       Layer {sampled_indices[i]}: mean={output.mean():.6f}, std={output.std():.6f}, "
                  f"has_nan={torch.isnan(output).any()}")
        
        # Compare main logits
        print(f"\n   Comparing HF vs Samba main logits:")
        logits_diff = (hf_logits - main_logits).abs()
        print(f"     Max diff: {logits_diff.max():.6f}")
        print(f"     Mean diff: {logits_diff.mean():.6f}")
        print(f"     Median diff: {logits_diff.median():.6f}")
        
        if logits_diff.max() > 1.0:
            print(f"\n     ⚠️ WARNING: Large difference detected!")
        elif logits_diff.max() > 0.1:
            print(f"\n     ⚠️ Moderate difference (may be due to numerical precision)")
        else:
            print(f"\n     ✅ Outputs match very well!")
        
        # Test loss computation
        print(f"\n5. Testing loss computation...")
        targets = torch.randint(0, 50280, (2, 16)).to(device)
        
        # Main loss
        main_loss = torch.nn.functional.cross_entropy(
            main_logits.reshape(-1, 50280),
            targets.reshape(-1)
        )
        print(f"   Main loss: {main_loss.item():.4f} (NaN: {torch.isnan(main_loss)})")
        
        # Readout loss
        readout_loss = torch.nn.functional.cross_entropy(
            readout_logits.reshape(-1, 50280),
            targets.reshape(-1)
        )
        print(f"   Readout loss: {readout_loss.item():.4f} (NaN: {torch.isnan(readout_loss)})")
        
        # Pruning loss
        pruning_loss = sum(h.abs().mean() for h in sampled_outputs) / len(sampled_outputs)
        print(f"   Pruning loss: {pruning_loss.item():.4f} (NaN: {torch.isnan(pruning_loss)})")
        
        # Combined
        total_loss = main_loss + 0.5 * readout_loss + 0.05 * pruning_loss
        print(f"   Total loss: {total_loss.item():.4f} (NaN: {torch.isnan(total_loss)})")
        
        if not torch.isnan(total_loss):
            print(f"\n   ✅ All losses computed successfully!")
        else:
            print(f"\n   ❌ NaN detected in loss!")
            
    except Exception as e:
        print(f"   ❌ Samba forward failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Test Complete!")
print("="*80)
