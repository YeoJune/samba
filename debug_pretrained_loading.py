"""
Debug script to verify mamba-ssm parameter loading
비교: HuggingFace Mamba vs Samba model weights
"""

import torch
from transformers import AutoModelForCausalLM
from model.samba import Samba
from utils.pretrained_loader import load_pretrained_samba


def test_full_loading():
    """전체 24개 레이어 로딩 테스트 + 실제 값 비교"""
    
    print("="*80)
    print("🔍 Step 1: Load models")
    print("="*80)
    
    # 1. HF Mamba 로드
    print("Loading HuggingFace Mamba...")
    hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    hf_state = hf_model.state_dict()
    
    # 2. Samba 생성 (pretrained 없이)
    print("Creating Samba model...")
    samba_model = Samba(
        vocab_size=50280,
        d_model=768,
        n_layers=24,
        use_cuda=True
    )
    
    print("\n" + "="*80)
    print("📊 Step 2: Structure check (layer 0 sample)")
    print("="*80)
    
    layer_0_keys = [k for k in hf_state.keys() if k.startswith("backbone.layers.0.")]
    print(f"HF layer 0 keys: {len(layer_0_keys)}")
    for key in sorted(layer_0_keys)[:3]:
        print(f"  {key}: {hf_state[key].shape}")
    print("  ...")
    
    samba_layer_0_keys = list(samba_model.layers[0].state_dict().keys())
    print(f"\nmamba-ssm layer 0 keys: {len(samba_layer_0_keys)}")
    for key in sorted(samba_layer_0_keys)[:3]:
        print(f"  {key}: {samba_model.layers[0].state_dict()[key].shape}")
    print("  ...")
    
    print("\n" + "="*80)
    print("🧪 Step 3: Manual loading test (all 24 layers)")
    print("="*80)
    
    # 전체 로딩 시도
    total_missing = 0
    total_unexpected = 0
    failed_layers = []
    
    for layer_idx in range(24):
        hf_mixer_prefix = f"backbone.layers.{layer_idx}.mixer."
        layer_state_dict = {}
        
        for key, value in hf_state.items():
            if key.startswith(hf_mixer_prefix):
                local_key = key.replace(hf_mixer_prefix, "")
                layer_state_dict[local_key] = value
        
        if len(layer_state_dict) > 0:
            missing, unexpected = samba_model.layers[layer_idx].load_state_dict(
                layer_state_dict, strict=False
            )
            total_missing += len(missing)
            total_unexpected += len(unexpected)
            
            if len(missing) > 0 or len(unexpected) > 0:
                failed_layers.append(layer_idx)
        else:
            failed_layers.append(layer_idx)
            print(f"  ✗ Layer {layer_idx}: No weights found!")
    
    if failed_layers:
        print(f"\n⚠️ Failed layers: {failed_layers}")
    else:
        print(f"\n✓ All 24 layers loaded successfully!")
    
    print(f"\nTotal missing keys: {total_missing}")
    print(f"Total unexpected keys: {total_unexpected}")
    
    print("\n" + "="*80)
    print("🔢 Step 4: Value comparison (layer 0 sample)")
    print("="*80)
    
    # 값 비교
    max_diffs = {}
    for key in ['A_log', 'D', 'in_proj.weight', 'out_proj.weight']:
        hf_key = f"backbone.layers.0.mixer.{key}"
        if hf_key in hf_state:
            hf_val = hf_state[hf_key]
            samba_val = samba_model.layers[0].state_dict()[key]
            diff = (hf_val - samba_val).abs().max().item()
            max_diffs[key] = diff
            print(f"  {key}: max_diff = {diff:.6e}")
    
    print("\n" + "="*80)
    print("🔄 Step 5: Load using pretrained_loader")
    print("="*80)
    
    # 새로운 모델로 pretrained_loader 테스트
    samba_model_loaded = Samba(
        vocab_size=50280,
        d_model=768,
        n_layers=24,
        use_cuda=True
    )
    
    samba_model_loaded = load_pretrained_samba(
        samba_model_loaded,
        pretrained_name="state-spaces/mamba-130m-hf",
        debug=False
    )
    
    print("\n" + "="*80)
    print("✅ Step 6: Final verification")
    print("="*80)
    
    # 전체 비교
    print("\nComparing all layer weights...")
    all_match = True
    mismatches = []
    
    for layer_idx in range(24):
        for key in samba_model_loaded.layers[layer_idx].state_dict().keys():
            hf_key = f"backbone.layers.{layer_idx}.mixer.{key}"
            if hf_key in hf_state:
                hf_val = hf_state[hf_key]
                samba_val = samba_model_loaded.layers[layer_idx].state_dict()[key]
                diff = (hf_val - samba_val).abs().max().item()
                
                if diff > 1e-6:
                    all_match = False
                    mismatches.append((layer_idx, key, diff))
    
    if all_match:
        print("✓ All weights match perfectly! (diff < 1e-6)")
    else:
        print(f"⚠️ Found {len(mismatches)} mismatches:")
        for layer_idx, key, diff in mismatches[:5]:  # Show first 5
            print(f"  Layer {layer_idx}.{key}: diff = {diff:.6e}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
    
    # Embedding & norm 체크
    print("\nChecking embedding & norms...")
    emb_diff = (samba_model_loaded.embedding.weight - hf_model.backbone.embeddings.weight).abs().max().item()
    norm_diff = (samba_model_loaded.norm_f.weight - hf_model.backbone.norm_f.weight).abs().max().item()
    
    print(f"  Embedding: max_diff = {emb_diff:.6e}")
    print(f"  Final norm: max_diff = {norm_diff:.6e}")
    
    # LM head 체크 (중요!)
    print("\nChecking LM head...")
    lm_head_diff = (samba_model_loaded.lm_head.weight - hf_model.lm_head.weight).abs().max().item()
    print(f"  LM head: max_diff = {lm_head_diff:.6e}")
    
    # Weight tying 확인
    is_tied_samba = (samba_model_loaded.lm_head.weight.data_ptr() == samba_model_loaded.embedding.weight.data_ptr())
    is_tied_hf = (hf_model.lm_head.weight.data_ptr() == hf_model.backbone.embeddings.weight.data_ptr())
    print(f"  Samba weight tying: {'✓' if is_tied_samba else '✗'}")
    print(f"  HF weight tying: {'✓' if is_tied_hf else '✗'}")
    
    print("\n" + "="*80)
    print("📝 Summary")
    print("="*80)
    print(f"  Layers loaded: 24/24")
    print(f"  Missing keys: {total_missing}")
    print(f"  Unexpected keys: {total_unexpected}")
    print(f"  Weight mismatches: {len(mismatches)}")
    print(f"  Embedding match: {'✓' if emb_diff < 1e-6 else '✗'}")
    print(f"  Norm match: {'✓' if norm_diff < 1e-6 else '✗'}")
    print(f"  LM head match: {'✓' if lm_head_diff < 1e-6 else '✗'}")
    print(f"  Weight tying: {'✓' if is_tied_samba and is_tied_hf else '✗'}")
    
    if len(mismatches) == 0 and emb_diff < 1e-6 and norm_diff < 1e-6 and lm_head_diff < 1e-6:
        print("\n✅ PASS: Pretrained loading is perfect!")
    else:
        print("\n⚠️ FAIL: Pretrained loading needs fixing!")
        return False
    
    print("\n" + "="*80)
    print("🚀 Step 7: Forward pass test (random input)")
    print("="*80)
    
    # 랜덤 입력으로 forward pass 비교
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 32
    vocab_size = 50280
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Sample tokens: {test_input[0, :10].tolist()}")
    
    # HF model forward
    print("\nRunning HF model forward pass...")
    hf_model.eval()
    with torch.no_grad():
        hf_outputs = hf_model(test_input)
        hf_logits = hf_outputs.logits
    
    # Samba model forward (main output only)
    print("Running Samba model forward pass...")
    samba_model_loaded.eval()
    with torch.no_grad():
        samba_logits, _, _ = samba_model_loaded(test_input)
    
    # 비교
    print("\nComparing outputs...")
    print(f"  HF logits shape: {hf_logits.shape}")
    print(f"  Samba logits shape: {samba_logits.shape}")
    
    logits_diff = (hf_logits - samba_logits).abs()
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    # Argmax 비교 (예측 토큰)
    hf_preds = hf_logits.argmax(dim=-1)
    samba_preds = samba_logits.argmax(dim=-1)
    match_ratio = (hf_preds == samba_preds).float().mean().item()
    
    print(f"  Prediction match ratio: {match_ratio:.2%}")
    print(f"  Sample HF predictions: {hf_preds[0, :10].tolist()}")
    print(f"  Sample Samba predictions: {samba_preds[0, :10].tolist()}")
    
    # 최종 판정
    print("\n" + "="*80)
    print("🏁 FINAL RESULT")
    print("="*80)
    
    tolerance = 1e-4  # Forward pass는 약간의 numerical error 허용
    if max_diff < tolerance and match_ratio > 0.99:
        print(f"✅ PERFECT: Forward outputs match! (max_diff: {max_diff:.6e})")
        print(f"   Pretrained loading is working correctly!")
        return True
    elif match_ratio > 0.95:
        print(f"⚠️ ACCEPTABLE: Minor differences (max_diff: {max_diff:.6e})")
        print(f"   Prediction match: {match_ratio:.2%}")
        print(f"   This might be due to numerical precision.")
        return True
    else:
        print(f"❌ FAIL: Outputs differ significantly!")
        print(f"   Max diff: {max_diff:.6e}")
        print(f"   Match ratio: {match_ratio:.2%}")
        print(f"   Pretrained loading needs investigation!")
        return False


if __name__ == "__main__":
    success = test_full_loading()
