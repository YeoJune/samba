"""
Debug script to verify mamba-ssm parameter loading
비교: HuggingFace Mamba vs Samba model weights
"""

import torch
from transformers import AutoModelForCausalLM
from model.samba import Samba

def compare_weights():
    """HF Mamba와 Samba의 파라미터 비교"""
    
    print("="*80)
    print("🔍 Loading models...")
    print("="*80)
    
    # 1. HF Mamba 로드
    hf_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    hf_state = hf_model.state_dict()
    
    # 2. Samba 생성 (pretrained 없이)
    samba_model = Samba(
        vocab_size=50280,
        d_model=768,
        n_layers=24,
        use_cuda=True
    )
    
    print("\n" + "="*80)
    print("📊 HuggingFace Mamba structure (layer 0)")
    print("="*80)
    layer_0_keys = [k for k in hf_state.keys() if k.startswith("backbone.layers.0.")]
    for key in sorted(layer_0_keys):
        print(f"  {key}: {hf_state[key].shape}")
    
    print("\n" + "="*80)
    print("📊 mamba-ssm structure (layer 0)")
    print("="*80)
    samba_layer_0_keys = list(samba_model.layers[0].state_dict().keys())
    for key in sorted(samba_layer_0_keys):
        print(f"  {key}: {samba_model.layers[0].state_dict()[key].shape}")
    
    print("\n" + "="*80)
    print("🔧 Key mapping test")
    print("="*80)
    
    # 테스트: layer 0의 파라미터 매핑
    hf_mixer_prefix = "backbone.layers.0.mixer."
    hf_mixer_keys = [k for k in hf_state.keys() if k.startswith(hf_mixer_prefix)]
    
    print(f"\nHF mixer keys ({len(hf_mixer_keys)} total):")
    for key in sorted(hf_mixer_keys):
        local_key = key.replace(hf_mixer_prefix, "")
        has_match = local_key in samba_model.layers[0].state_dict()
        status = "✓" if has_match else "✗"
        print(f"  {status} {local_key}: {hf_state[key].shape}")
    
    print(f"\nmamba-ssm keys ({len(samba_layer_0_keys)} total):")
    for key in sorted(samba_layer_0_keys):
        hf_key = hf_mixer_prefix + key
        has_match = hf_key in hf_state
        status = "✓" if has_match else "✗"
        print(f"  {status} {key}: {samba_model.layers[0].state_dict()[key].shape}")
    
    print("\n" + "="*80)
    print("🧪 Manual weight loading test (layer 0)")
    print("="*80)
    
    # 수동으로 로딩 시도
    layer_state_dict = {}
    for key, value in hf_state.items():
        if key.startswith(hf_mixer_prefix):
            local_key = key.replace(hf_mixer_prefix, "")
            layer_state_dict[local_key] = value
    
    print(f"\nExtracted {len(layer_state_dict)} keys from HF")
    missing, unexpected = samba_model.layers[0].load_state_dict(layer_state_dict, strict=False)
    
    print(f"\n✓ load_state_dict completed")
    print(f"  Missing keys: {len(missing)}")
    if missing:
        print(f"    {missing}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"    {unexpected}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete")
    print("="*80)
    
    # 요약
    print("\n📝 Summary:")
    print(f"  HF mixer params: {len(hf_mixer_keys)}")
    print(f"  mamba-ssm params: {len(samba_layer_0_keys)}")
    print(f"  Missing after load: {len(missing)}")
    print(f"  Unexpected after load: {len(unexpected)}")
    
    if len(missing) > 0 or len(unexpected) > 0:
        print("\n⚠️ Parameter mismatch detected!")
        print("   This suggests the loading strategy needs adjustment.")
    else:
        print("\n✓ All parameters matched successfully!")


if __name__ == "__main__":
    compare_weights()
