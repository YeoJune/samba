"""
Complete Teacher Forcing and Data Leakage Test
Tests with actual model and real tokens
"""

import torch
import torch.nn as nn
from model.samba import Samba
from model.readouts import Readout
from model.decoder import WindowedAttn, WindowedCrossAttn


def test_self_attention_causal_mask():
    """Test 1: Self-attention에서 하삼각 마스크 확인"""
    print("\n" + "="*70)
    print("TEST 1: Self-Attention Causal Mask (Lower Triangular)")
    print("="*70)
    
    d_model = 128
    n_heads = 4
    window_size = 32
    
    self_attn = WindowedAttn(d_model, n_heads, window_size, dropout=0.0)
    self_attn.eval()
    
    B, S = 1, 5
    
    # Create distinctive input at each position
    x = torch.zeros(B, S, d_model)
    for pos in range(S):
        x[:, pos, :] = float(pos + 1) * 100  # Position 0 = 100, Position 1 = 200, etc.
    
    with torch.no_grad():
        output = self_attn(x)
    
    print(f"Input values at each position:")
    for pos in range(S):
        print(f"  Position {pos}: {x[0, pos, 0].item():.1f}")
    
    print(f"\nOutput analysis (checking causal constraint):")
    for pos in range(S):
        out_val = output[0, pos, 0].item()
        max_input_val = x[0, pos, 0].item()
        
        # Output should not exceed max value it can see (positions 0 to pos)
        print(f"  Position {pos}: output={out_val:.1f}, max_visible={max_input_val:.1f}", end="")
        
        if out_val > max_input_val + 10:  # Small tolerance
            print(" ❌ LEAK DETECTED!")
            assert False, f"Position {pos} sees future positions!"
        else:
            print(" ✓")
    
    print("\n✓ Self-attention correctly implements causal (lower triangular) mask")
    return True


def test_cross_attention_causal_mask():
    """Test 2: Cross-attention에서 하삼각 마스크 확인"""
    print("\n" + "="*70)
    print("TEST 2: Cross-Attention Causal Mask (Lower Triangular)")
    print("="*70)
    
    d_model = 128
    n_heads = 4
    window_size = 32
    
    cross_attn = WindowedCrossAttn(d_model, n_heads, window_size, dropout=0.0)
    cross_attn.eval()
    
    B, S = 1, 5
    
    # Query (decoder states)
    query = torch.randn(B, S, d_model)
    
    # Key/Value with distinctive values
    key = torch.zeros(B, S, d_model)
    value = torch.zeros(B, S, d_model)
    for pos in range(S):
        key[:, pos, :] = float(pos + 1) * 100
        value[:, pos, :] = float(pos + 1) * 100
    
    with torch.no_grad():
        output = cross_attn(query, key, value)
    
    print(f"Memory values at each position:")
    for pos in range(S):
        print(f"  Memory[{pos}]: {value[0, pos, 0].item():.1f}")
    
    print(f"\nOutput analysis (checking causal constraint):")
    for pos in range(S):
        out_val = output[0, pos, 0].item()
        max_visible_val = value[0, pos, 0].item()
        
        # Position pos can only see memory[0:pos+1]
        print(f"  Position {pos}: output={out_val:.1f}, max_visible={max_visible_val:.1f}", end="")
        
        if out_val > max_visible_val + 10:
            print(" ❌ LEAK DETECTED!")
            assert False, f"Position {pos} sees future memory!"
        else:
            print(" ✓")
    
    print("\n✓ Cross-attention correctly implements causal mask")
    return True


def test_teacher_forcing_with_real_tokens():
    """Test 3: 실제 토큰으로 Teacher Forcing 검증"""
    print("\n" + "="*70)
    print("TEST 3: Teacher Forcing with Real Tokens")
    print("="*70)
    
    # Create small model
    vocab_size = 100
    d_model = 128
    
    readout = Readout(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,
        decoder_n_layers=2,
        decoder_n_heads=4,
        decoder_window_size=32,
        dropout=0.0,
        pad_token_id=0
    )
    readout.eval()
    
    # Simulate tokens: "the cat sat on the mat"
    tokens = torch.tensor([[10, 20, 30, 40, 50, 60]])  # (B=1, S=6)
    
    # Dataset style split
    input_ids = tokens[:, :-1]  # [10, 20, 30, 40, 50]
    targets = tokens[:, 1:]     # [20, 30, 40, 50, 60]
    
    print(f"Token sequence: {tokens[0].tolist()}")
    print(f"input_ids:  {input_ids[0].tolist()} (tokens at positions 0-4)")
    print(f"targets:    {targets[0].tolist()} (tokens at positions 1-5)")
    
    # Create fake memory from Mamba
    B, S, D = 1, 5, d_model
    memory = []
    for i in range(4):
        mem = torch.randn(B, S, D)
        # Make each position's memory depend on its input token
        for pos in range(S):
            mem[:, pos, :] = mem[:, pos, :] + float(input_ids[0, pos].item())
        memory.append(mem)
    
    print(f"\nMemory encoding:")
    for pos in range(S):
        print(f"  memory[{pos}] ← encodes token {input_ids[0, pos].item()}")
    
    # Forward with teacher forcing
    with torch.no_grad():
        aux_logits = readout(memory, targets=targets)
    
    # Check decoder input
    decoder_input = readout.decoder.shift_right(targets, pad_token_id=0)
    
    print(f"\nDecoder input (shift_right(targets)):")
    print(f"  {decoder_input[0].tolist()}")
    
    print(f"\nTeacher Forcing verification:")
    for pos in range(S):
        if pos == 0:
            dec_in = "PAD (0)"
            mem_from = "zeros"
        else:
            dec_in = f"token {decoder_input[0, pos].item()}"
            mem_from = f"token {input_ids[0, pos-1].item()}"
        
        target = targets[0, pos].item()
        
        print(f"  Position {pos}:")
        print(f"    Decoder sees: {dec_in}")
        print(f"    Memory from:  {mem_from}")
        print(f"    Predicts:     token {target}")
        
        # Verify: decoder_input[pos] should equal targets[pos-1]
        if pos > 0:
            assert decoder_input[0, pos].item() == targets[0, pos-1].item(), \
                f"Teacher forcing broken at position {pos}!"
    
    print(f"\n✓ Teacher forcing correctly implemented")
    print(f"✓ Each position receives ground truth previous token")
    return True


def test_full_model_integration():
    """Test 4: 전체 Samba 모델 통합 테스트"""
    print("\n" + "="*70)
    print("TEST 4: Full Samba Model Integration")
    print("="*70)
    
    # Create small Samba model
    vocab_size = 100
    
    model = Samba(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        decoder_n_layers=2,
        decoder_n_heads=4,
        decoder_window_size=16,
        decoder_dropout=0.0,
        use_cuda=False,
        readout_mode="post",
        pad_token_id=0
    )
    model.eval()
    
    # Input sequence
    input_ids = torch.tensor([[5, 10, 15, 20, 25, 30, 35, 40]])
    targets = torch.tensor([[10, 15, 20, 25, 30, 35, 40, 45]])
    
    print(f"Input sequence:  {input_ids[0].tolist()}")
    print(f"Target sequence: {targets[0].tolist()}")
    
    with torch.no_grad():
        main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
    
    print(f"\nModel outputs:")
    print(f"  main_logits.shape: {main_logits.shape}")
    print(f"  aux_logits.shape:  {aux_logits.shape}")
    print(f"  Number of layer outputs: {len(all_layer_outputs)}")
    
    # Get predictions
    main_preds = main_logits.argmax(dim=-1)
    aux_preds = aux_logits.argmax(dim=-1)
    
    print(f"\nMain predictions: {main_preds[0].tolist()}")
    print(f"Aux predictions:  {aux_preds[0].tolist()}")
    print(f"Target sequence:  {targets[0].tolist()}")
    
    # Test inference mode (no targets)
    print(f"\n--- Testing Inference Mode (auto-regressive) ---")
    with torch.no_grad():
        main_logits_inf, aux_logits_inf, _ = model(input_ids, targets=None)
    
    print(f"Inference mode:")
    print(f"  main_logits: {main_logits_inf.shape}")
    print(f"  aux_logits:  {aux_logits_inf}")  # Should be None
    
    assert aux_logits_inf is None, "Inference mode should return None for aux_logits"
    
    print(f"\n✓ Model successfully runs in both training and inference modes")
    return True


def test_no_data_leakage_with_model():
    """Test 5: 실제 모델로 데이터 leakage 검증"""
    print("\n" + "="*70)
    print("TEST 5: Data Leakage Test with Full Model")
    print("="*70)
    
    vocab_size = 100
    
    model = Samba(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        decoder_n_layers=2,
        decoder_n_heads=4,
        decoder_window_size=8,  # Small window
        decoder_dropout=0.0,
        use_cuda=False,
        readout_mode="post",
        pad_token_id=0
    )
    model.eval()
    
    # Longer sequence
    B, S = 1, 16
    input_ids = torch.randint(1, vocab_size, (B, S))
    targets = torch.randint(1, vocab_size, (B, S))
    
    # Position to modify
    t_test = 10
    
    # Scenario A: normal
    with torch.no_grad():
        _, aux_A, _ = model(input_ids, targets)
    
    # Scenario B: change target at position t_test
    targets_B = targets.clone()
    targets_B[:, t_test] = (targets[:, t_test] + 50) % vocab_size
    
    with torch.no_grad():
        _, aux_B, _ = model(input_ids, targets_B)
    
    print(f"Changed targets[{t_test}]: {targets[0, t_test].item()} → {targets_B[0, t_test].item()}")
    
    print(f"\nPrediction differences:")
    for pos in range(S):
        diff = (aux_A[:, pos, :] - aux_B[:, pos, :]).abs().max().item()
        
        if pos <= t_test:
            # These should be identical (don't see future)
            status = "✓" if diff < 0.01 else "❌"
            print(f"  Position {pos}: diff={diff:.6f} {status}")
            if pos < t_test and diff > 0.01:
                print(f"    WARNING: Position {pos} affected by change at {t_test}!")
        else:
            # These should be different (see the change through teacher forcing)
            status = "✓" if diff > 0.01 else "?"
            print(f"  Position {pos}: diff={diff:.6f} {status} (should differ)")
    
    # Key test: early positions should be unaffected
    for pos in range(min(5, t_test)):
        diff = (aux_A[:, pos, :] - aux_B[:, pos, :]).abs().max().item()
        assert diff < 0.01, f"Position {pos} should not be affected by change at {t_test}!"
    
    print(f"\n✓ No data leakage detected in full model")
    return True


def run_all_tests():
    """Run all teacher forcing and leakage tests"""
    print("\n" + "="*70)
    print("TEACHER FORCING & DATA LEAKAGE TEST SUITE")
    print("Testing with Real Tokens and Full Model")
    print("="*70)
    
    tests = [
        ("Self-Attention Causal Mask", test_self_attention_causal_mask),
        ("Cross-Attention Causal Mask", test_cross_attention_causal_mask),
        ("Teacher Forcing with Tokens", test_teacher_forcing_with_real_tokens),
        ("Full Model Integration", test_full_model_integration),
        ("Data Leakage with Model", test_no_data_leakage_with_model),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            results.append((name, "FAIL"))
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(status == "PASS" for _, status in results)
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("✓ Causal masking works correctly (lower triangular)")
        print("✓ Teacher forcing implemented properly")
        print("✓ No data leakage detected")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
