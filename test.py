"""
Sanity check script for Samba
Quick test to verify model forward pass and loss computation
Tests new chunked mamba-ssm architecture
"""

import torch
from model.samba import Samba
from loss.readout_loss import ReadoutLossWithMetrics
from loss.pruning_loss import PruningLossWithMetrics


def test_model_forward():
    """Test model forward pass (new chunked architecture)"""
    print("Testing model forward pass...")
    
    # Small model for testing (must be divisible by readout_stride)
    vocab_size = 100
    batch_size = 4
    seq_len = 16
    
    print("⚠️ Note: This test requires mamba-ssm to be installed")
    print("   Install with: pip install mamba-ssm causal-conv1d\n")
    
    try:
        model = Samba(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=8,  # 8 layers = 2 chunks with stride=4
            d_state=8,
            expand_factor=2,
            readout_stride=4,  # 2 chunks
            use_cuda=True
        )
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Skipping test (mamba-ssm not installed)")
        return None, None, None, None, None, None
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass (new API returns 4 values)
    main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices = model(input_ids)
    
    print(f"✓ Main logits shape: {main_logits.shape}")
    print(f"✓ Readout logits shape: {readout_logits.shape}")
    print(f"✓ Number of sampled layers (chunks): {len(sampled_layer_outputs)}")
    print(f"✓ Layer output shape (per chunk): {sampled_layer_outputs[0].shape}")
    print(f"✓ Sampled layer indices: {sampled_layer_indices}")
    
    assert main_logits.shape == (batch_size, seq_len, vocab_size)
    assert readout_logits.shape == (batch_size, seq_len, vocab_size)
    assert len(sampled_layer_outputs) == 2  # 2 chunks
    assert sampled_layer_outputs[0].shape == (batch_size, seq_len, 64)  # d_model=64
    
    print("✓ Model forward pass successful!\n")
    return model, input_ids, main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices


def test_losses(model, input_ids, main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices):
    """Test loss computation (new API with sampled layer outputs)"""
    print("Testing loss computation...")
    
    vocab_size = main_logits.shape[-1]
    targets = torch.randint(0, vocab_size, input_ids.shape)
    
    # Main loss
    main_loss_fn = torch.nn.CrossEntropyLoss()
    main_loss = main_loss_fn(main_logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"✓ Main loss: {main_loss.item():.4f}")
    
    # Readout loss
    readout_loss_fn = ReadoutLossWithMetrics()
    readout_loss, readout_metrics = readout_loss_fn(readout_logits, targets)
    print(f"✓ Readout loss: {readout_loss.item():.4f}")
    print(f"  - Readout accuracy: {readout_metrics['readout_accuracy']:.4f}")
    
    # Pruning loss (uses sampled layer outputs, not hidden states)
    pruning_loss_fn = PruningLossWithMetrics()
    pruning_loss, pruning_metrics = pruning_loss_fn(sampled_layer_outputs, sampled_layer_indices)
    print(f"✓ Pruning loss (L1): {pruning_loss.item():.4f}")
    print(f"  - Near-zero ratio: {pruning_metrics['avg_near_zero_ratio']:.4f}")
    print(f"  - L0 sparsity: {pruning_metrics['avg_l0_sparsity']:.4f}")
    
    # Combined loss
    alpha = 0.5
    lambda_ = 0.1
    total_loss = main_loss + alpha * readout_loss + lambda_ * pruning_loss
    print(f"✓ Total loss: {total_loss.item():.4f}")
    
    print("✓ Loss computation successful!\n")
    return total_loss


def test_backward():
    """Test backward pass (new chunked architecture)"""
    print("Testing backward pass...")
    
    try:
        model = Samba(
            vocab_size=100, 
            d_model=64, 
            n_layers=8,  # 2 chunks with stride=4
            readout_stride=4,
            use_cuda=True
        )
    except ImportError:
        print("⚠️ Skipping backward test (mamba-ssm not installed)")
        return
    
    input_ids = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))
    
    # Forward (new API)
    main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices = model(input_ids)
    
    # Loss
    main_loss = torch.nn.functional.cross_entropy(
        main_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    readout_loss = torch.nn.functional.cross_entropy(
        readout_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    # Pruning loss on layer outputs (not hidden states)
    pruning_loss = sum(h.abs().mean() for h in sampled_layer_outputs) / len(sampled_layer_outputs)
    
    total_loss = main_loss + 0.5 * readout_loss + 0.1 * pruning_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")
    print("✓ Backward pass successful!\n")


def test_sparsity_stats():
    """Test sparsity statistics (new API with sampled layer outputs)"""
    print("Testing sparsity statistics...")
    
    try:
        model = Samba(
            vocab_size=100, 
            d_model=64, 
            n_layers=8,
            readout_stride=4,
            use_cuda=True
        )
    except ImportError:
        print("⚠️ Skipping sparsity test (mamba-ssm not installed)")
        return
    
    input_ids = torch.randint(0, 100, (4, 16))
    
    # Forward (new API)
    _, _, sampled_layer_outputs, _ = model(input_ids)
    
    # Get sparsity stats from model
    stats = model.get_sparsity_stats(sampled_layer_outputs)
    
    print("Sparsity statistics:")
    print(f"  Average near-zero ratio: {stats['avg_near_zero_ratio']:.4f}")
    print(f"  Average L0 sparsity: {stats['avg_l0_sparsity']:.4f}")
    print(f"  Average L1 norm: {stats['avg_l1_norm']:.4f}")
    
    print("✓ Sparsity statistics successful!\n")


def main():
    print("=" * 60)
    print("Samba Sanity Check (Chunked mamba-ssm Architecture)")
    print("=" * 60 + "\n")
    
    # Test 1: Forward pass
    result = test_model_forward()
    if result[0] is None:
        print("\n⚠️ Tests skipped - mamba-ssm not installed")
        print("Install with: pip install mamba-ssm causal-conv1d")
        return
    
    model, input_ids, main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices = result
    
    # Test 2: Loss computation
    test_losses(model, input_ids, main_logits, readout_logits, sampled_layer_outputs, sampled_layer_indices)
    
    # Test 3: Backward pass
    test_backward()
    
    # Test 4: Sparsity statistics
    test_sparsity_stats()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
