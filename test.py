"""
Sanity check script for Samba
Quick test to verify model forward pass and loss computation
"""

import torch
from model.samba import Samba
from loss.readout_loss import ReadoutLossWithMetrics
from loss.pruning_loss import PruningLossWithMetrics


def test_model_forward():
    """Test model forward pass"""
    print("Testing model forward pass...")
    
    # Small model for testing
    vocab_size = 100
    batch_size = 4
    seq_len = 16
    
    model = Samba(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=8,
        expand_factor=2
    )
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    main_logits, readout_logits, all_hidden_states = model(input_ids)
    
    print(f"✓ Main logits shape: {main_logits.shape}")
    print(f"✓ Readout logits shape: {readout_logits.shape}")
    print(f"✓ Number of hidden state layers: {len(all_hidden_states)}")
    print(f"✓ Hidden state shape (per layer): {all_hidden_states[0].shape}")
    
    assert main_logits.shape == (batch_size, seq_len, vocab_size)
    assert readout_logits.shape == (batch_size, seq_len, vocab_size)
    
    print("✓ Model forward pass successful!\n")
    return model, input_ids, main_logits, readout_logits, all_hidden_states


def test_losses(model, input_ids, main_logits, readout_logits, all_hidden_states):
    """Test loss computation"""
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
    
    # Pruning loss
    pruning_loss_fn = PruningLossWithMetrics()
    pruning_loss, pruning_metrics = pruning_loss_fn(all_hidden_states)
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
    """Test backward pass"""
    print("Testing backward pass...")
    
    model = Samba(vocab_size=100, d_model=64, n_layers=2)
    input_ids = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))
    
    # Forward
    main_logits, readout_logits, all_hidden_states = model(input_ids)
    
    # Loss
    main_loss = torch.nn.functional.cross_entropy(
        main_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    readout_loss = torch.nn.functional.cross_entropy(
        readout_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    pruning_loss = sum(h.abs().mean() for h in all_hidden_states) / len(all_hidden_states)
    
    total_loss = main_loss + 0.5 * readout_loss + 0.1 * pruning_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")
    print("✓ Backward pass successful!\n")


def test_sparsity_stats():
    """Test sparsity statistics"""
    print("Testing sparsity statistics...")
    
    model = Samba(vocab_size=100, d_model=64, n_layers=2)
    input_ids = torch.randint(0, 100, (4, 16))
    
    _, _, all_hidden_states = model(input_ids)
    
    stats = model.get_sparsity_stats(all_hidden_states)
    
    print("Sparsity statistics:")
    for layer_name, layer_stats in stats.items():
        if isinstance(layer_stats, dict):
            print(f"  {layer_name}:")
            for k, v in layer_stats.items():
                print(f"    {k}: {v:.4f}")
    
    print(f"  Average near-zero ratio: {stats['avg_near_zero_ratio']:.4f}")
    print(f"  Average L1 norm: {stats['avg_l1_norm']:.4f}")
    
    print("✓ Sparsity statistics successful!\n")


def main():
    print("=" * 60)
    print("Samba Sanity Check")
    print("=" * 60 + "\n")
    
    # Test 1: Forward pass
    model, input_ids, main_logits, readout_logits, all_hidden_states = test_model_forward()
    
    # Test 2: Loss computation
    test_losses(model, input_ids, main_logits, readout_logits, all_hidden_states)
    
    # Test 3: Backward pass
    test_backward()
    
    # Test 4: Sparsity statistics
    test_sparsity_stats()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
