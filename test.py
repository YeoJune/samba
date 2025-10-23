"""
Sanity check script for Samba (3-Loss Hybrid Architecture)
Quick test to verify:
1. Model forward pass
2. Weight sharing (embedding, aux_head)
3. Loss computation
4. Memory usage
"""

import torch
from model.samba import Samba
from loss.readout_loss import AuxLossWithMetrics
from loss.pruning_loss import L1LossWithMetrics


def test_weight_sharing():
    """Test that embeddings and heads are properly shared"""
    print("Testing weight sharing...")
    
    try:
        model = Samba(
            vocab_size=100,
            d_model=64,
            n_layers=8,
            decoder_n_layers=2,
            decoder_n_heads=4,
            decoder_window_size=16,
            use_cuda=False  # CPU for testing
        )
    except ImportError as e:
        print(f"❌ Error: {e}")
        return None
    
    # Check embedding sharing
    emb_shared = model.readout.parent_embedding is model.embedding
    print(f"  {'✓' if emb_shared else '❌'} Decoder uses Samba embedding: {emb_shared}")
    
    # Check aux_head sharing with lm_head
    head_shared = model.readout.aux_head.weight is model.lm_head.weight
    print(f"  {'✓' if head_shared else '❌'} aux_head shares weights with lm_head: {head_shared}")
    
    # Check lm_head tied with embedding
    lm_tied = model.lm_head.weight is model.embedding.weight
    print(f"  {'✓' if lm_tied else '❌'} lm_head tied with embedding: {lm_tied}")
    
    if emb_shared and head_shared and lm_tied:
        print("✓ All weight sharing properly configured!\n")
        return model
    else:
        print("❌ Weight sharing configuration error!\n")
        return None


def test_model_forward():
    """Test model forward pass (3-loss architecture)"""
    print("Testing model forward pass...")
    
    vocab_size = 100
    batch_size = 2
    seq_len = 32
    
    print("⚠️ Note: This test requires mamba-ssm to be installed")
    print("   Install with: pip install mamba-ssm causal-conv1d\n")
    
    try:
        model = Samba(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=8,
            d_state=8,
            expand_factor=2,
            decoder_n_layers=2,
            decoder_n_heads=4,
            decoder_window_size=16,
            use_cuda=True
        )
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Skipping test (mamba-ssm not installed)")
        return None, None, None, None, None
    
    # Random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    
    # Forward pass (new API: requires targets)
    main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
    
    print(f"✓ Main logits shape: {main_logits.shape}")
    print(f"✓ Aux logits shape: {aux_logits.shape}")
    print(f"✓ Number of layer outputs: {len(all_layer_outputs)}")
    print(f"✓ Layer output shape: {all_layer_outputs[0].shape}")
    
    assert main_logits.shape == (batch_size, seq_len, vocab_size)
    assert aux_logits.shape == (batch_size, seq_len, vocab_size)
    assert len(all_layer_outputs) == 8  # All 8 layers
    assert all_layer_outputs[0].shape == (batch_size, seq_len, 64)  # d_model=64
    
    print("✓ Model forward pass successful!\n")
    return model, input_ids, targets, main_logits, aux_logits, all_layer_outputs


def test_losses(model, input_ids, targets, main_logits, aux_logits, all_layer_outputs):
    """Test loss computation (3-loss system)"""
    print("Testing 3-loss computation...")
    
    vocab_size = main_logits.shape[-1]
    
    # Main loss
    main_loss_fn = torch.nn.CrossEntropyLoss()
    main_loss = main_loss_fn(main_logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"✓ Main loss: {main_loss.item():.4f}")
    
    # Auxiliary loss
    aux_loss_fn = AuxLossWithMetrics()
    aux_loss, aux_metrics = aux_loss_fn(aux_logits, targets)
    print(f"✓ Aux loss: {aux_loss.item():.4f}")
    print(f"  - Aux accuracy: {aux_metrics['aux_accuracy']:.4f}")
    
    # L1 loss (on all layer outputs)
    l1_loss_fn = L1LossWithMetrics()
    l1_loss, l1_metrics = l1_loss_fn(all_layer_outputs)
    print(f"✓ L1 loss: {l1_loss.item():.4f}")
    print(f"  - Near-zero ratio: {l1_metrics['avg_near_zero_ratio']:.4f}")
    print(f"  - L0 sparsity: {l1_metrics['avg_l0_sparsity']:.4f}")
    
    # Combined loss
    aux_weight = 0.5
    l1_weight = 0.05
    total_loss = main_loss + aux_weight * aux_loss + l1_weight * l1_loss
    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"  (Formula: main + {aux_weight}*aux + {l1_weight}*l1)")
    
    print("✓ Loss computation successful!\n")
    return total_loss


def test_backward():
    """Test backward pass (3-loss architecture)"""
    print("Testing backward pass...")
    
    try:
        model = Samba(
            vocab_size=100, 
            d_model=64, 
            n_layers=8,
            decoder_n_layers=2,
            decoder_n_heads=4,
            decoder_window_size=16,
            use_cuda=True
        ).cuda()
    except ImportError:
        print("⚠️ Skipping backward test (mamba-ssm not installed)")
        return
    
    input_ids = torch.randint(0, 100, (2, 32)).cuda()
    targets = torch.randint(0, 100, (2, 32)).cuda()
    
    # Forward
    main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
    
    # Losses
    main_loss = torch.nn.functional.cross_entropy(
        main_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    aux_loss = torch.nn.functional.cross_entropy(
        aux_logits.reshape(-1, 100), 
        targets.reshape(-1)
    )
    l1_loss = sum(h.abs().mean() for h in all_layer_outputs) / len(all_layer_outputs)
    
    total_loss = main_loss + 0.5 * aux_loss + 0.05 * l1_loss
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")
    print("✓ Backward pass successful!\n")


def test_memory():
    """Test memory usage with different batch sizes"""
    print("Testing memory usage...")
    
    try:
        model = Samba(
            vocab_size=50280, 
            d_model=768, 
            n_layers=24,
            decoder_n_layers=6,
            decoder_n_heads=12,
            decoder_window_size=32,
            use_cuda=True
        ).cuda()
    except ImportError:
        print("⚠️ Skipping memory test (mamba-ssm not installed)")
        return
    
    torch.cuda.reset_peak_memory_stats()
    
    seq_len = 512
    input_ids = torch.randint(0, 50280, (1, seq_len)).cuda()
    targets = torch.randint(0, 50280, (1, seq_len)).cuda()
    
    # Forward
    main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
    
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"  Memory allocated: {mem_allocated:.2f} GB")
    print(f"  Memory reserved: {mem_reserved:.2f} GB")
    print(f"  Peak memory: {mem_peak:.2f} GB")
    print(f"  Batch=1, Seq={seq_len}, Layers=24 (all stored)")
    
    print("✓ Memory test complete!\n")


def main():
    print("=" * 60)
    print("Samba Sanity Check (3-Loss Hybrid Architecture)")
    print("=" * 60 + "\n")
    
    # Test 0: Weight sharing
    model = test_weight_sharing()
    if model is None:
        return
    
    # Test 1: Forward pass
    result = test_model_forward()
    if result[0] is None:
        print("\n⚠️ Tests skipped - mamba-ssm not installed")
        print("Install with: pip install mamba-ssm causal-conv1d")
        return
    
    model, input_ids, targets, main_logits, aux_logits, all_layer_outputs = result
    
    # Test 2: Loss computation
    test_losses(model, input_ids, targets, main_logits, aux_logits, all_layer_outputs)
    
    # Test 3: Backward pass
    test_backward()
    
    # Test 4: Memory usage
    test_memory()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
