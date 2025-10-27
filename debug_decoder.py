"""
Debug decoder attention to understand why position 0-10 are affected
"""
import torch
import torch.nn as nn
from model.samba import Samba

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

vocab_size = 100
B, S = 2, 16

model = Samba(
    vocab_size=vocab_size,
    d_model=64,
    n_layers=2,
    d_state=16,
    d_conv=4,
    expand_factor=2,
    decoder_n_layers=2,
    decoder_n_heads=4,
    decoder_window_size=8,
    decoder_dropout=0.0,
    use_cuda=True,
    readout_mode="post",
    pad_token_id=0
)

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

# Create batch with identical inputs
torch.manual_seed(123)
input_ids = torch.randint(1, vocab_size, (B, S))
targets = torch.randint(1, vocab_size, (B, S))

input_ids[1, :] = input_ids[0, :]
targets[1, :] = targets[0, :]

# Change only targets[1, 10]
t_test = 10
targets[1, t_test] = (targets[0, t_test] + 50) % vocab_size

if torch.cuda.is_available():
    input_ids = input_ids.cuda()
    targets = targets.cuda()

print(f"Input IDs identical: {torch.equal(input_ids[0], input_ids[1])}")
print(f"Targets[0:10] identical: {torch.equal(targets[0, :t_test], targets[1, :t_test])}")
print(f"Targets[10] different: {targets[0, t_test].item()} vs {targets[1, t_test].item()}")
print(f"Targets[11:] identical: {torch.equal(targets[0, t_test+1:], targets[1, t_test+1:])}")

# Forward pass
with torch.no_grad():
    main_logits, aux_logits, all_layers = model(input_ids, targets)

# Check Mamba layers
print(f"\nMamba layers:")
for i in range(len(all_layers)):
    diff = (all_layers[i][0] - all_layers[i][1]).abs().max().item()
    print(f"  Layer {i}: max_diff={diff:.8f}")

# Get readout internals
print(f"\nInspecting Readout forward...")

# Manually call readout to inspect
from model.readouts import Readout
readout = model.readout

# LSM mixing
y_stacked = torch.stack(all_layers, dim=0)
print(f"y_stacked shape: {y_stacked.shape}")

weights = torch.nn.functional.softmax(readout.layer_weights, dim=0)
print(f"LSM weights: {weights}")

memory = torch.einsum('l,lbsd->bsd', weights, y_stacked)
print(f"memory shape: {memory.shape}")

# Check if memory is identical for both samples
memory_diff = (memory[0] - memory[1]).abs().max().item()
print(f"Memory max diff: {memory_diff:.8f}")

# Memory shifting
memory_shifted = torch.zeros_like(memory)
memory_shifted[:, 1:, :] = memory[:, :-1, :].clone()

memory_shifted_diff = (memory_shifted[0] - memory_shifted[1]).abs().max().item()
print(f"Memory shifted max diff: {memory_shifted_diff:.8f}")

# Decoder input
decoder_input = readout.decoder.shift_right(targets, pad_token_id=readout.pad_token_id)
print(f"\nDecoder input:")
print(f"  Sample 0: {decoder_input[0].tolist()}")
print(f"  Sample 1: {decoder_input[1].tolist()}")
print(f"  Difference at position 11: {decoder_input[0, 11].item()} vs {decoder_input[1, 11].item()}")

# Decoder embedding
decoder_emb = readout.decoder.embedding(decoder_input)
print(f"\nDecoder embedding shape: {decoder_emb.shape}")
emb_diff = (decoder_emb[0] - decoder_emb[1]).abs()
print(f"Embedding diffs by position:")
for pos in range(S):
    diff = emb_diff[pos].max().item()
    status = "DIFF" if diff > 0.01 else "SAME"
    print(f"  Position {pos}: {diff:.6f} {status}")

# Now the key question: does decoder layer 0 position 0 see position 11?
print(f"\nChecking decoder attention patterns...")
print(f"Window size: {readout.decoder.layers[0].self_attn.window_size}")

# Manually compute what position 0 can see in layer 0
window_size = 8
idx = torch.arange(S)
causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # q_idx >= k_idx
window = idx.unsqueeze(1) < (idx.unsqueeze(0) + window_size)  # k_idx < q_idx + w
can_attend = causal & window

print(f"\nPosition 0 can attend to: {torch.where(can_attend[0])[0].tolist()}")
print(f"Position 11 can attend to: {torch.where(can_attend[11])[0].tolist()}")

# Check if there's any path from position 11 to position 0 through layers
print(f"\n=== CRITICAL QUESTION ===")
print(f"Can information from position 11 leak to position 0?")
print(f"- Direct attention: NO (causal mask)")
print(f"- Through multiple layers: Checking...")

# In layer 1, position 0 uses layer 0's position 0 output
# Layer 0's position 0 cannot see position 11 (causal)
# So layer 1's position 0 also cannot see position 11

print(f"\nConclusion: Position 0 should NOT be affected by position 11 changes!")
print(f"But the test shows it IS affected... Why?")
