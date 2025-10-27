"""
Step-by-step decoder debugging
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

# Create batch
torch.manual_seed(123)
input_ids = torch.randint(1, vocab_size, (B, S))
targets = torch.randint(1, vocab_size, (B, S))

input_ids[1, :] = input_ids[0, :]
targets[1, :] = targets[0, :]

t_test = 10
targets[1, t_test] = (targets[0, t_test] + 50) % vocab_size

if torch.cuda.is_available():
    input_ids = input_ids.cuda()
    targets = targets.cuda()

print("="*70)
print("STEP-BY-STEP DECODER DEBUGGING")
print("="*70)

# Get memory
with torch.no_grad():
    _, _, all_layers = model(input_ids, targets)

readout = model.readout
y_stacked = torch.stack(all_layers, dim=0)
weights = torch.nn.functional.softmax(readout.layer_weights, dim=0)
memory = torch.einsum('l,lbsd->bsd', weights, y_stacked)

# Memory shift
memory_shifted = torch.zeros_like(memory)
memory_shifted[:, 1:, :] = memory[:, :-1, :].clone()

# Decoder input
decoder_input = readout.decoder.shift_right(targets, pad_token_id=readout.pad_token_id)

print(f"\n✓ Memory shifted: identical for both samples")
print(f"✓ Decoder input position 11: different (8 vs 58)")
print(f"✓ All other decoder inputs: identical")

# STEP 1: Token embedding
decoder = readout.decoder
token_emb = readout.parent_embedding(decoder_input)

print(f"\n--- STEP 1: Token Embedding ---")
for pos in [0, 10, 11, 12]:
    diff = (token_emb[0, pos] - token_emb[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# STEP 2: Add positional embedding
positions = torch.arange(S, device=decoder_input.device).unsqueeze(0)
pos_emb = decoder.pos_embedding(positions)

print(f"\n--- STEP 2: Positional Embedding ---")
print(f"pos_emb shape: {pos_emb.shape}")
print(f"pos_emb[0] same as pos_emb[1]: {torch.equal(pos_emb[0], pos_emb[1])}")
for pos in [0, 10, 11, 12]:
    print(f"Position {pos}: {pos_emb[0, pos, :3].tolist()}")

x = token_emb + pos_emb

print(f"\n--- After Token + Pos Embedding ---")
for pos in [0, 10, 11, 12]:
    diff = (x[0, pos] - x[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# STEP 3: Dropout (should be disabled)
x = decoder.dropout(x)

print(f"\n--- STEP 3: After Dropout ---")
for pos in [0, 10, 11, 12]:
    diff = (x[0, pos] - x[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# STEP 4: Layer 0 - Self Attention
print(f"\n--- STEP 4: Layer 0 Self-Attention ---")
layer0 = decoder.layers[0]

# Pre-norm
x_norm = layer0.norm1(x)
print(f"After LayerNorm:")
for pos in [0, 10, 11, 12]:
    diff = (x_norm[0, pos] - x_norm[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# Self-attention
attn_out = layer0.self_attn(x_norm)
print(f"\nAfter Self-Attention:")
for pos in [0, 10, 11, 12]:
    diff = (attn_out[0, pos] - attn_out[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# Residual
x = x + attn_out
print(f"\nAfter Residual:")
for pos in [0, 10, 11, 12]:
    diff = (x[0, pos] - x[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# STEP 5: Layer 0 - Cross Attention
print(f"\n--- STEP 5: Layer 0 Cross-Attention ---")
x_norm = layer0.norm2(x)
cross_out = layer0.cross_attn(x_norm, memory_shifted, memory_shifted)
print(f"After Cross-Attention:")
for pos in [0, 10, 11, 12]:
    diff = (cross_out[0, pos] - cross_out[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

x = x + cross_out
print(f"\nAfter Residual:")
for pos in [0, 10, 11, 12]:
    diff = (x[0, pos] - x[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

# STEP 6: Layer 0 - FFN
print(f"\n--- STEP 6: Layer 0 FFN ---")
x_norm = layer0.norm3(x)
ffn_out = layer0.ffn(x_norm)
x = x + ffn_out
print(f"After FFN + Residual:")
for pos in [0, 10, 11, 12]:
    diff = (x[0, pos] - x[1, pos]).abs().max().item()
    print(f"Position {pos}: diff={diff:.6f}")

print(f"\n" + "="*70)
print(f"CONCLUSION:")
print(f"Find which step causes position 0 to diverge!")
print(f"="*70)
