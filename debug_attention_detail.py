"""
Detailed self-attention analysis for position 10
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

# Get inputs to self-attention
with torch.no_grad():
    _, _, all_layers = model(input_ids, targets)

readout = model.readout
y_stacked = torch.stack(all_layers, dim=0)
weights = torch.nn.functional.softmax(readout.layer_weights, dim=0)
memory = torch.einsum('l,lbsd->bsd', weights, y_stacked)

memory_shifted = torch.zeros_like(memory)
memory_shifted[:, 1:, :] = memory[:, :-1, :].clone()

decoder_input = readout.decoder.shift_right(targets, pad_token_id=readout.pad_token_id)

decoder = readout.decoder
token_emb = readout.parent_embedding(decoder_input)
positions = torch.arange(S, device=decoder_input.device).unsqueeze(0)
pos_emb = decoder.pos_embedding(positions)
x = token_emb + pos_emb
x = decoder.dropout(x)

layer0 = decoder.layers[0]
x_norm = layer0.norm1(x)

print("="*70)
print("SELF-ATTENTION DETAILED ANALYSIS")
print("="*70)

print(f"\nInput to self-attention (x_norm):")
print(f"  Position 10 diff: {(x_norm[0, 10] - x_norm[1, 10]).abs().max().item():.6f}")
print(f"  Position 11 diff: {(x_norm[0, 11] - x_norm[1, 11]).abs().max().item():.6f}")

# Manually compute QKV
self_attn = layer0.self_attn
qkv = self_attn.qkv_proj(x_norm)
q, k, v = qkv.chunk(3, dim=-1)

D = x_norm.shape[-1]
n_heads = self_attn.n_heads
d_head = self_attn.d_head

q = q.view(B, S, n_heads, d_head).transpose(1, 2)
k = k.view(B, S, n_heads, d_head).transpose(1, 2)
v = v.view(B, S, n_heads, d_head).transpose(1, 2)

print(f"\nQ, K, V shapes: {q.shape}")

# Check which keys are different
print(f"\nKey differences:")
for pos in range(S):
    diff = (k[0, :, pos, :] - k[1, :, pos, :]).abs().max().item()
    if diff > 0.01:
        print(f"  Position {pos}: diff={diff:.6f}")

# Check which values are different
print(f"\nValue differences:")
for pos in range(S):
    diff = (v[0, :, pos, :] - v[1, :, pos, :]).abs().max().item()
    if diff > 0.01:
        print(f"  Position {pos}: diff={diff:.6f}")

# Compute attention scores for position 10
attn = torch.matmul(q, k.transpose(-2, -1)) * self_attn.scale

# Apply mask
idx = torch.arange(S, device=x_norm.device)
causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
window = idx.unsqueeze(1) < (idx.unsqueeze(0) + self_attn.window_size)
mask = ~(causal & window)
attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

# Softmax
attn_weights = torch.softmax(attn, dim=-1)

print(f"\nAttention weights for position 10 (sample 0, head 0):")
print(f"  {attn_weights[0, 0, 10].cpu().numpy()}")

print(f"\nAttention weights for position 10 (sample 1, head 0):")
print(f"  {attn_weights[1, 0, 10].cpu().numpy()}")

print(f"\nPosition 10 attends to: {torch.where(~mask[10])[0].tolist()}")

# Check if attention weights are different
attn_diff = (attn_weights[0, :, 10, :] - attn_weights[1, :, 10, :]).abs()
print(f"\nAttention weight differences for position 10:")
for pos in range(S):
    diff = attn_diff[:, pos].max().item()
    if diff > 0.001:
        print(f"  Position {pos}: diff={diff:.6f}")

# Compute output
out = torch.matmul(attn_weights, v)

print(f"\nOutput diff for position 10: {(out[0, :, 10, :] - out[1, :, 10, :]).abs().max().item():.6f}")

print(f"\n" + "="*70)
print(f"FINDING:")
print(f"Position 10 attention weights should be identical if all keys [3-10] are identical")
print(f"But if key 11 is different and affects the attention computation somehow...")
print(f"="*70)
