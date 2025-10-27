"""
Verify Teacher Forcing Implementation
Trace through exact data flow step by step
"""

import torch

# Simulate dataset output
print("="*60)
print("DATASET OUTPUT")
print("="*60)

# Original sequence in document
original_sequence = ['the', 'cat', 'sat', 'on', 'the', 'mat']
token_ids = [10, 20, 30, 40, 50, 60]

print(f"Original sequence: {original_sequence}")
print(f"Token IDs: {token_ids}")

# Dataset creates chunk
chunk = torch.tensor(token_ids)
input_chunk = chunk[:-1]  # [10, 20, 30, 40, 50]
target_chunk = chunk[1:]  # [20, 30, 40, 50, 60]

print(f"\nDataset output:")
print(f"  input_ids  = {input_chunk.tolist()}  # tokens: {original_sequence[:-1]}")
print(f"  targets    = {target_chunk.tolist()}  # tokens: {original_sequence[1:]}")

# Main model forward
print("\n" + "="*60)
print("MAIN MODEL (Mamba backbone)")
print("="*60)

input_ids = input_chunk
targets = target_chunk

print(f"Main model receives:")
print(f"  input_ids[t] → Mamba → hidden[t]")
print(f"  input_ids = {input_ids.tolist()}")

# Simulate Mamba processing
print(f"\nMamba processes each position:")
for t in range(len(input_ids)):
    print(f"  Position {t}: input={input_ids[t].item()} ('{original_sequence[t]}') → hidden[{t}]")

print(f"\nMain model predicts:")
print(f"  Position t uses hidden[t] to predict targets[t]")
for t in range(len(input_ids)):
    print(f"  Position {t}: hidden[{t}] → predict {targets[t].item()} ('{original_sequence[t+1]}')")

# Auxiliary decoder
print("\n" + "="*60)
print("AUXILIARY DECODER (Current Implementation - AFTER FIX)")
print("="*60)

# Memory from Mamba
memory = torch.randn(1, len(input_ids), 128)  # Simulated
print(f"Memory contains information from input_ids:")
for t in range(len(input_ids)):
    print(f"  memory[{t}] ← info from input[{t}]={input_ids[t].item()} ('{original_sequence[t]}')")

# Step 1: Memory shifting
memory_shifted = torch.zeros_like(memory)
memory_shifted[:, 1:, :] = memory[:, :-1, :].clone()

print(f"\nAfter memory shifting:")
for t in range(len(input_ids)):
    if t == 0:
        print(f"  memory_shifted[{t}] = zeros (no previous context)")
    else:
        print(f"  memory_shifted[{t}] = memory[{t-1}] ← info from '{original_sequence[t-1]}'")

# Step 2: Decoder input (CURRENT - using targets)
decoder_input = torch.zeros_like(targets)
decoder_input[1:] = targets[:-1].clone()
decoder_input[0] = 0  # PAD

print(f"\nDecoder input (Teacher Forcing with targets):")
print(f"  decoder_input = shift_right(targets)")
print(f"  decoder_input = {decoder_input.tolist()}")
for t in range(len(decoder_input)):
    if t == 0:
        print(f"  Position {t}: input=PAD")
    else:
        print(f"  Position {t}: input={decoder_input[t].item()} ('{original_sequence[t]}')")

# Step 3: Prediction
print(f"\nAuxiliary decoder prediction:")
print(f"  Position t receives:")
print(f"    - decoder_input[t] (token embedding)")
print(f"    - memory_shifted[t] (context via cross-attention)")
print(f"  → Predicts targets[t]")
print()

for t in range(len(input_ids)):
    if t == 0:
        decoder_token = "PAD"
        memory_info = "zeros"
        predict_token = original_sequence[t+1]
        predict_id = targets[t].item()
    else:
        decoder_token = original_sequence[t]
        memory_info = f"'{original_sequence[t-1]}'"
        predict_token = original_sequence[t+1]
        predict_id = targets[t].item()
    
    print(f"  Position {t}:")
    print(f"    Decoder input: {decoder_token}")
    print(f"    Memory from:   {memory_info}")
    print(f"    → Predict:     {predict_id} ('{predict_token}')")
    print()

# OLD implementation for comparison
print("\n" + "="*60)
print("OLD IMPLEMENTATION (BEFORE FIX - using input_ids)")
print("="*60)

decoder_input_old = torch.zeros_like(input_ids)
decoder_input_old[1:] = input_ids[:-1].clone()
decoder_input_old[0] = 0  # PAD

print(f"Decoder input (OLD - using input_ids):")
print(f"  decoder_input = shift_right(input_ids)")
print(f"  decoder_input = {decoder_input_old.tolist()}")

print(f"\nOLD auxiliary decoder prediction:")
for t in range(len(input_ids)):
    if t == 0:
        decoder_token = "PAD"
        memory_info = "zeros"
    else:
        decoder_token = original_sequence[t-1]  # Wrong! One step behind
        memory_info = f"'{original_sequence[t-1]}'"
    
    predict_token = original_sequence[t+1]
    predict_id = targets[t].item()
    
    print(f"  Position {t}:")
    print(f"    Decoder input: {decoder_token}")
    print(f"    Memory from:   {memory_info}")
    print(f"    → Predict:     {predict_id} ('{predict_token}')")
    
    if t > 0:
        print(f"    ❌ Problem: Seeing '{decoder_token}' but predicting '{predict_token}'")
        print(f"       Missing '{original_sequence[t]}' in between!")
    print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print("\n✓ NEW (Correct Teacher Forcing):")
print("  Position t receives:")
print("    - Token: targets[t-1] = actual token at position t")
print("    - Memory: info from position t-1")
print("    - Predicts: targets[t] = token at position t+1")
print("  → Standard autoregressive: Given current token, predict next")

print("\n✗ OLD (Wrong):")
print("  Position t receives:")
print("    - Token: input_ids[t-1] = actual token at position t-1")
print("    - Memory: info from position t-1")
print("    - Predicts: targets[t] = token at position t+1")
print("  → Skips current token! Predicts 2 steps ahead with 1 step context")

print("\n" + "="*60)
