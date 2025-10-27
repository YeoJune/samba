"""
Check self-attention mask in detail
"""
import torch

S = 16
window_size = 8

print("="*70)
print("ATTENTION MASK VERIFICATION")
print("="*70)

# Self-attention mask from decoder.py
idx = torch.arange(S)
causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # q_idx >= k_idx
window = idx.unsqueeze(1) < (idx.unsqueeze(0) + window_size)  # k_idx < q_idx + w
can_attend = causal & window

print(f"\nSelf-Attention Mask Logic:")
print(f"  causal: q_idx >= k_idx")
print(f"  window: k_idx < q_idx + window_size")

print(f"\nPosition 10 (q_idx=10):")
print(f"  Can attend to k where:")
print(f"    10 >= k  AND  k < 10 + 8")
print(f"    10 >= k  AND  k < 18")
print(f"  Result: k in [0, 10]")
print(f"  Actual: {torch.where(can_attend[10])[0].tolist()}")

print(f"\nPosition 11 (q_idx=11):")
print(f"  Can attend to k where:")
print(f"    11 >= k  AND  k < 11 + 8")
print(f"    11 >= k  AND  k < 19")
print(f"  Result: k in [0, 11]")
print(f"  Actual: {torch.where(can_attend[11])[0].tolist()}")

print(f"\n" + "="*70)
print(f"CRITICAL BUG FOUND!")
print(f"="*70)
print(f"Position 10 can see position 11: {can_attend[10, 11].item()}")
print(f"Position 11 can see position 10: {can_attend[11, 10].item()}")

print(f"\nThe mask is WRONG!")
print(f"  Current: causal = q_idx >= k_idx")
print(f"  Should be: causal = k_idx <= q_idx")
print(f"\nThey are the same mathematically, but let's check window constraint...")

print(f"\n  Current window: k_idx < q_idx + window_size")
print(f"  This is WRONG! Should be: k_idx >= q_idx - window_size + 1")

print(f"\nFor position 10 with window 8:")
print(f"  WRONG: k < 10 + 8 = k < 18  → k in [0, 17]")
print(f"  RIGHT: k >= 10 - 8 + 1 = k >= 3  → k in [3, 10]")

print(f"\nCombined with causal (k <= 10):")
print(f"  WRONG: [0, 10] ∩ [0, 17] = [0, 10]  (no window effect!)")
print(f"  RIGHT: [0, 10] ∩ [3, 10] = [3, 10]  (windowed!)")

print(f"\n" + "="*70)
print(f"CONCLUSION: Window constraint is inverted!")
print(f"="*70)
