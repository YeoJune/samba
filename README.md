# Samba: Sparse Mamba

**Samba** (Sparse Mamba) explores whether neural language models can use sparse representations efficiently, inspired by the sparse coding observed in biological neural systems.

## Motivation

Most state-of-the-art NLP models use dense representations, but the brain processes information sparsely. This suggests that sparse NLP processing should be theoretically possible. We investigate this by:

1. Drawing inspiration from Liquid State Machines (LSM) that can readout dense meaningful representations from sparse activations
2. Adapting Mamba (a State Space Model) which shares mathematical similarities with LSM but is trainable
3. Inducing sparsity in Mamba's hidden states while maintaining performance

## Approach

### Model Architecture

- **Base**: Mamba model with multiple layers
- **Readout**: MLP that aggregates all hidden states into a dense vector
- **Dual outputs**: 
  - Main output from Mamba (for maintaining original performance)
  - Readout output from dense vector (for semantic verification)

### Loss Functions

1. **Main Loss**: Standard cross-entropy on Mamba's output
   - Ensures the model maintains language modeling performance
   
2. **Readout Loss**: Cross-entropy on the dense readout vector
   - Forces the aggregated hidden states to contain meaningful semantic information
   - Verifies that sparse representations preserve task-relevant information
   
3. **Pruning Loss**: L1 regularization on hidden states
   - Encourages sparsity in hidden state activations
   - Mimics the sparse coding in biological neural systems

```
Total Loss = Main Loss + α × Readout Loss + λ × Pruning Loss
```

Where:
- `α` (readout_weight): balances semantic preservation
- `λ` (pruning_weight): controls sparsity level

## Project Structure

```
samba/
├── model/
│   ├── mamba.py          # Base Mamba implementation
│   └── samba.py          # Samba with MLP readout
├── loss/
│   ├── readout_loss.py   # Auxiliary loss for dense vector
│   └── pruning_loss.py   # L1 sparsity regularization
├── utils/
│   └── data.py           # Data loading utilities
├── train.py              # Training script
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Dummy Data)

```bash
python train.py \
    --vocab_size 10000 \
    --d_model 256 \
    --n_layers 4 \
    --batch_size 32 \
    --seq_len 128 \
    --epochs 10 \
    --readout_weight 0.5 \
    --pruning_weight 0.1
```

### With W&B Logging

```bash
python train.py \
    --use_wandb \
    --project_name samba \
    --readout_weight 0.5 \
    --pruning_weight 0.1
```

### Key Hyperparameters

- `--readout_weight` (α): Weight for readout loss (default: 0.5)
  - Higher → stronger semantic preservation
  - Lower → more freedom for main model
  
- `--pruning_weight` (λ): Weight for pruning loss (default: 0.1)
  - Higher → more aggressive sparsity
  - Lower → denser representations

## Research Questions

1. **Can we achieve comparable performance with sparse hidden states?**
   - Measure: Compare perplexity/accuracy of Samba vs. baseline Mamba
   
2. **How sparse can we make the representations?**
   - Measure: L0 norm, near-zero ratio of hidden states
   
3. **Do sparse representations preserve semantic information?**
   - Measure: Readout accuracy (how well the dense vector predicts)

## Results

(To be filled after experiments)

- **Sparsity achieved**: X% of hidden state values near zero
- **Performance**: Y% of baseline Mamba performance
- **Efficiency**: Z% reduction in active parameters

## Future Work

- [ ] Test on real language modeling datasets (WikiText, PTB)
- [ ] Compare with other sparsity methods (magnitude pruning, top-k)
- [ ] Analyze learned sparse patterns
- [ ] Investigate computational efficiency gains
- [ ] Extend to larger models

## Citation

```bibtex
@misc{samba2024,
  title={Samba: Sparse Representation Learning in Mamba Models},
  author={[Authors]},
  year={2024}
}
```

## License

MIT
