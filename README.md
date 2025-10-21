# Samba: Sparse Mamba (130M Parameters)

**Samba** (Sparse Mamba) explores whether neural language models can use sparse representations efficiently, inspired by the sparse coding observed in biological neural systems.

## ✨ Features

- 🎯 **161M Parameters** - Mamba backbone (129M) + Attention readout (32M)
- 🔄 **Pretrained Weights** - Load from `state-spaces/mamba-130m-hf`
- 🧠 **Attention-based Readout** - Adaptive layer aggregation with minimal information loss
- 📊 **WikiText-2 Dataset** - Standard language modeling benchmark
- ⚙️ **Config-based Training** - YAML configuration for easy experimentation
- 🔍 **Sparsity Analysis** - Track and analyze SSM hidden state sparsity
- 📈 **Interpretability** - Visualize which layers contribute to predictions

## Motivation

Most state-of-the-art NLP models use dense representations, but the brain processes information sparsely. This suggests that sparse NLP processing should be theoretically possible. We investigate this by:

1. Drawing inspiration from Liquid State Machines (LSM) that can readout dense meaningful representations from sparse activations
2. Adapting Mamba (a State Space Model) which shares mathematical similarities with LSM but is trainable
3. Inducing sparsity in Mamba's hidden states while maintaining performance

## Approach

### Model Architecture

- **Base**: Mamba-130m (768d, 24 layers) - HuggingFace compatible
- **Readout**: Attention-based aggregation of all hidden states
  - Query/Key networks: Learn which layers are relevant
  - Value network: Transform layer representations  
  - Adaptive: Different timesteps attend to different layers
- **Dual outputs**: 
  - Main output from Mamba (for maintaining original performance)
  - Readout output from attended representation (for semantic verification)

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
- `α` (readout_weight): balances semantic preservation (default: 0.5)
- `λ` (pruning_weight): controls sparsity level (default: 0.05)

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `transformers>=4.39.0` (for pretrained Mamba)
- `datasets>=2.14.0` (for WikiText-2)
- `pyyaml>=6.0` (for config)

## Usage

### 1. Quick Start with Config

```bash
# Train with pretrained weights (recommended)
python train.py --config config/config.yaml

# Train from scratch
python train.py --config config/config.yaml --no-pretrained
```

### 2. Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model (130M parameters - HuggingFace compatible)
model:
  vocab_size: 50280
  d_model: 768
  n_layers: 24
  d_state: 16
  d_conv: 4
  expand_factor: 2

# Pretrained
pretrained:
  use_pretrained: true
  model_name: "state-spaces/mamba-130m-hf"
  freeze_backbone: false

# Training
training:
  batch_size: 16
  seq_len: 512
  epochs: 10
  lr: 5e-5
  readout_weight: 0.5  # α
  pruning_weight: 0.05  # λ

# Dataset
dataset:
  name: "wikitext"
  config_name: "wikitext-2-raw-v1"
  tokenizer: "gpt2"
```

### 3. Monitoring with W&B

```yaml
# In config.yaml
logging:
  use_wandb: true
  project_name: "samba"
```

Then run:
```bash
python train.py --config config/config.yaml
```

## Project Structure

```
samba/
├── config/
│   └── config.yaml       # Configuration file
├── model/
│   ├── mamba.py          # Mamba-130m (HF compatible)
│   └── samba.py          # Samba with readout
├── loss/
│   ├── readout_loss.py   # Readout loss
│   └── pruning_loss.py   # Pruning loss (L1)
├── utils/
│   ├── data.py           # WikiText-2 dataloader
│   └── pretrained_loader.py  # Load HF weights
├── train.py              # Training script
└── requirements.txt
```

## Model Details

### Architecture Specifications

```python
Samba (161M total):
- Vocabulary: 50,280 (GPT-2 tokenizer)
- Hidden Size: 768
- Layers: 24 Mamba blocks
- SSM State: 16

Parameters:
- Mamba backbone: 129M (80.2%)
- Attention Readout: 32M (19.8%)
  - Query/Key: 6M (attention mechanism)
  - Value: 13M (layer transformation)
  - Output: 13M (vocab projection)
- Total: ~161M parameters

Readout Design:
- Attention-based: Adaptive layer selection
- Minimal bottleneck: Preserves layer-specific info
- Interpretable: Can visualize layer importance
```

### Pretrained Weights

Loads weights from HuggingFace `state-spaces/mamba-130m-hf`:
- Trained on The Pile (300B tokens)
- Compatible architecture
- Automatic weight verification

## Research Questions

1. **Can we achieve comparable performance with sparse hidden states?**
   - Measure: Compare perplexity/accuracy of Samba vs. baseline Mamba
   
2. **How sparse can we make the representations?**
   - Measure: L0 norm, near-zero ratio of hidden states
   
3. **Do sparse representations preserve semantic information?**
   - Measure: Readout accuracy (how well the dense vector predicts)

## Expected Results

With proper hyperparameter tuning:
- **Target Sparsity**: 40-60% of hidden states near zero
- **Performance**: <10% degradation from baseline
- **Readout Accuracy**: Within 5% of main output

## Hyperparameter Tuning Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `readout_weight` (α) | 0.3-0.7 | Higher = better semantic preservation |
| `pruning_weight` (λ) | 0.01-0.1 | Higher = more aggressive sparsity |
| `learning_rate` | 1e-5 to 5e-5 | Lower if using pretrained |
| `batch_size` | 8-32 | Depends on GPU memory |

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
