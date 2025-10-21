"""
Data utilities for Samba training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DummyLanguageDataset(Dataset):
    """
    Dummy dataset for testing
    Generates random sequences with simple next-token prediction
    """
    
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        
        # Input: all tokens except last
        input_ids = sequence[:-1]
        
        # Target: all tokens except first (shifted by 1)
        targets = sequence[1:]
        
        return input_ids, targets


class SimplePatternDataset(Dataset):
    """
    Simple pattern dataset for sanity check
    Creates sequences with repeating patterns
    """
    
    def __init__(self, vocab_size, seq_len, num_samples=10000, pattern_length=4):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pattern_length = pattern_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a random pattern
        pattern = torch.randint(0, self.vocab_size, (self.pattern_length,))
        
        # Repeat pattern to fill sequence
        num_repeats = (self.seq_len + 1) // self.pattern_length + 1
        sequence = pattern.repeat(num_repeats)[:self.seq_len + 1]
        
        input_ids = sequence[:-1]
        targets = sequence[1:]
        
        return input_ids, targets


def get_dataloaders(vocab_size, seq_len, batch_size, dataset_type='dummy', 
                   train_samples=10000, val_samples=1000):
    """
    Get train and validation dataloaders
    
    Args:
        vocab_size: vocabulary size
        seq_len: sequence length
        batch_size: batch size
        dataset_type: 'dummy' or 'pattern'
        train_samples: number of training samples
        val_samples: number of validation samples
        
    Returns:
        train_loader, val_loader
    """
    
    if dataset_type == 'dummy':
        train_dataset = DummyLanguageDataset(vocab_size, seq_len, train_samples)
        val_dataset = DummyLanguageDataset(vocab_size, seq_len, val_samples)
    elif dataset_type == 'pattern':
        train_dataset = SimplePatternDataset(vocab_size, seq_len, train_samples)
        val_dataset = SimplePatternDataset(vocab_size, seq_len, val_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for simplicity
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


# For real datasets (to be implemented later)
def get_wikitext_dataloaders(seq_len, batch_size):
    """Load WikiText-2 or WikiText-103 dataset"""
    raise NotImplementedError("WikiText dataset loading not yet implemented")


def get_ptb_dataloaders(seq_len, batch_size):
    """Load Penn TreeBank dataset"""
    raise NotImplementedError("PTB dataset loading not yet implemented")
