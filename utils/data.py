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


class WikiTextDataset(Dataset):
    """
    WikiText dataset for language modeling
    Uses sliding window for long documents
    """
    
    def __init__(self, encodings, seq_len, stride):
        self.encodings = encodings
        self.seq_len = seq_len
        self.stride = stride
        
        # Calculate number of chunks
        self.num_chunks = 0
        for input_ids in encodings['input_ids']:
            if len(input_ids) > seq_len:
                self.num_chunks += (len(input_ids) - seq_len) // stride + 1
        
        # Create index mapping
        self.index_map = []
        for doc_idx, input_ids in enumerate(encodings['input_ids']):
            if len(input_ids) <= seq_len:
                self.index_map.append((doc_idx, 0))
            else:
                for start_idx in range(0, len(input_ids) - seq_len, stride):
                    self.index_map.append((doc_idx, start_idx))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        doc_idx, start_idx = self.index_map[idx]
        input_ids = self.encodings['input_ids'][doc_idx]
        
        # Extract chunk
        end_idx = start_idx + self.seq_len + 1
        chunk = input_ids[start_idx:end_idx]
        
        # Pad if necessary
        pad_length = self.seq_len + 1 - len(chunk)
        if pad_length > 0:
            chunk = torch.cat([
                chunk, 
                torch.full((pad_length,), 50256, dtype=torch.long)  # PAD token
            ])
        
        # Split into input and target
        input_chunk = chunk[:-1]
        target_chunk = chunk[1:].clone()
        
        # Replace padding tokens in targets with -100 (ignore_index for loss)
        # Padding starts at position (original_length - 1)
        if pad_length > 0:
            target_chunk[-(pad_length):] = -100
        
        return input_chunk, target_chunk


def get_wikitext_dataloaders(config):
    """
    Get WikiText dataloaders
    
    Args:
        config: Config dict with dataset settings
        
    Returns:
        train_loader, val_loader
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Please install: pip install datasets transformers")
    
    print(f"Loading {config['name']} dataset...")
    
    # Load dataset
    dataset = load_dataset(config['name'], config['config_name'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizing dataset...")
    
    # Tokenize function
    def tokenize_function(examples):
        # Filter out empty texts
        texts = [text for text in examples['text'] if text.strip()]
        if not texts:
            return {'input_ids': []}
        
        return tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,  # We'll handle long sequences with sliding window
            return_tensors=None
        )
    
    # Tokenize datasets
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing train"
    )
    
    tokenized_val = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing validation"
    )
    
    # Convert to our dataset format
    train_encodings = {'input_ids': [torch.tensor(ids) for ids in tokenized_train['input_ids'] if len(ids) > 0]}
    val_encodings = {'input_ids': [torch.tensor(ids) for ids in tokenized_val['input_ids'] if len(ids) > 0]}
    
    seq_len = config.get('max_length', 512)
    stride = config.get('stride', 256)
    
    train_dataset = WikiTextDataset(train_encodings, seq_len, stride)
    val_dataset = WikiTextDataset(val_encodings, seq_len, stride)
    
    print(f"Train dataset size: {len(train_dataset)} chunks")
    print(f"Validation dataset size: {len(val_dataset)} chunks")
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_dataloaders(vocab_size, seq_len, batch_size, dataset_type='dummy', 
                   train_samples=10000, val_samples=1000):
    """
    Get train and validation dataloaders (for testing)
    
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
        num_workers=0,
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

