"""
Training script for Samba
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
import wandb
import os

from model.samba import Samba
from loss.readout_loss import ReadoutLossWithMetrics
from loss.pruning_loss import PruningLossWithMetrics
from utils.data import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train Samba model')
    
    # Model args
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--expand_factor', type=int, default=2)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Loss weights
    parser.add_argument('--readout_weight', type=float, default=0.5, 
                       help='Weight for readout loss (α)')
    parser.add_argument('--pruning_weight', type=float, default=0.1, 
                       help='Weight for pruning loss (λ)')
    
    # Others
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='samba')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, main_loss_fn, readout_loss_fn, 
                pruning_loss_fn, args, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_readout_loss = 0
    total_pruning_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        input_ids = input_ids.to(args.device)
        targets = targets.to(args.device)
        
        # Forward pass
        main_logits, readout_logits, all_hidden_states = model(input_ids)
        
        # Calculate losses
        # Main loss: original Mamba loss
        main_loss = main_loss_fn(
            main_logits.reshape(-1, args.vocab_size), 
            targets.reshape(-1)
        )
        
        # Readout loss: auxiliary loss on dense vector
        readout_loss, readout_metrics = readout_loss_fn(readout_logits, targets)
        
        # Pruning loss: L1 sparsity on hidden states
        pruning_loss, pruning_metrics = pruning_loss_fn(all_hidden_states)
        
        # Combined loss
        loss = main_loss + \
               args.readout_weight * readout_loss + \
               args.pruning_weight * pruning_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_readout_loss += readout_loss.item()
        total_pruning_loss += pruning_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'main': main_loss.item(),
            'readout': readout_loss.item(),
            'pruning': pruning_loss.item(),
            'sparsity': pruning_metrics['avg_near_zero_ratio']
        })
        
        # Log to wandb
        if args.use_wandb and batch_idx % args.log_interval == 0:
            wandb.log({
                'train/total_loss': loss.item(),
                'train/main_loss': main_loss.item(),
                'train/readout_loss': readout_loss.item(),
                'train/pruning_loss': pruning_loss.item(),
                'train/readout_accuracy': readout_metrics['readout_accuracy'],
                'train/sparsity': pruning_metrics['avg_near_zero_ratio'],
                'train/l1_norm': pruning_metrics['l1_loss'],
                'step': epoch * len(dataloader) + batch_idx
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_readout_loss = total_readout_loss / len(dataloader)
    avg_pruning_loss = total_pruning_loss / len(dataloader)
    
    return avg_loss, avg_main_loss, avg_readout_loss, avg_pruning_loss


@torch.no_grad()
def evaluate(model, dataloader, main_loss_fn, readout_loss_fn, pruning_loss_fn, args):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_readout_loss = 0
    total_pruning_loss = 0
    
    for input_ids, targets in tqdm(dataloader, desc='Evaluating'):
        input_ids = input_ids.to(args.device)
        targets = targets.to(args.device)
        
        # Forward pass
        main_logits, readout_logits, all_hidden_states = model(input_ids)
        
        # Calculate losses
        main_loss = main_loss_fn(
            main_logits.reshape(-1, args.vocab_size), 
            targets.reshape(-1)
        )
        readout_loss, _ = readout_loss_fn(readout_logits, targets)
        pruning_loss, pruning_metrics = pruning_loss_fn(all_hidden_states)
        
        loss = main_loss + \
               args.readout_weight * readout_loss + \
               args.pruning_weight * pruning_loss
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_readout_loss += readout_loss.item()
        total_pruning_loss += pruning_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_readout_loss = total_readout_loss / len(dataloader)
    avg_pruning_loss = total_pruning_loss / len(dataloader)
    
    # Get sparsity stats
    sparsity_stats = model.get_sparsity_stats(all_hidden_states)
    
    return avg_loss, avg_main_loss, avg_readout_loss, avg_pruning_loss, sparsity_stats


def main():
    args = parse_args()
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.project_name, config=vars(args))
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize model
    model = Samba(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand_factor=args.expand_factor
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Initialize losses
    main_loss_fn = nn.CrossEntropyLoss()
    readout_loss_fn = ReadoutLossWithMetrics()
    pruning_loss_fn = PruningLossWithMetrics()
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_main, train_readout, train_pruning = train_epoch(
            model, train_loader, optimizer, main_loss_fn, 
            readout_loss_fn, pruning_loss_fn, args, epoch
        )
        
        # Evaluate
        val_loss, val_main, val_readout, val_pruning, sparsity_stats = evaluate(
            model, val_loader, main_loss_fn, readout_loss_fn, 
            pruning_loss_fn, args
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} (Main: {train_main:.4f}, "
              f"Readout: {train_readout:.4f}, Pruning: {train_pruning:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Main: {val_main:.4f}, "
              f"Readout: {val_readout:.4f}, Pruning: {val_pruning:.4f})")
        print(f"Sparsity: {sparsity_stats['avg_near_zero_ratio']:.4f}, "
              f"L1: {sparsity_stats['avg_l1_norm']:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'val/total_loss': val_loss,
                'val/main_loss': val_main,
                'val/readout_loss': val_readout,
                'val/pruning_loss': val_pruning,
                'val/sparsity': sparsity_stats['avg_near_zero_ratio'],
                'val/l1_norm': sparsity_stats['avg_l1_norm'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sparsity_stats': sparsity_stats,
                'args': vars(args)
            }, save_path)
            print(f"Saved best model to {save_path}")
    
    print("\nTraining completed!")
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
