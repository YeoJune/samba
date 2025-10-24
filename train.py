"""
Training script for Samba (Config-based)
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
import yaml

from model.samba import Samba
from loss.readout_loss import AuxLossWithMetrics
from loss.pruning_loss import L1LossWithMetrics
from utils.data import get_wikitext_dataloaders
from utils.pretrained_loader import (
    load_pretrained_samba, 
    verify_samba_weights, 
    load_pretrained_decoder
)


def load_config(config_path):
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Samba model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch (ignore pretrained setting)')
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, main_loss_fn, aux_loss_fn, 
                l1_loss_fn, config, epoch, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_l1_loss = 0
    total_main_acc = 0
    total_aux_acc = 0
    
    use_amp = config['training'].get('use_amp', False)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass (now requires targets for auxiliary decoder)
            main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
            
            vocab_size = main_logits.size(-1)
            
            # Calculate losses
            main_loss = main_loss_fn(
                main_logits.reshape(-1, vocab_size), 
                targets.reshape(-1)
            )
            
            aux_loss, aux_metrics = aux_loss_fn(aux_logits, targets)
            l1_loss, l1_metrics = l1_loss_fn(all_layer_outputs)
            
            # Combined loss (3-Loss system)
            aux_weight = config['training']['aux_weight']
            l1_weight = config['training']['l1_weight']
            
            loss = main_loss + \
                   aux_weight * aux_loss + \
                   l1_weight * l1_loss
        
        # Calculate accuracies (outside autocast, detached from graph)
        with torch.no_grad():
            main_preds = main_logits.argmax(dim=-1)
            main_acc = (main_preds == targets).float().mean().item()
        aux_acc = aux_metrics['aux_accuracy']  # already .item() in loss function
        
        # Backward pass with AMP
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_l1_loss += l1_loss.item()
        total_main_acc += main_acc
        total_aux_acc += aux_acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'main': f"{main_loss.item():.4f}",
            'aux': f"{aux_loss.item():.4f}",
            'l1': f"{l1_loss.item():.4f}",
            'main_acc': f"{main_acc:.3f}",
            'aux_acc': f"{aux_acc:.3f}",
            'sparsity': f"{l1_metrics['avg_near_zero_ratio']:.3f}"
        })
        
        # Log to wandb
        if config['logging']['use_wandb'] and batch_idx % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/total_loss': loss.item(),
                'train/main_loss': main_loss.item(),
                'train/aux_loss': aux_loss.item(),
                'train/l1_loss': l1_loss.item(),
                'train/main_accuracy': main_acc,
                'train/aux_accuracy': aux_acc,
                'train/sparsity': l1_metrics['avg_near_zero_ratio'],
                'train/l1_norm': l1_metrics['l1_loss'],
                'step': epoch * len(dataloader) + batch_idx
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_aux_loss = total_aux_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_main_acc = total_main_acc / len(dataloader)
    avg_aux_acc = total_aux_acc / len(dataloader)
    
    return avg_loss, avg_main_loss, avg_aux_loss, avg_l1_loss, avg_main_acc, avg_aux_acc


@torch.no_grad()
def evaluate(model, dataloader, main_loss_fn, aux_loss_fn, l1_loss_fn, config, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_l1_loss = 0
    total_main_acc = 0
    total_aux_acc = 0
    
    use_amp = config['training'].get('use_amp', False)
    
    for input_ids, targets in tqdm(dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass (requires targets for auxiliary decoder)
            main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
            
            vocab_size = main_logits.size(-1)
            
            # Calculate losses
            main_loss = main_loss_fn(
                main_logits.reshape(-1, vocab_size), 
                targets.reshape(-1)
            )
            aux_loss, aux_metrics = aux_loss_fn(aux_logits, targets)
            l1_loss, l1_metrics = l1_loss_fn(all_layer_outputs)
            
            aux_weight = config['training']['aux_weight']
            l1_weight = config['training']['l1_weight']
            
            loss = main_loss + \
                   aux_weight * aux_loss + \
                   l1_weight * l1_loss
        
        # Calculate accuracies (outside autocast, already in no_grad context)
        main_preds = main_logits.argmax(dim=-1)
        main_acc = (main_preds == targets).float().mean().item()
        aux_acc = aux_metrics['aux_accuracy']  # already .item() in loss function
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_l1_loss += l1_loss.item()
        total_main_acc += main_acc
        total_aux_acc += aux_acc
    
    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_aux_loss = total_aux_loss / len(dataloader)
    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_main_acc = total_main_acc / len(dataloader)
    avg_aux_acc = total_aux_acc / len(dataloader)
    
    # Get sparsity stats from all layer outputs
    sparsity_stats = model.get_sparsity_stats(all_layer_outputs)
    
    return avg_loss, avg_main_loss, avg_aux_loss, avg_l1_loss, avg_main_acc, avg_aux_acc, sparsity_stats


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Set device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config.get('seed', 42))
    
    # Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project_name'], 
            config=config
        )
    
    # Create save directory
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Initialize model
    print("\n" + "="*80)
    print("Initializing model...")
    print("="*80)
    
    model_config = config['model']
    model = Samba(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        d_state=model_config['d_state'],
        d_conv=model_config['d_conv'],
        expand_factor=model_config['expand_factor'],
        dt_rank=model_config.get('dt_rank', 'auto'),
        decoder_n_layers=model_config.get('decoder_n_layers', 6),
        decoder_n_heads=model_config.get('decoder_n_heads', 12),
        decoder_window_size=model_config.get('decoder_window_size', 32),
        decoder_dropout=model_config.get('decoder_dropout', 0.1),
        readout_mode=model_config.get('readout_mode', 'post')
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Readout mode: {model_config.get('readout_mode', 'post')}")
    
    # Load pretrained weights
    if config['pretrained']['use_pretrained'] and not args.no_pretrained:
        print("\n" + "="*80)
        print("Loading pretrained weights...")
        print("="*80)
        
        # Load Mamba backbone weights
        model = load_pretrained_samba(
            model,
            config['pretrained']['mamba_model'],
            debug=True  # Show key mapping details
        )
        
        # Verify Mamba weights
        verify_samba_weights(
            model,
            config['pretrained']['mamba_model']
        )
        
        # Load GPT-2 decoder weights
        model.readout.decoder = load_pretrained_decoder(
            model.readout.decoder,
            pretrained_name=config['pretrained']['decoder_model'],
            target_vocab_size=config['model']['vocab_size'],
            debug=True
        )
        
        # Freeze backbone if specified
        if config['pretrained']['freeze_backbone']:
            print("\nFreezing Mamba backbone (layers)...")
            for layer in model.layers:
                for param in layer.parameters():
                    param.requires_grad = False
            print("✓ Backbone frozen (only readout will be trained)")
    else:
        print("\n⚠️ Training from scratch (no pretrained weights)")
    
    # Initialize losses
    main_loss_fn = nn.CrossEntropyLoss()
    aux_loss_fn = AuxLossWithMetrics()
    l1_loss_fn = L1LossWithMetrics()
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Initialize AMP scaler
    use_amp = config['training'].get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    if use_amp:
        print("\n✓ AMP (Automatic Mixed Precision) enabled - FP16 training")
    
    # Get dataloaders
    print("\n" + "="*80)
    print("Loading dataset...")
    print("="*80)
    
    dataset_config = {
        **config['dataset'],
        'batch_size': config['training']['batch_size']
    }
    
    train_loader, val_loader = get_wikitext_dataloaders(dataset_config)
    
    print(f"✓ Dataset loaded: {config['dataset']['name']}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_main, train_aux, train_l1, train_main_acc, train_aux_acc = train_epoch(
            model, train_loader, optimizer, main_loss_fn, 
            aux_loss_fn, l1_loss_fn, config, epoch, device, scaler
        )
        
        # Evaluate
        val_loss, val_main, val_aux, val_l1, val_main_acc, val_aux_acc, sparsity_stats = evaluate(
            model, val_loader, main_loss_fn, aux_loss_fn, 
            l1_loss_fn, config, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f} (Main: {train_main:.4f}, "
              f"Aux: {train_aux:.4f}, L1: {train_l1:.4f})")
        print(f"Train Acc: Main: {train_main_acc:.4f}, Aux: {train_aux_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} (Main: {val_main:.4f}, "
              f"Aux: {val_aux:.4f}, L1: {val_l1:.4f})")
        print(f"Val Acc: Main: {val_main_acc:.4f}, Aux: {val_aux_acc:.4f}")
        print(f"Sparsity: {sparsity_stats['avg_near_zero_ratio']:.4f}, "
              f"L1: {sparsity_stats['avg_l1_norm']:.4f}")
        
        # Log to wandb
        if config['logging']['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_main_acc': train_main_acc,
                'train/epoch_aux_acc': train_aux_acc,
                'val/total_loss': val_loss,
                'val/main_loss': val_main,
                'val/aux_loss': val_aux,
                'val/l1_loss': val_l1,
                'val/main_accuracy': val_main_acc,
                'val/aux_accuracy': val_aux_acc,
                'val/sparsity': sparsity_stats['avg_near_zero_ratio'],
                'val/l1_norm': sparsity_stats['avg_l1_norm'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['logging']['save_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sparsity_stats': sparsity_stats,
                'config': config
            }, save_path)
            print(f"✓ Saved best model to {save_path}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()
