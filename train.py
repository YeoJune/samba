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
from utils.experiment import ExperimentLogger, CheckpointManager, MetricsTracker


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
                l1_loss_fn, config, epoch, device, logger, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    total_l1_loss = 0
    total_main_acc = 0
    total_aux_acc = 0
    total_main_perplexity = 0
    total_aux_perplexity = 0
    
    use_amp = config['training'].get('use_amp', False)
    log_interval = config['logging'].get('log_interval', 10)
    pad_token_id = config['dataset'].get('pad_token_id', 50256)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            main_logits, aux_logits, all_layer_outputs = model(input_ids, targets)
            vocab_size = main_logits.size(-1)
            
            # Calculate losses
            main_loss = main_loss_fn(
                main_logits.reshape(-1, vocab_size), 
                targets.reshape(-1)
            )
            aux_loss, aux_metrics = aux_loss_fn(aux_logits, targets)
            l1_loss, l1_metrics = l1_loss_fn(all_layer_outputs)
            
            # Combined loss
            aux_weight = config['training']['aux_weight']
            l1_weight = config['training']['l1_weight']
            loss = main_loss + aux_weight * aux_loss + l1_weight * l1_loss
        
        # Calculate accuracies and perplexity
        with torch.no_grad():
            main_preds = main_logits.argmax(dim=-1)
            # Calculate accuracy only on non-padding tokens
            mask = (targets != pad_token_id)
            if mask.sum() > 0:
                correct = (main_preds == targets) & mask
                main_acc = (correct.sum().float() / mask.sum().float()).item()
            else:
                main_acc = 0.0
            main_perplexity = torch.exp(main_loss).item()
            aux_perplexity = torch.exp(aux_loss).item()
        aux_acc = aux_metrics['aux_accuracy']
        
        # Backward pass
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
        
        # Accumulate metrics
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_l1_loss += l1_loss.item()
        total_main_acc += main_acc
        total_aux_acc += aux_acc
        total_main_perplexity += main_perplexity
        total_aux_perplexity += aux_perplexity
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'main_acc': f"{main_acc:.3f}",
            'aux_acc': f"{aux_acc:.3f}",
            'main_ppl': f"{main_perplexity:.2f}",
            'aux_ppl': f"{aux_perplexity:.2f}",
            'sparsity': f"{l1_metrics['avg_near_zero_ratio']:.3f}"
        })
        
        # Log metrics
        if batch_idx % log_interval == 0:
            step = epoch * len(dataloader) + batch_idx
            logger.log_metrics({
                'total_loss': loss.item(),
                'main_loss': main_loss.item(),
                'aux_loss': aux_loss.item(),
                'l1_loss': l1_loss.item(),
                'main_accuracy': main_acc,
                'aux_accuracy': aux_acc,
                'main_perplexity': main_perplexity,
                'aux_perplexity': aux_perplexity,
                'sparsity': l1_metrics['avg_near_zero_ratio'],
            }, step=step, prefix='train')
    
    # Return average metrics
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'main_loss': total_main_loss / n,
        'aux_loss': total_aux_loss / n,
        'l1_loss': total_l1_loss / n,
        'main_acc': total_main_acc / n,
        'aux_acc': total_aux_acc / n,
        'main_ppl': total_main_perplexity / n,
        'aux_ppl': total_aux_perplexity / n
    }


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
    total_main_perplexity = 0
    total_aux_perplexity = 0
    
    use_amp = config['training'].get('use_amp', False)
    pad_token_id = config['dataset'].get('pad_token_id', 50256)
    
    for input_ids, targets in tqdm(dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
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
            loss = main_loss + aux_weight * aux_loss + l1_weight * l1_loss
        
        # Calculate accuracies and perplexity
        main_preds = main_logits.argmax(dim=-1)
        # Calculate accuracy only on non-padding tokens
        mask = (targets != pad_token_id)
        if mask.sum() > 0:
            correct = (main_preds == targets) & mask
            main_acc = (correct.sum().float() / mask.sum().float()).item()
        else:
            main_acc = 0.0
        main_perplexity = torch.exp(main_loss).item()
        aux_perplexity = torch.exp(aux_loss).item()
        aux_acc = aux_metrics['aux_accuracy']
        
        total_loss += loss.item()
        total_main_loss += main_loss.item()
        total_aux_loss += aux_loss.item()
        total_l1_loss += l1_loss.item()
        total_main_acc += main_acc
        total_aux_acc += aux_acc
        total_main_perplexity += main_perplexity
        total_aux_perplexity += aux_perplexity
    
    n = len(dataloader)
    sparsity_stats = model.get_sparsity_stats(all_layer_outputs)
    
    return {
        'loss': total_loss / n,
        'main_loss': total_main_loss / n,
        'aux_loss': total_aux_loss / n,
        'l1_loss': total_l1_loss / n,
        'main_acc': total_main_acc / n,
        'aux_acc': total_aux_acc / n,
        'main_ppl': total_main_perplexity / n,
        'aux_ppl': total_aux_perplexity / n,
        'sparsity': sparsity_stats['avg_near_zero_ratio'],
        'l1_norm': sparsity_stats['avg_l1_norm']
    }


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
    
    # Initialize experiment logger
    exp_name = config['logging'].get('exp_name', 'samba_exp')
    logger = ExperimentLogger(
        exp_name=exp_name,
        save_dir=config['logging'].get('save_dir', 'experiments'),
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging']['project_name'],
        config=config
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=logger.checkpoint_dir,
        max_keep=3,
        save_interval=1
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Initialize model
    logger.log("="*80)
    logger.log("Initializing model...")
    logger.log("="*80)
    
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
    
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.log(f"Readout mode: {model_config.get('readout_mode', 'post')}")
    
    # Load pretrained weights
    if config['pretrained']['use_pretrained'] and not args.no_pretrained:
        logger.log("="*80)
        logger.log("Loading pretrained weights...")
        logger.log("="*80)
        
        model = load_pretrained_samba(
            model,
            config['pretrained']['mamba_model'],
            debug=True
        )
        verify_samba_weights(model, config['pretrained']['mamba_model'])
        
        model.readout.decoder = load_pretrained_decoder(
            model.readout.decoder,
            pretrained_name=config['pretrained']['decoder_model'],
            target_vocab_size=config['model']['vocab_size'],
            debug=True
        )
        
        if config['pretrained']['freeze_backbone']:
            logger.log("Freezing Mamba backbone...")
            for layer in model.layers:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.log("✓ Backbone frozen")
    else:
        logger.log("⚠️ Training from scratch")
    
    # Initialize losses
    pad_token_id = config['dataset'].get('pad_token_id', 50256)
    main_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    aux_loss_fn = AuxLossWithMetrics(ignore_index=pad_token_id)
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
        logger.log("✓ AMP (Automatic Mixed Precision) enabled")
    
    # Get dataloaders
    logger.log("="*80)
    logger.log("Loading dataset...")
    logger.log("="*80)
    
    dataset_config = {
        **config['dataset'],
        'batch_size': config['training']['batch_size']
    }
    train_loader, val_loader = get_wikitext_dataloaders(dataset_config)
    
    logger.log(f"✓ Dataset: {config['dataset']['name']}")
    logger.log(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    logger.log("="*80)
    logger.log("Starting training...")
    logger.log("="*80)
    
    for epoch in range(config['training']['epochs']):
        logger.log(f"\n{'='*80}")
        logger.log(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.log(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, main_loss_fn, 
            aux_loss_fn, l1_loss_fn, config, epoch, device, logger, scaler
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, main_loss_fn, aux_loss_fn, 
            l1_loss_fn, config, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch metrics
        logger.log(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Main Acc: {train_metrics['main_acc']:.4f}, "
                  f"Aux Acc: {train_metrics['aux_acc']:.4f}, "
                  f"Main PPL: {train_metrics['main_ppl']:.2f}, "
                  f"Aux PPL: {train_metrics['aux_ppl']:.2f}")
        logger.log(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Main Acc: {val_metrics['main_acc']:.4f}, "
                  f"Aux Acc: {val_metrics['aux_acc']:.4f}, "
                  f"Main PPL: {val_metrics['main_ppl']:.2f}, "
                  f"Aux PPL: {val_metrics['aux_ppl']:.2f}, "
                  f"Sparsity: {val_metrics['sparsity']:.4f}")
        
        # Log to experiment logger
        logger.log_metrics({
            'epoch_loss': train_metrics['loss'],
            'epoch_main_acc': train_metrics['main_acc'],
            'epoch_aux_acc': train_metrics['aux_acc'],
            'epoch_main_ppl': train_metrics['main_ppl'],
            'epoch_aux_ppl': train_metrics['aux_ppl'],
        }, step=epoch, prefix='train')
        
        logger.log_metrics({
            **val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch, prefix='val')
        
        # Update metrics tracker
        metrics_tracker.update({
            'val_loss': val_metrics['loss'],
            'val_main_acc': val_metrics['main_acc'],
            'val_sparsity': val_metrics['sparsity']
        })
        
        # Save checkpoint
        is_best = checkpoint_manager.is_best(val_metrics['loss'], lower_is_better=True)
        if checkpoint_manager.should_save(epoch):
            checkpoint_manager.save(
                model, optimizer, epoch, val_metrics, is_best=is_best
            )
    
    # Save final results
    final_results = {
        'final_epoch': config['training']['epochs'],
        'best_val_loss': checkpoint_manager.best_metric,
        'metrics_summary': metrics_tracker.get_summary()
    }
    logger.save_final_results(final_results)
    
    logger.log("="*80)
    logger.log("Training completed!")
    logger.log("="*80)
    logger.finish()


if __name__ == '__main__':
    main()
