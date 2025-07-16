#!/usr/bin/env python3
"""
Example training script demonstrating the improved U-Net with multi-level attention
and alternative conditioning strategies.
"""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.setup import (
    setup_device, create_dataset, create_dataloader, add_common_args,
    create_scheduler, setup_directories, print_config, validate_paths,
    get_model_config_from_args, create_unet_model
)
from src.diffusion_modules.rrdb import RRDBNet
from src.trainers.diffusion_trainer import DiffusionTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Diffusion Model with Improved U-Net')
    
    # Add common arguments
    parser = add_common_args(parser, mode='train')
    
    # Model arguments
    parser.add_argument('--base_dim', type=int, default=64,
                        help='Base dimension for U-Net')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for U-Net levels')
    parser.add_argument('--cond_dim', type=int, default=64,
                        help='Conditioning dimension')
    parser.add_argument('--rrdb_num_blocks', type=int, default=8,
                        help='Number of RRDB blocks')
    
    # Paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data (optional)')
    parser.add_argument('--rrdb_checkpoint', type=str, required=True,
                        help='Path to pretrained RRDB checkpoint')
    
    # Training arguments
    parser.add_argument('--dataset_type', type=str, default='rrdb',
                        choices=['rrdb', 'bicubic', 'standard'],
                        help='Type of dataset to use')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print_config(args, "Improved U-Net Training Configuration")
    
    # Validate paths
    validate_paths(args.data_path, args.rrdb_checkpoint)
    if args.val_data_path:
        validate_paths(args.val_data_path)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Create datasets
    print("\n" + "="*60)
    print("CREATING DATASETS")
    print("="*60)
    
    train_dataset = create_dataset(
        dataset_type=args.dataset_type,
        folder_path=args.data_path,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor,
        apply_hflip=args.apply_hflip
    )
    
    val_dataset = None
    if args.val_data_path:
        val_dataset = create_dataset(
            dataset_type=args.dataset_type,
            folder_path=args.val_data_path,
            img_size=args.img_size,
            downscale_factor=args.downscale_factor,
            apply_hflip=args.apply_val_hflip
        )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_batch_size = args.val_batch_size if args.val_batch_size else args.batch_size
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    # Create models
    print("\n" + "="*60)
    print("CREATING MODELS")
    print("="*60)
    
    # Load pretrained RRDB
    print(f"Loading RRDB from: {args.rrdb_checkpoint}")
    rrdb_model = RRDBNet(
        in_channels=args.img_channels,
        out_channels=args.img_channels,
        rrdb_in_channels=args.cond_dim,
        number_of_rrdb_blocks=args.rrdb_num_blocks,
        sr_scale=args.downscale_factor
    )
    
    # Load RRDB checkpoint
    rrdb_checkpoint = torch.load(args.rrdb_checkpoint, map_location='cpu')
    if 'model_state_dict' in rrdb_checkpoint:
        rrdb_model.load_state_dict(rrdb_checkpoint['model_state_dict'])
    else:
        rrdb_model.load_state_dict(rrdb_checkpoint)
    
    rrdb_model = rrdb_model.to(device)
    rrdb_model.eval()  # Keep RRDB frozen
    
    # Create U-Net model
    unet_model = create_unet_model(
        use_improved=args.use_improved_unet,
        base_dim=args.base_dim,
        out_dim=args.img_channels,
        dim_mults=tuple(args.dim_mults),
        cond_dim=args.cond_dim,
        rrdb_num_blocks=args.rrdb_num_blocks,
        sr_scale=args.downscale_factor,
        use_attention=True,
        conditioning_strategy=args.conditioning_strategy,
        attention_levels=args.attention_levels,
        attention_heads=args.attention_heads
    )
    
    unet_model = unet_model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler_type,
        epochs=args.epochs,
        lr_decay_epochs=args.lr_decay_epochs,
        lr_decay_factor=args.lr_decay_factor,
        cosine_t_max=args.cosine_t_max
    )
    
    # Create trainer
    print("\n" + "="*60)
    print("CREATING TRAINER")
    print("="*60)
    
    trainer = DiffusionTrainer(
        unet_model=unet_model,
        rrdb_model=rrdb_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        accumulation_steps=args.accumulation_steps
    )
    
    # Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        output_dir=args.output_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        val_every_n_epochs=args.val_every_n_epochs
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import torch
    main()
