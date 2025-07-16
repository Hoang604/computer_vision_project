"""
Common utilities for training and evaluation scripts.
This module provides reusable functions to eliminate code duplication across scripts.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from typing import Union, Optional, Dict, Any, Tuple, List
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def setup_device(device_str: str) -> torch.device:
    """
    Setup and validate device for training/evaluation.
    
    Args:
        device_str: Device string (e.g., 'cuda:0', 'cuda:1', 'cpu')
    
    Returns:
        torch.device: Validated and available device
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {device_str} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    elif not device_str.startswith("cuda") and device_str != "cpu":
        print(f"Invalid device specified: {device_str}. Using best available device.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    return device


def create_dataset(dataset_type: str, 
                  folder_path: str,
                  img_size: int = 160,
                  downscale_factor: int = 4,
                  apply_hflip: bool = False,
                  use_preprocessed: bool = True,
                  **kwargs):
    """
    Create dataset based on type and parameters.
    
    Args:
        dataset_type: Type of dataset ('standard', 'rrdb', 'bicubic')
        folder_path: Path to data folder
        img_size: Target image size
        downscale_factor: Scale factor for downsampling
        apply_hflip: Whether to apply horizontal flip augmentation
        use_preprocessed: Whether to use preprocessed data if available
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Dataset instance
    """
    # Lazy import to avoid torchvision import issues
    from src.data_handling.dataset import ImageDataset, ImageDatasetRRDB, ImageDatasetBicubic
    
    print(f"Creating {dataset_type} dataset from: {folder_path}")
    
    if dataset_type == 'rrdb':
        dataset = ImageDatasetRRDB(
            preprocessed_folder_path=folder_path,
            img_size=img_size,
            downscale_factor=downscale_factor,
            apply_hflip=apply_hflip
        )
    elif dataset_type == 'bicubic':
        dataset = ImageDatasetBicubic(
            preprocessed_folder_path=folder_path,
            img_size=img_size,
            downscale_factor=downscale_factor,
            apply_hflip=apply_hflip
        )
    elif dataset_type == 'standard':
        # Get upscale function if provided
        upscale_function = kwargs.get('upscale_function')
        if upscale_function is None:
            from src.utils.bicubic import upscale_image
            upscale_function = upscale_image
            
        dataset = ImageDataset(
            folder_path=folder_path,
            img_size=img_size,
            downscale_factor=downscale_factor,
            upscale_function=upscale_function,
            use_preprocessed=use_preprocessed
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def create_dataloader(dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """
    Create DataLoader with common settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def add_common_args(parser: argparse.ArgumentParser, 
                   mode: str = 'train') -> argparse.ArgumentParser:
    """
    Add common arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance
        mode: Mode of operation ('train', 'eval', 'both')
    
    Returns:
        ArgumentParser with added common arguments
    """
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    
    # Data arguments
    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (height and width)')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Downscale factor for LR images')
    
    # U-Net architecture arguments
    parser.add_argument('--use_improved_unet', action='store_true',
                        help='Use improved U-Net with multi-level attention')
    parser.add_argument('--conditioning_strategy', type=str, default='mixed',
                        choices=['cross_attention', 'additive', 'concatenation', 'mixed'],
                        help='Conditioning strategy for U-Net')
    parser.add_argument('--attention_levels', type=int, nargs='+', default=[2, 3],
                        help='Levels to apply attention (0-indexed from input)')
    parser.add_argument('--attention_heads', type=int, default=8,
                        help='Number of attention heads')
    
    if mode in ['train', 'both']:
        # Training-specific arguments
        parser.add_argument('--epochs', type=int, default=60,
                            help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Training batch size')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                            help='Initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-2,
                            help='Weight decay for optimizer')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of DataLoader workers')
        parser.add_argument('--accumulation_steps', type=int, default=1,
                            help='Gradient accumulation steps')
        parser.add_argument('--apply_hflip', action='store_true',
                            help='Apply horizontal flip augmentation')
        
        # Scheduler arguments
        parser.add_argument('--scheduler_type', type=str, default='cosineannealinglr',
                            choices=['none', 'steplr', 'cosineannealinglr', 'exponentiallr'],
                            help='Type of learning rate scheduler')
        parser.add_argument('--lr_decay_epochs', type=int, default=30,
                            help='Epochs between LR decay (for StepLR)')
        parser.add_argument('--lr_decay_factor', type=float, default=0.5,
                            help='LR decay factor (for StepLR and ExponentialLR)')
        parser.add_argument('--cosine_t_max', type=int, default=None,
                            help='T_max for CosineAnnealingLR (defaults to epochs)')
        
        # Validation arguments
        parser.add_argument('--val_every_n_epochs', type=int, default=1,
                            help='Validation frequency in epochs')
        parser.add_argument('--val_batch_size', type=int, default=None,
                            help='Validation batch size (defaults to batch_size)')
        parser.add_argument('--apply_val_hflip', action='store_true',
                            help='Apply horizontal flip to validation data')
        
        # Output arguments
        parser.add_argument('--output_dir', type=str, default='checkpoints',
                            help='Directory to save model checkpoints')
        parser.add_argument('--save_every_n_epochs', type=int, default=10,
                            help='Save checkpoint every N epochs')
        
    if mode in ['eval', 'both']:
        # Evaluation-specific arguments
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Evaluation batch size')
        parser.add_argument('--max_samples', type=int, default=1000,
                            help='Maximum number of samples to evaluate')
        parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                            help='Output file for evaluation results')
    
    return parser


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str,
                    epochs: int,
                    lr_decay_epochs: int = 30,
                    lr_decay_factor: float = 0.5,
                    cosine_t_max: Optional[int] = None,
                    eta_min: float = 1e-6):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
        epochs: Total training epochs
        lr_decay_epochs: Epochs between decay (for StepLR)
        lr_decay_factor: Decay factor
        cosine_t_max: T_max for CosineAnnealingLR
        eta_min: Minimum learning rate for CosineAnnealingLR
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_decay_epochs, 
            gamma=lr_decay_factor
        )
    elif scheduler_type == 'cosineannealinglr':
        t_max = cosine_t_max if cosine_t_max is not None else epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min
        )
    elif scheduler_type == 'exponentiallr':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=lr_decay_factor
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def setup_directories(output_dir: str) -> None:
    """
    Create necessary directories for training/evaluation.
    
    Args:
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")


def print_config(args: argparse.Namespace, title: str = "Configuration") -> None:
    """
    Print configuration in a formatted way.
    
    Args:
        args: Parsed arguments
        title: Title for the configuration section
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
    
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    print("=" * 60)


def validate_paths(*paths: str) -> None:
    """
    Validate that all provided paths exist.
    
    Args:
        *paths: Variable number of path strings to validate
    
    Raises:
        FileNotFoundError: If any path doesn't exist
    """
    for path in paths:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")


def get_model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Extract model configuration from parsed arguments.
    
    Args:
        args: Parsed arguments
    
    Returns:
        Dictionary containing model configuration
    """
    config = {
        'img_size': args.img_size,
        'img_channels': args.img_channels,
        'downscale_factor': args.downscale_factor,
    }
    
    # Add U-Net specific configs
    if hasattr(args, 'use_improved_unet'):
        config['use_improved_unet'] = args.use_improved_unet
    if hasattr(args, 'conditioning_strategy'):
        config['conditioning_strategy'] = args.conditioning_strategy
    if hasattr(args, 'attention_levels'):
        config['attention_levels'] = args.attention_levels
    if hasattr(args, 'attention_heads'):
        config['attention_heads'] = args.attention_heads
    
    # Add model-specific configs if they exist
    model_specific_args = [
        'num_feat', 'num_block', 'gc', 'sr_scale',  # RRDBNet args
        'context_dim', 'time_emb_dim', 'num_layers', 'num_heads',  # UNet args
    ]
    
    for arg in model_specific_args:
        if hasattr(args, arg):
            config[arg] = getattr(args, arg)
    
    return config


def create_unet_model(use_improved: bool = False,
                     base_dim: int = 64,
                     out_dim: int = 3,
                     dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
                     cond_dim: int = 64,
                     rrdb_num_blocks: int = 8,
                     sr_scale: int = 4,
                     use_attention: bool = True,
                     conditioning_strategy: str = "mixed",
                     attention_levels: Optional[List[int]] = None,
                     attention_heads: int = 8,
                     **kwargs):
    """
    Create U-Net model (original or improved version).
    
    Args:
        use_improved: Whether to use the improved U-Net
        base_dim: Base dimension for the U-Net
        out_dim: Output dimensions
        dim_mults: Dimension multipliers for each level
        cond_dim: Conditioning dimension
        rrdb_num_blocks: Number of RRDB blocks
        sr_scale: Super-resolution scale factor
        use_attention: Whether to use attention
        conditioning_strategy: Strategy for conditioning ('cross_attention', 'additive', 'concatenation', 'mixed')
        attention_levels: List of levels to apply attention (for improved U-Net)
        attention_heads: Number of attention heads
        **kwargs: Additional arguments
    
    Returns:
        U-Net model instance
    """
    if attention_levels is None:
        attention_levels = [2, 3]
    
    if use_improved:
        from src.diffusion_modules.unet_improved import UnetImproved
        
        print(f"Creating improved U-Net with:")
        print(f"  - Conditioning strategy: {conditioning_strategy}")
        print(f"  - Attention levels: {attention_levels}")
        print(f"  - Attention heads: {attention_heads}")
        
        model = UnetImproved(
            base_dim=base_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            cond_dim=cond_dim,
            rrdb_num_blocks=rrdb_num_blocks,
            sr_scale=sr_scale,
            use_attention=use_attention,
            attention_levels=attention_levels,
            conditioning_strategy=conditioning_strategy,
            attention_heads=attention_heads,
            **kwargs
        )
    else:
        from src.diffusion_modules.unet import Unet
        
        print("Creating original U-Net")
        
        model = Unet(
            base_dim=base_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            cond_dim=cond_dim,
            rrdb_num_blocks=rrdb_num_blocks,
            sr_scale=sr_scale,
            use_attention=use_attention,
            **kwargs
        )
    
    return model
