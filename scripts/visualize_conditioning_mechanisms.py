#!/usr/bin/env python3
"""
Visualization script for understanding different conditioning mechanisms
in the improved U-Net architecture.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.setup import setup_device, create_unet_model
from src.diffusion_modules.rrdb import RRDBNet


def visualize_attention_maps(model: nn.Module, 
                           input_tensor: torch.Tensor,
                           time_step: torch.Tensor,
                           rrdb_features: List[torch.Tensor],
                           device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Extract and visualize attention maps from the model.
    """
    attention_maps = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'fn') and hasattr(module.fn, 'self_attn'):
                # This is a BasicTransformerBlock wrapped in Residual
                # We need to access the attention weights
                attention_maps[name] = output.detach()
        return hook
    
    # Register hooks for attention blocks
    hooks = []
    if hasattr(model, 'down_attentions'):
        for i, attn in enumerate(model.down_attentions):
            if attn is not None:
                hook = attn.register_forward_hook(attention_hook(f'down_attn_{i}'))
                hooks.append(hook)
    
    if hasattr(model, 'mid_attn') and model.mid_attn is not None:
        hook = model.mid_attn.register_forward_hook(attention_hook('mid_attn'))
        hooks.append(hook)
    
    if hasattr(model, 'up_attentions'):
        for i, attn in enumerate(model.up_attentions):
            if attn is not None:
                hook = attn.register_forward_hook(attention_hook(f'up_attn_{i}'))
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor, time_step, rrdb_features)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps


def visualize_feature_flow(model: nn.Module,
                          input_tensor: torch.Tensor,
                          time_step: torch.Tensor,
                          rrdb_features: List[torch.Tensor],
                          device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Visualize how features flow through different conditioning mechanisms.
    """
    feature_maps = {}
    
    def feature_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_maps[name] = output.detach()
        return hook
    
    hooks = []
    
    # Hook conditioning blocks
    if hasattr(model, 'down_conditioning_blocks'):
        for i, cond_block in enumerate(model.down_conditioning_blocks):
            if cond_block is not None:
                hook = cond_block.register_forward_hook(feature_hook(f'down_cond_{i}'))
                hooks.append(hook)
    
    if hasattr(model, 'up_conditioning_blocks'):
        for i, cond_block in enumerate(model.up_conditioning_blocks):
            if cond_block is not None:
                hook = cond_block.register_forward_hook(feature_hook(f'up_cond_{i}'))
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor, time_step, rrdb_features)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    feature_maps['output'] = output.detach()
    
    return feature_maps


def plot_attention_comparison(attention_results: Dict[str, Dict[str, torch.Tensor]], 
                            save_path: str = "attention_comparison.png"):
    """
    Plot comparison of attention maps across different configurations.
    """
    fig, axes = plt.subplots(len(attention_results), 4, figsize=(16, 4*len(attention_results)))
    if len(attention_results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (config_name, attention_maps) in enumerate(attention_results.items()):
        # Select a few key attention maps to display
        selected_maps = []
        map_names = []
        
        for name, attn_map in attention_maps.items():
            if len(selected_maps) < 4:  # Show up to 4 attention maps
                selected_maps.append(attn_map)
                map_names.append(name)
        
        for j, (attn_map, map_name) in enumerate(zip(selected_maps, map_names)):
            if j < 4:
                # Convert attention map to numpy and take mean across channels
                attn_np = attn_map[0].mean(dim=0).cpu().numpy()  # Take first batch, mean across channels
                
                # Reshape if needed (attention maps might be flattened)
                if len(attn_np.shape) == 1:
                    size = int(np.sqrt(attn_np.shape[0]))
                    attn_np = attn_np.reshape(size, size)
                
                im = axes[i, j].imshow(attn_np, cmap='viridis', aspect='auto')
                axes[i, j].set_title(f'{config_name}\n{map_name}')
                axes[i, j].axis('off')
                plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
        
        # Fill empty subplots
        for j in range(len(selected_maps), 4):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention comparison saved to: {save_path}")


def plot_feature_statistics(feature_results: Dict[str, Dict[str, torch.Tensor]], 
                          save_path: str = "feature_statistics.png"):
    """
    Plot statistics of feature maps across different conditioning strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    config_names = list(feature_results.keys())
    colors = sns.color_palette("husl", len(config_names))
    
    # Collect statistics
    stats = {config: {'means': [], 'stds': [], 'max_vals': [], 'min_vals': []} 
             for config in config_names}
    
    for config_name, feature_maps in feature_results.items():
        for feat_name, feat_map in feature_maps.items():
            if feat_map is not None and feat_map.numel() > 0:
                stats[config_name]['means'].append(feat_map.mean().item())
                stats[config_name]['stds'].append(feat_map.std().item())
                stats[config_name]['max_vals'].append(feat_map.max().item())
                stats[config_name]['min_vals'].append(feat_map.min().item())
    
    # Plot mean values
    axes[0, 0].set_title('Feature Means')
    for i, (config, color) in enumerate(zip(config_names, colors)):
        axes[0, 0].hist(stats[config]['means'], alpha=0.7, label=config, color=color, bins=20)
    axes[0, 0].set_xlabel('Mean Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Plot standard deviations
    axes[0, 1].set_title('Feature Standard Deviations')
    for i, (config, color) in enumerate(zip(config_names, colors)):
        axes[0, 1].hist(stats[config]['stds'], alpha=0.7, label=config, color=color, bins=20)
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Plot max values
    axes[1, 0].set_title('Feature Maximum Values')
    for i, (config, color) in enumerate(zip(config_names, colors)):
        axes[1, 0].hist(stats[config]['max_vals'], alpha=0.7, label=config, color=color, bins=20)
    axes[1, 0].set_xlabel('Maximum Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot min values
    axes[1, 1].set_title('Feature Minimum Values')
    for i, (config, color) in enumerate(zip(config_names, colors)):
        axes[1, 1].hist(stats[config]['min_vals'], alpha=0.7, label=config, color=color, bins=20)
    axes[1, 1].set_xlabel('Minimum Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature statistics saved to: {save_path}")


def create_architecture_diagram(configs: List[Dict], save_path: str = "architecture_diagram.png"):
    """
    Create a visual diagram showing the different architecture configurations.
    """
    fig, axes = plt.subplots(1, len(configs), figsize=(5*len(configs), 8))
    if len(configs) == 1:
        axes = [axes]
    
    for i, config in enumerate(configs):
        ax = axes[i]
        
        # Draw U-Net structure
        levels = len(config.get('dim_mults', [1, 2, 4, 8]))
        
        # Draw encoder
        for level in range(levels):
            y_pos = levels - level - 1
            width = 0.8 / (level + 1)
            
            # Draw encoder block
            rect = plt.Rectangle((0.1, y_pos), width, 0.8, 
                               facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            
            # Add attention if applicable
            if config.get('use_improved', False):
                attention_levels = config.get('attention_levels', [])
                if level in attention_levels:
                    # Draw attention block
                    attn_rect = plt.Rectangle((0.1 + width + 0.05, y_pos + 0.2), 
                                            0.2, 0.4, facecolor='red', edgecolor='black')
                    ax.add_patch(attn_rect)
                    ax.text(0.1 + width + 0.15, y_pos + 0.4, 'Attn', 
                           ha='center', va='center', fontsize=8, color='white')
            
            # Add conditioning info
            strategy = config.get('conditioning_strategy', 'none')
            if strategy != 'cross_attention' and strategy != 'none':
                cond_rect = plt.Rectangle((0.1 + width + 0.3, y_pos + 0.2), 
                                        0.2, 0.4, facecolor='green', edgecolor='black')
                ax.add_patch(cond_rect)
                ax.text(0.1 + width + 0.4, y_pos + 0.4, 'Cond', 
                       ha='center', va='center', fontsize=8, color='white')
        
        # Draw decoder (mirrored)
        for level in range(levels):
            y_pos = level
            width = 0.8 / (levels - level)
            
            # Draw decoder block
            rect = plt.Rectangle((0.6, y_pos), width, 0.8, 
                               facecolor='lightgreen', edgecolor='black')
            ax.add_patch(rect)
        
        # Add title
        config_name = config.get('name', f'Config {i+1}')
        ax.set_title(config_name, fontsize=12, fontweight='bold')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.5, levels + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to: {save_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize U-Net Conditioning Mechanisms')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--input_size', type=int, default=160,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for visualization')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define configurations to compare
    configs = [
        {
            "name": "Cross-Attention",
            "use_improved": True,
            "conditioning_strategy": "cross_attention",
            "attention_levels": [2, 3],
            "attention_heads": 8
        },
        {
            "name": "Additive",
            "use_improved": True,
            "conditioning_strategy": "additive",
            "attention_levels": [2, 3],
            "attention_heads": 8
        },
        {
            "name": "Mixed",
            "use_improved": True,
            "conditioning_strategy": "mixed",
            "attention_levels": [2, 3],
            "attention_heads": 8
        }
    ]
    
    # Create dummy inputs
    input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    time_step = torch.randint(0, 1000, (args.batch_size,), device=device)
    
    # Create dummy RRDB features
    rrdb_features = []
    cond_dim = 64
    num_blocks = 8
    
    # Cần tạo 9 features tổng cộng (8 blocks + 1)
    for i in range(num_blocks + 1):
        feat_h = args.input_size // 4
        feat_w = args.input_size // 4
        feat = torch.randn(args.batch_size, cond_dim, feat_h, feat_w, device=device)
        rrdb_features.append(feat)
    
    print("Generating visualizations...")
    
    # Create architecture diagram
    create_architecture_diagram(configs, 
                               os.path.join(args.output_dir, "architecture_diagram.png"))
    
    # Analyze each configuration
    attention_results = {}
    feature_results = {}
    
    for config in configs:
        config_name = config["name"]
        print(f"Analyzing {config_name}...")
        
        try:
            # Create model - exclude 'name' from config
            model_config = {k: v for k, v in config.items() if k != 'name'}
            model = create_unet_model(**model_config)
            model = model.to(device)
            model.eval()
            
            # Extract attention maps
            attention_maps = visualize_attention_maps(model, input_tensor, 
                                                    time_step, rrdb_features, device)
            attention_results[config_name] = attention_maps
            
            # Extract feature maps
            feature_maps = visualize_feature_flow(model, input_tensor, 
                                                time_step, rrdb_features, device)
            feature_results[config_name] = feature_maps
            
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            print(f"Error analyzing {config_name}: {e}")
    
    # Generate comparison plots
    if attention_results:
        plot_attention_comparison(attention_results, 
                                os.path.join(args.output_dir, "attention_comparison.png"))
    
    if feature_results:
        plot_feature_statistics(feature_results, 
                               os.path.join(args.output_dir, "feature_statistics.png"))
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()
