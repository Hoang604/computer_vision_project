#!/usr/bin/env python3
"""
Analysis script to compare different U-Net architectures and conditioning strategies.
This script helps evaluate the performance of various configurations.
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, List, Any
import time
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.setup import setup_device, create_unet_model
from src.diffusion_modules.rrdb import RRDBNet


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory_usage(model: nn.Module, input_shape: tuple, device: torch.device) -> Dict[str, Any]:
    """Measure memory usage of a model."""
    model.eval()
    
    # Create dummy inputs
    batch_size, channels, height, width = input_shape
    x = torch.randn(batch_size, channels, height, width, device=device)
    time_step = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create dummy RRDB features for conditioning
    rrdb_features = []
    cond_dim = 64
    num_blocks = 8
    
    # Simulate RRDB feature extraction
    for i in range(num_blocks + 1):
        if i % 3 == 2:  # Every 3rd feature (as used in the model)
            feat_h = height // 4  # LR resolution
            feat_w = width // 4
            feat = torch.randn(batch_size, cond_dim, feat_h, feat_w, device=device)
            rrdb_features.append(feat)
    
    # Measure memory before forward pass
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    memory_before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(x, time_step, rrdb_features)
            memory_after = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
            memory_used = memory_after - memory_before
            success = True
            error_msg = None
        except Exception as e:
            memory_used = 0
            success = False
            error_msg = str(e)
            output = None
    
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return {
        'success': success,
        'error': error_msg,
        'memory_used_mb': memory_used / (1024 ** 2) if memory_used > 0 else 0,
        'output_shape': list(output.shape) if output is not None else None
    }


def measure_inference_time(model: nn.Module, input_shape: tuple, device: torch.device, num_runs: int = 10) -> Dict[str, float]:
    """Measure inference time of a model."""
    model.eval()
    
    # Create dummy inputs
    batch_size, channels, height, width = input_shape
    x = torch.randn(batch_size, channels, height, width, device=device)
    time_step = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create dummy RRDB features
    rrdb_features = []
    cond_dim = 64
    num_blocks = 8
    
    for i in range(num_blocks + 1):
        if i % 3 == 2:
            feat_h = height // 4
            feat_w = width // 4
            feat = torch.randn(batch_size, cond_dim, feat_h, feat_w, device=device)
            rrdb_features.append(feat)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            try:
                _ = model(x, time_step, rrdb_features)
            except:
                return {'avg_time': float('inf'), 'std_time': float('inf')}
    
    # Measure time
    times = []
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            try:
                _ = model(x, time_step, rrdb_features)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                times.append(end_time - start_time)
            except:
                times.append(float('inf'))
    
    times = [t for t in times if t != float('inf')]
    if not times:
        return {'avg_time': float('inf'), 'std_time': float('inf')}
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {'avg_time': avg_time, 'std_time': std_time}


def analyze_unet_configurations(device: torch.device, 
                               input_shapes: List[tuple] = None,
                               output_file: str = "unet_analysis.json") -> Dict[str, Any]:
    """
    Analyze different U-Net configurations.
    
    Args:
        device: Device to run analysis on
        input_shapes: List of input shapes to test
        output_file: Output file for results
    
    Returns:
        Analysis results dictionary
    """
    if input_shapes is None:
        input_shapes = [
            (1, 3, 160, 160),   # Standard training size
            (1, 3, 256, 256),   # Larger test size
            (4, 3, 160, 160),   # Batch processing
        ]
    
    # Configuration to test
    configurations = [
        # Original U-Net
        {
            "name": "Original U-Net (No Attention)",
            "config": {
                "use_improved": False,
                "use_attention": False,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Original U-Net (With Attention)",
            "config": {
                "use_improved": False,
                "use_attention": True,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        # Improved U-Net variants
        {
            "name": "Improved U-Net (Cross-Attention)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "cross_attention",
                "attention_levels": [2, 3],
                "attention_heads": 8,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Improved U-Net (Additive)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "additive",
                "attention_levels": [2, 3],
                "attention_heads": 8,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Improved U-Net (Concatenation)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "concatenation",
                "attention_levels": [2, 3],
                "attention_heads": 8,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Improved U-Net (Mixed)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "mixed",
                "attention_levels": [2, 3],
                "attention_heads": 8,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Improved U-Net (Multi-Level Attention)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "mixed",
                "attention_levels": [1, 2, 3],  # More attention levels
                "attention_heads": 8,
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        },
        {
            "name": "Improved U-Net (More Attention Heads)",
            "config": {
                "use_improved": True,
                "conditioning_strategy": "mixed",
                "attention_levels": [2, 3],
                "attention_heads": 16,  # More heads
                "base_dim": 64,
                "dim_mults": (1, 2, 4, 8)
            }
        }
    ]
    
    results = {
        "device": str(device),
        "input_shapes": input_shapes,
        "configurations": {}
    }
    
    print(f"Analyzing {len(configurations)} U-Net configurations...")
    print(f"Device: {device}")
    print(f"Input shapes: {input_shapes}")
    print("="*80)
    
    for i, config_info in enumerate(configurations):
        config_name = config_info["name"]
        config = config_info["config"]
        
        print(f"\n[{i+1}/{len(configurations)}] Analyzing: {config_name}")
        print("-" * 60)
        
        try:
            # Create model
            model = create_unet_model(**config)
            model = model.to(device)
            
            # Count parameters
            param_count = count_parameters(model)
            print(f"Parameters: {param_count:,}")
            
            # Initialize results for this configuration
            config_results = {
                "config": config,
                "parameters": param_count,
                "analysis_results": {}
            }
            
            # Test each input shape
            for shape_idx, input_shape in enumerate(input_shapes):
                shape_name = f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}x{input_shape[3]}"
                print(f"  Testing shape {shape_name}...")
                
                # Memory analysis
                memory_results = measure_memory_usage(model, input_shape, device)
                
                # Timing analysis (only if memory test passed)
                if memory_results['success']:
                    timing_results = measure_inference_time(model, input_shape, device)
                else:
                    timing_results = {'avg_time': float('inf'), 'std_time': float('inf')}
                
                config_results["analysis_results"][shape_name] = {
                    **memory_results,
                    **timing_results
                }
                
                if memory_results['success']:
                    print(f"    ✓ Memory: {memory_results['memory_used_mb']:.1f} MB")
                    print(f"    ✓ Time: {timing_results['avg_time']*1000:.1f}±{timing_results['std_time']*1000:.1f} ms")
                else:
                    print(f"    ✗ Failed: {memory_results['error']}")
            
            results["configurations"][config_name] = config_results
            
            # Clean up
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            print(f"  ✗ Failed to create model: {e}")
            results["configurations"][config_name] = {
                "config": config,
                "error": str(e),
                "parameters": 0,
                "analysis_results": {}
            }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*80)
    print(f"Analysis complete! Results saved to: {output_file}")
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    # Sort configurations by parameter count
    configs = []
    for name, data in results["configurations"].items():
        if "error" not in data:
            configs.append((name, data))
    
    configs.sort(key=lambda x: x[1]["parameters"])
    
    print(f"{'Configuration':<35} {'Parameters':<12} {'Status':<10} {'Avg Time (ms)':<15}")
    print("-" * 80)
    
    for name, data in configs:
        param_count = data["parameters"]
        
        # Get average timing across all successful shapes
        times = []
        success_count = 0
        total_shapes = len(results["input_shapes"])
        
        for shape_results in data["analysis_results"].values():
            if shape_results.get("success", False):
                success_count += 1
                if shape_results["avg_time"] != float('inf'):
                    times.append(shape_results["avg_time"] * 1000)  # Convert to ms
        
        if times:
            avg_time = sum(times) / len(times)
            time_str = f"{avg_time:.1f}"
        else:
            time_str = "Failed"
        
        status = f"{success_count}/{total_shapes}"
        
        print(f"{name:<35} {param_count:>12,} {status:<10} {time_str:<15}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze U-Net Configurations')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for analysis')
    parser.add_argument('--output_file', type=str, default='unet_analysis.json',
                        help='Output file for analysis results')
    parser.add_argument('--input_sizes', type=int, nargs='+', 
                        default=[160, 256],
                        help='Input sizes to test (will create square images)')
    parser.add_argument('--batch_sizes', type=int, nargs='+',
                        default=[1, 4],
                        help='Batch sizes to test')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create input shapes
    input_shapes = []
    for size in args.input_sizes:
        for batch_size in args.batch_sizes:
            input_shapes.append((batch_size, 3, size, size))
    
    # Run analysis
    results = analyze_unet_configurations(
        device=device,
        input_shapes=input_shapes,
        output_file=args.output_file
    )
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
