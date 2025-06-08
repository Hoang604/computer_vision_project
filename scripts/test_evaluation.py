"""
Test script để kiểm tra evaluation functionality với synthetic data
"""

import torch
import numpy as np
import os
import json
import sys

# Add the project root to Python path
sys.path.append('/home/nguyen_quoc_hieu/python/machine_learning/computer_vision_project')

from scripts.evaluate import ImageQualityEvaluator


def create_synthetic_images(batch_size=4, img_size=160):
    """Tạo synthetic images để test evaluation."""
    gt_images = torch.randn(batch_size, 3, img_size, img_size)
    gt_images = torch.tanh(gt_images)  # Đưa về range [-1, 1]
    
    pred_images = []
    
    pred_images.append(gt_images.clone())
    
    # Slightly noisy prediction
    noisy = gt_images + 0.1 * torch.randn_like(gt_images)
    pred_images.append(torch.clamp(noisy, -1, 1))
    
    # More noisy prediction
    very_noisy = gt_images + 0.3 * torch.randn_like(gt_images)
    pred_images.append(torch.clamp(very_noisy, -1, 1))
    
    # Random prediction (should give worst scores)
    random_pred = torch.randn_like(gt_images)
    random_pred = torch.tanh(random_pred)
    pred_images.append(random_pred)
    
    return gt_images, pred_images


def test_evaluation_metrics():
    """Test các metrics evaluation."""
    print("Testing evaluation metrics with synthetic data...")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    evaluator = ImageQualityEvaluator(device=device)
    
    gt_images, pred_images = create_synthetic_images()
    
    test_names = [
        "Perfect Prediction",
        "Slightly Noisy",
        "Very Noisy", 
        "Random Prediction"
    ]
    
    print(f"Ground truth shape: {gt_images.shape}")
    print(f"Device: {device}")
    print("\n" + "="*80)
    print("EVALUATION METRICS TEST")
    print("="*80)
    
    # Print header
    header = f"{'Test Case':<20}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}{'MSE':>12}{'MAE':>12}"
    print(header)
    print("-" * len(header))
    
    for i, (pred, name) in enumerate(zip(pred_images, test_names)):
        try:
            metrics = evaluator.evaluate_batch(pred, gt_images)
            
            row = f"{name:<20}"
            row += f"{metrics['PSNR']:>10.2f}"
            row += f"{metrics['SSIM']:>10.4f}"
            row += f"{metrics['LPIPS']:>10.4f}"
            row += f"{metrics['MSE']:>12.6f}"
            row += f"{metrics['MAE']:>12.6f}"
            print(row)
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    
    # Test single image evaluation
    print("\nTesting single image evaluation...")
    single_gt = gt_images[0]  # (3, H, W)
    single_pred = pred_images[1][0]  # (3, H, W) - slightly noisy
    
    try:
        single_metrics = evaluator.evaluate_batch(single_pred, single_gt)
        print(f"Single image metrics: {single_metrics}")
    except Exception as e:
        print(f"Error in single image evaluation: {e}")
        import traceback
        traceback.print_exc()


def test_tensor_conversion():
    """Test tensor conversion functions."""
    print("\nTesting tensor conversion...")
    
    evaluator = ImageQualityEvaluator()
    
    batch_tensor = torch.randn(2, 3, 64, 64)
    batch_tensor = torch.tanh(batch_tensor)  # [-1, 1]
    
    batch_np = evaluator.tensor_to_numpy(batch_tensor)
    print(f"Batch tensor shape: {batch_tensor.shape} -> {batch_np.shape}")
    print(f"Batch tensor range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}] -> [{batch_np.min():.3f}, {batch_np.max():.3f}]")
    
    single_tensor = torch.randn(3, 64, 64)
    single_tensor = torch.tanh(single_tensor)  # [-1, 1]
    
    single_np = evaluator.tensor_to_numpy(single_tensor)
    print(f"Single tensor shape: {single_tensor.shape} -> {single_np.shape}")
    print(f"Single tensor range: [{single_tensor.min():.3f}, {single_tensor.max():.3f}] -> [{single_np.min():.3f}, {single_np.max():.3f}]")


def test_metrics_expected_behavior():
    """Test expected behavior của metrics."""
    print("\nTesting expected metric behaviors...")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    evaluator = ImageQualityEvaluator(device=device)
    
    img_size = 64  
    gt = torch.zeros(1, 3, img_size, img_size)  
    
    test_cases = [
        ("Identical", gt.clone()),
        ("White vs Black", torch.ones_like(gt)),
        ("Half intensity", gt + 0.5),
        ("Random noise", torch.randn_like(gt))
    ]
    
    print(f"{'Test Case':<15}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}")
    print("-" * 50)
    
    for name, pred in test_cases:
        try:
            pred = torch.clamp(pred, -1, 1)
            
            psnr_val = evaluator.calculate_psnr(pred, gt)
            ssim_val = evaluator.calculate_ssim(pred, gt)
            lpips_val = evaluator.calculate_lpips(pred, gt)
            
            print(f"{name:<15}{psnr_val:>10.2f}{ssim_val:>10.4f}{lpips_val:>10.4f}")
            
        except Exception as e:
            print(f"Error in {name}: {e}")


if __name__ == '__main__':
    print("Starting evaluation script testing...")
    
    # Test 1: Basic metrics functionality
    test_evaluation_metrics()
    
    # Test 2: Tensor conversion
    test_tensor_conversion()
    
    # Test 3: Expected behavior
    test_metrics_expected_behavior()
    
    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80)
