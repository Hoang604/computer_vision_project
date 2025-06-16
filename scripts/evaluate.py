import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm
import json
from typing import Dict
from torch.utils.data import DataLoader

import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your project modules
from src.data_handling.dataset import ImageDataset, ImageDatasetRRDB
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
from src.trainers.diffusion_trainer import DiffusionTrainer, ResidualGenerator

class ImageQualityEvaluator:
    """
    Class to evaluate image quality using various metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index Measure)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    """
    
    def __init__(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize LPIPS model
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor from [-1, 1] range to [0, 1] range and to numpy array.
        Input tensor shape: (C, H, W) or (B, C, H, W)
        Output numpy array shape: (H, W, C) or (B, H, W, C)
        """
        if tensor.dim() == 4:  # Batch dimension
            # (B, C, H, W) -> (B, H, W, C)
            tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.dim() == 3:  # Single image
            # (C, H, W) -> (H, W, C)
            tensor = tensor.permute(1, 2, 0)
        
        # Convert from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor.detach().cpu().numpy()
    
    def calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate PSNR between prediction and target images."""
        pred_np = self.tensor_to_numpy(pred)
        target_np = self.tensor_to_numpy(target)
        
        if pred_np.ndim == 4:  # Batch
            psnr_values = []
            for i in range(pred_np.shape[0]):
                psnr_val = psnr(target_np[i], pred_np[i], data_range=1.0)
                psnr_values.append(psnr_val)
            return np.mean(psnr_values)
        else:  # Single image
            return psnr(target_np, pred_np, data_range=1.0)
    
    def calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate SSIM between prediction and target images."""
        pred_np = self.tensor_to_numpy(pred)
        target_np = self.tensor_to_numpy(target)
        
        if pred_np.ndim == 4:  # Batch
            ssim_values = []
            for i in range(pred_np.shape[0]):
                if pred_np.shape[-1] == 3:  # RGB
                    ssim_val = ssim(target_np[i], pred_np[i], data_range=1.0, channel_axis=-1)
                else:  # Grayscale
                    ssim_val = ssim(target_np[i], pred_np[i], data_range=1.0)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:  # Single image
            if pred_np.shape[-1] == 3:  # RGB
                return ssim(target_np, pred_np, data_range=1.0, channel_axis=-1)
            else:  # Grayscale
                return ssim(target_np, pred_np, data_range=1.0)
    
    def calculate_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS between prediction and target images."""
        # LPIPS expects tensors in [-1, 1] range, which matches our tensor format
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        if pred.dim() == 3:  # Single image, add batch dimension
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        with torch.no_grad():
            lpips_val = self.lpips_fn(pred, target)
        
        return lpips_val.mean().item()
    
    def calculate_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate MSE between prediction and target images."""
        mse = F.mse_loss(pred, target)
        return mse.item()
    
    def calculate_mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate MAE between prediction and target images."""
        mae = F.l1_loss(pred, target)
        return mae.item()
    
    def evaluate_batch(self, pred_batch: torch.Tensor, target_batch: torch.Tensor) -> Dict[str, float]:
        """Evaluate a batch of images and return all metrics."""
        metrics = {}
        
        metrics['PSNR'] = self.calculate_psnr(pred_batch, target_batch)
        metrics['SSIM'] = self.calculate_ssim(pred_batch, target_batch)
        metrics['LPIPS'] = self.calculate_lpips(pred_batch, target_batch)
        metrics['MSE'] = self.calculate_mse(pred_batch, target_batch)
        metrics['MAE'] = self.calculate_mae(pred_batch, target_batch)
        
        return metrics


def evaluate_rrdb_model(args):
    """Evaluate standalone RRDBNet model."""
    print("Evaluating RRDBNet model...")
    
    model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=args.rrdb_model_path,
        device=args.device
    )
    
    # Load dataset
    dataset = ImageDataset(
        folder_path=args.test_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor
    )
    
    evaluator = ImageQualityEvaluator(device=args.device)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for consistent evaluation
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )
    
    all_metrics = []
    num_samples = min(len(dataset), args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating RRDB"):
            lr_img, _, hr_img, _ = batch
            
            # Add batch dimension and move to device
            lr_img = lr_img.to(args.device)
            hr_img = hr_img.to(args.device)
            
            pred_hr = model(lr_img)
            
            # Calculate metrics
            metrics = evaluator.evaluate_batch(pred_hr, hr_img)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics

def evaluate_diffusion_model_batched(args):
    """
    Evaluate diffusion model (refining RRDBNet output) using batch processing.

    This version is optimized to process multiple images in parallel, significantly
    speeding up the evaluation process on a GPU.

    Args:
        args: An object or dictionary containing configuration, including:
              - rrdb_context_model_path (str): Path to the RRDBNet model.
              - diffusion_model_path (str): Path to the U-Net model.
              - preprocessed_data_folder (str): Path to the dataset.
              - img_size (int): Image size.
              - downscale_factor (int): Downscaling factor.
              - max_samples (int): Maximum number of samples to evaluate.
              - num_inference_steps (int): Number of steps for diffusion generation.
              - batch_size (int): Number of images to process in each batch.
              - device (str): The device to run on ('cuda' or 'cpu').
    """
    print("Evaluating Diffusion model with batch processing...")

    # 1. --- Initialization ---
    device = args.device
    context_model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=args.rrdb_context_model_path,
        device=device
    ).eval()

    unet = DiffusionTrainer.load_diffusion_unet(
        args.diffusion_model_path,
        device=device
    ).eval()

    generator = ResidualGenerator(
        img_size=160,  # Ensure this matches your model's expected input size
        device=device,
        predict_mode='noise'
    )

    # Load dataset
    dataset = ImageDatasetRRDB(
        preprocessed_folder_path=args.preprocessed_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor,
        apply_hflip=False
    )
    # Use a subset of the dataset if max_samples is specified
    if args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))


    # NEW: Use DataLoader for efficient batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for consistent evaluation
        num_workers=4,  # Adjust based on your system
        pin_memory=True
    )

    evaluator = ImageQualityEvaluator(device=device)
    all_metrics = []

    print(f"Evaluating on {len(dataset)} samples with batch size {args.batch_size}...")

    # 2. --- Batch Evaluation Loop ---
    with torch.no_grad():
        # Iterate over batches from the DataLoader
        for batch in tqdm(dataloader, desc="Evaluating Diffusion in Batches"):
            lr_img, rrdb_upscaled, hr_img, _ = batch

            # Move the entire batch to the target device
            lr_img = lr_img.to(device)
            rrdb_upscaled = rrdb_upscaled.to(device)
            hr_img = hr_img.to(device)

            # Extract context features for the whole batch at once
            # This returns a list of tensors, e.g., [feats_level1, feats_level2]
            # where each tensor has shape (batch_size, C, H, W)
            context_features_batched = context_model(lr_img, get_fea=True)[1]

            # Re-structure the batched features into the format expected by
            # `generate_multiple_residuals`: a list of lists of features.
            current_batch_size = lr_img.shape[0]
            list_of_features = []
            for i in range(current_batch_size):
                # For each image in the batch, gather its corresponding feature slices
                features_for_one_image = [feat_level[i:i+1] for feat_level in context_features_batched]
                list_of_features.append(features_for_one_image)

            # Generate all residuals for the batch in parallel
            residuals_batch = generator.generate_multiple_residuals(
                model=unet,
                list_of_features=list_of_features,
                num_inference_steps=args.num_inference_steps,
            )

            # Refine the entire batch of images
            refined_hr_batch = rrdb_upscaled + residuals_batch

            # Calculate metrics for the processed batch
            metrics = evaluator.evaluate_batch(refined_hr_batch, hr_img)
            all_metrics.append(metrics)

    # 3. --- Aggregate and Return Results ---
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    else:
        print("Warning: No metrics were calculated.")

    return avg_metrics




def evaluate_bicubic_baseline(args):
    """Evaluate bicubic interpolation baseline."""
    print("Evaluating Bicubic baseline...")
    
    # Load dataset
    dataset = ImageDataset(
        folder_path=args.test_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor
    )
    
    evaluator = ImageQualityEvaluator(device=args.device)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    
    all_metrics = []
    num_samples = min(len(dataset), args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    for batch in tqdm(data_loader, desc="Evaluating Bicubic"):
        _, bicubic_up, hr_img, _ = batch

        # Move to device
        bicubic_up = bicubic_up.to(args.device)
        hr_img = hr_img.to(args.device)
        
        # Calculate metrics
        metrics = evaluator.evaluate_batch(bicubic_up, hr_img)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics


def save_results(results: Dict, output_file: str):
    """Save evaluation results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {}
        for metric_name, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable_results[model_name][metric_name] = float(value)
            else:
                serializable_results[model_name][metric_name] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {output_file}")


def print_results(results: Dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Get all metric names
    all_metrics = set()
    for model_metrics in results.values():
        all_metrics.update([k for k in model_metrics.keys() if not k.endswith('_std')])
    
    all_metrics = sorted(list(all_metrics))
    
    # Print header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        header += f"{metric:>12}"
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, metrics in results.items():
        row = f"{model_name:<20}"
        for metric in all_metrics:
            if metric in metrics:
                value = metrics[metric]
                if metric in ['PSNR']:
                    row += f"{value:>12.2f}"
                elif metric in ['SSIM']:
                    row += f"{value:>12.4f}"
                elif metric in ['LPIPS', 'MSE', 'MAE']:
                    row += f"{value:>12.6f}"
                else:
                    row += f"{value:>12.4f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate image super-resolution models')
    
    # Data arguments
    parser.add_argument('--test_data_folder', type=str, required=True,
                        help='Path to test data folder containing HR images')
    parser.add_argument('--preprocessed_data_folder', type=str,
                        help='Path to preprocessed data folder for diffusion evaluation')
    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Downscale factor for LR images')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    # Model evaluation flags
    parser.add_argument('--eval_bicubic', action='store_true',
                        help='Evaluate bicubic baseline')
    parser.add_argument('--eval_rrdb', action='store_true',
                        help='Evaluate RRDBNet model')
    parser.add_argument('--eval_diffusion', action='store_true',
                        help='Evaluate diffusion model')
    
    # RRDBNet arguments
    parser.add_argument('--rrdb_model_path', type=str, help='Path to RRDBNet model checkpoint')

    # Diffusion model arguments
    parser.add_argument('--diffusion_model_path', type=str, help='Path to diffusion model checkpoint')

    parser.add_argument('--rrdb_context_model_path', type=str, help='Path to RRDBNet context extractor checkpoint')
    
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps for sampling')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for evaluation')
    
    args = parser.parse_args()

    # print all args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*80)
    
    # Validate arguments
    if not (args.eval_bicubic or args.eval_rrdb or args.eval_diffusion):
        parser.error("At least one evaluation flag must be set: --eval_bicubic, --eval_rrdb, or --eval_diffusion")
    
    if args.eval_rrdb and not args.rrdb_model_path:
        parser.error("--rrdb_model_path is required when --eval_rrdb is set")
    
    if args.eval_diffusion:
        if not args.diffusion_model_path:
            parser.error("--diffusion_model_path is required when --eval_diffusion is set")
        if not args.rrdb_context_model_path:
            parser.error("--rrdb_context_model_path is required when --eval_diffusion is set")
        if not args.preprocessed_data_folder:
            parser.error("--preprocessed_data_folder is required when --eval_diffusion is set")
    
    # Run evaluations
    results = {}
    
    if args.eval_bicubic:
        try:
            results['Bicubic'] = evaluate_bicubic_baseline(args)
        except Exception as e:
            print(f"Error evaluating bicubic baseline: {e}")
            import traceback
            traceback.print_exc()
    
    if args.eval_rrdb:
        try:
            results['RRDBNet'] = evaluate_rrdb_model(args)
        except Exception as e:
            print(f"Error evaluating RRDBNet: {e}")
            import traceback
            traceback.print_exc()
    
    if args.eval_diffusion:
        try:
            results['Diffusion'] = evaluate_diffusion_model_batched(args)
        except Exception as e:
            print(f"Error evaluating diffusion model: {e}")
            import traceback
            traceback.print_exc()
    
    # Print and save results
    if results:
        print_results(results)
        save_results(results, args.output_file)
    else:
        print("No evaluation results obtained.")


if __name__ == '__main__':
    main()
