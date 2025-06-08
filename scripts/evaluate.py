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
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your project modules
from src.data_handling.dataset import ImageDataset, ImageDatasetRRDB
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
from src.trainers.diffusion_trainer import DiffusionTrainer
from src.diffusion_modules.unet import Unet


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
    
    # Load RRDBNet model
    rrdb_config = {
        'in_nc': 3, 'out_nc': 3, 
        'num_feat': args.rrdb_num_feat, 
        'num_block': args.rrdb_num_block, 
        'gc': args.rrdb_gc, 
        'sr_scale': args.downscale_factor
    }
    
    model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=args.rrdb_model_path,
        model_config=rrdb_config,
        device=args.device
    )
    
    # Load dataset
    dataset = ImageDataset(
        folder_path=args.test_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor
    )
    
    evaluator = ImageQualityEvaluator(device=args.device)
    
    all_metrics = []
    num_samples = min(len(dataset), args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating RRDB"):
            lr_img, _, hr_img, _ = dataset[i]
            
            # Add batch dimension and move to device
            lr_img = lr_img.unsqueeze(0).to(args.device)
            hr_img = hr_img.unsqueeze(0).to(args.device)
            
            # Generate prediction
            if args.predict_residual:
                # Model predicts residual, need bicubic upscaling
                from src.utils.bicubic import upscale_image
                lr_for_bicubic = (lr_img + 1.0) / 2.0  # Convert to [0,1] for bicubic
                bicubic_up = upscale_image(
                    lr_for_bicubic.squeeze(0),
                    scale_factor=args.downscale_factor,
                    save_image=False
                )
                if isinstance(bicubic_up, np.ndarray):
                    bicubic_up = torch.from_numpy(bicubic_up).float()
                bicubic_up = bicubic_up.permute(2, 0, 1).unsqueeze(0).to(args.device)
                bicubic_up = bicubic_up * 2.0 - 1.0  # Convert back to [-1,1]
                
                residual = model(lr_img)
                pred_hr = bicubic_up + residual
            else:
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


def evaluate_diffusion_model(args):
    """Evaluate diffusion model (refining RRDBNet output)."""
    print("Evaluating Diffusion model...")
    
    # Load context extractor RRDBNet
    context_config = {
        'in_nc': 3, 'out_nc': 3,
        'num_feat': args.rrdb_num_feat_context,
        'num_block': args.rrdb_num_block_context,
        'gc': args.rrdb_gc_context,
        'sr_scale': args.downscale_factor
    }
    
    context_model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=args.rrdb_context_model_path,
        model_config=context_config,
        device=args.device
    )
    
    # Load UNet
    unet = Unet(
        in_channels=3,
        out_channels=3,
        base_dim=args.unet_base_dim,
        dim_mults=args.unet_dim_mults,
        context_dim=sum([args.rrdb_num_feat_context * mult for mult in [1, 2, 4]]),  # Feature dimensions
        use_attention=args.use_attention
    ).to(args.device)
    
    # Load diffusion trainer for inference
    diffusion_trainer = DiffusionTrainer(
        model=unet,
        diffusion_mode=args.diffusion_mode,
        timesteps=args.timesteps,
        device=args.device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.diffusion_model_path, map_location=args.device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    # Load preprocessed dataset
    dataset = ImageDatasetRRDB(
        preprocessed_folder_path=args.preprocessed_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor,
        apply_hflip=False
    )
    
    evaluator = ImageQualityEvaluator(device=args.device)
    
    all_metrics = []
    num_samples = min(len(dataset), args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    context_model.eval()
    unet.eval()
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating Diffusion"):
            lr_img, rrdb_upscaled, hr_img, _ = dataset[i]
            
            # Add batch dimension and move to device
            lr_img = lr_img.unsqueeze(0).to(args.device)
            rrdb_upscaled = rrdb_upscaled.unsqueeze(0).to(args.device)
            hr_img = hr_img.unsqueeze(0).to(args.device)
            
            # Extract context features from LR image
            context_features = context_model(lr_img, get_fea=True)[1]
            
            # Sample from diffusion model
            refined_hr = diffusion_trainer.sample(
                shape=(1, 3, args.img_size, args.img_size),
                context=context_features,
                initial_image=rrdb_upscaled,
                num_inference_steps=args.num_inference_steps
            )
            
            # Calculate metrics
            metrics = evaluator.evaluate_batch(refined_hr, hr_img)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics


def evaluate_bicubic_baseline(args):
    """Evaluate bicubic interpolation baseline."""
    print("Evaluating Bicubic baseline...")
    
    from src.utils.bicubic import upscale_image
    
    # Load dataset
    dataset = ImageDataset(
        folder_path=args.test_data_folder,
        img_size=args.img_size,
        downscale_factor=args.downscale_factor
    )
    
    evaluator = ImageQualityEvaluator(device=args.device)
    
    all_metrics = []
    num_samples = min(len(dataset), args.max_samples)
    
    print(f"Evaluating on {num_samples} samples...")
    
    for i in tqdm(range(num_samples), desc="Evaluating Bicubic"):
        lr_img, bicubic_up, hr_img, _ = dataset[i]
        
        # The dataset already provides bicubic upscaled image
        # Add batch dimension for evaluation
        bicubic_up = bicubic_up.unsqueeze(0)
        hr_img = hr_img.unsqueeze(0)
        
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
    
    # Model evaluation flags
    parser.add_argument('--eval_bicubic', action='store_true',
                        help='Evaluate bicubic baseline')
    parser.add_argument('--eval_rrdb', action='store_true',
                        help='Evaluate RRDBNet model')
    parser.add_argument('--eval_diffusion', action='store_true',
                        help='Evaluate diffusion model')
    
    # RRDBNet arguments
    parser.add_argument('--rrdb_model_path', type=str, help='Path to RRDBNet model checkpoint')
    parser.add_argument('--rrdb_num_feat', type=int, default=64, help='Number of features for RRDBNet')
    parser.add_argument('--rrdb_num_block', type=int, default=17, help='Number of RRDB blocks')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='Growth channel for RRDBNet')
    parser.add_argument('--predict_residual', action='store_true',
                        help='Whether RRDBNet predicts residual')
    
    # Diffusion model arguments
    parser.add_argument('--diffusion_model_path', type=str, help='Path to diffusion model checkpoint')

    parser.add_argument('--rrdb_context_model_path', type=str, help='Path to RRDBNet context extractor checkpoint')

    parser.add_argument('--rrdb_num_feat_context', type=int, default=64, help='Number of features for context RRDBNet')

    parser.add_argument('--rrdb_num_block_context', type=int, default=17, help='Number of RRDB blocks for context extractor')
    
    parser.add_argument('--rrdb_gc_context', type=int, default=32, help='Growth channel for context RRDBNet')
    
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base dimension for UNet')
    
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Dimension multipliers for UNet')
    
    parser.add_argument('--use_attention', action='store_true', help='Use attention in UNet')
    
    parser.add_argument('--diffusion_mode', type=str, default='noise', choices=['noise', 'v_prediction'], help='Diffusion mode')
    
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps for sampling')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    
    args = parser.parse_args()
    
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
            results['Diffusion'] = evaluate_diffusion_model(args)
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
