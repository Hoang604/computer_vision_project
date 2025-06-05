import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from tqdm import tqdm
import pandas as pd

# --- Metric Calculation Functions ---

def calculate_mse(img1_np, img2_np):
    """Calculate Mean Squared Error (MSE) between two NumPy images."""
    # Ensure images are float type to prevent overflow during squaring
    img1_np = img1_np.astype(np.float64)
    img2_np = img2_np.astype(np.float64)
    return np.mean((img1_np - img2_np) ** 2)

def calculate_mae(img1_np, img2_np):
    """Calculate Mean Absolute Error (MAE) between two NumPy images."""
    # Ensure images are float type
    img1_np = img1_np.astype(np.float64)
    img2_np = img2_np.astype(np.float64)
    return np.mean(np.abs(img1_np - img2_np))

def preprocess_image_for_lpips(img_pil, device):
    """
    Convert PIL image to PyTorch Tensor, normalize to [-1, 1], and move to device.
    Input: PIL Image (H, W, C) or (H, W), pixel values [0, 255].
    Output: Tensor (1, C, H, W), pixel values [-1, 1].
    """
    # Convert to RGB if grayscale
    if img_pil.mode == 'L':
        img_pil = img_pil.convert('RGB')
    elif img_pil.mode == 'RGBA': # Remove alpha channel if present
        img_pil = img_pil.convert('RGB')

    img_tensor = TF.to_tensor(img_pil)  # Convert to Tensor (C, H, W), range [0, 1]
    img_tensor_norm = (img_tensor * 2.0) - 1.0  # Normalize to range [-1, 1]
    return img_tensor_norm.unsqueeze(0).to(device) # Add batch dimension and move to device

# --- Main Evaluation Function ---

def evaluate_model(generated_dir, gt_dir, output_csv=None, device_str='cuda'):
    """
    Evaluate the super-resolution model by comparing generated images with ground-truth images.

    Args:
        generated_dir (str): Directory containing the super-resolved images.
        gt_dir (str): Directory containing the ground-truth (high-resolution) images.
        output_csv (str, optional): Path to save the metric results as a CSV file.
        device_str (str): Device to run LPIPS on ('cuda' or 'cpu').
    """
    if not os.path.isdir(generated_dir):
        print(f"Error: Generated images directory not found: {generated_dir}")
        return
    if not os.path.isdir(gt_dir):
        print(f"Error: Ground-truth images directory not found: {gt_dir}")
        return

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LPIPS")

    # Initialize LPIPS model
    # Uses 'alex' network by default. Can also choose 'vgg'.
    try:
        loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
        loss_fn_lpips.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error initializing LPIPS model: {e}")
        print("Please ensure the lpips library is installed and you have an internet connection to download weights for the first time.")
        return

    generated_files = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    # Create a map of ground-truth filenames for easy lookup
    gt_file_map = {os.path.splitext(f)[0]: f for f in gt_files}

    metrics_results = {
        'filename': [],
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'mse': [],
        'mae': []
    }

    print(f"Found {len(generated_files)} images in '{generated_dir}'.")
    if not generated_files:
        print("No images to evaluate.")
        return

    for gen_filename_with_ext in tqdm(generated_files, desc="Evaluating images"):
        gen_filename_base = os.path.splitext(gen_filename_with_ext)[0]

        # Find the corresponding ground-truth file
        # Assumes that the filenames (excluding extension) of generated and GT images match.
        # E.g., generated_dir/image1.png and gt_dir/image1.png
        gt_filename_to_use = None
        if gen_filename_base in gt_file_map:
            gt_filename_to_use = gt_file_map[gen_filename_base]
        else:
            # Try to find common suffixes like _gen, _sr, ... in the generated filename
            found_gt = False
            common_suffixes_to_remove = ["_sr", "_gen", "_output", "_result", "_x2", "_x3", "_x4", "_x8"] # Add common suffixes
            for suffix in common_suffixes_to_remove:
                if gen_filename_base.endswith(suffix):
                    potential_gt_base = gen_filename_base[:-len(suffix)]
                    if potential_gt_base in gt_file_map:
                        gt_filename_to_use = gt_file_map[potential_gt_base]
                        found_gt = True
                        break
            if not found_gt:
                print(f"Warning: Corresponding ground-truth image not found for '{gen_filename_with_ext}' (base: '{gen_filename_base}'). Skipping.")
                continue

        gen_img_path = os.path.join(generated_dir, gen_filename_with_ext)
        gt_img_path = os.path.join(gt_dir, gt_filename_to_use) # Use the determined GT filename

        try:
            # Load images using Pillow
            img_gen_pil = Image.open(gen_img_path).convert('RGB')
            img_gt_pil = Image.open(gt_img_path).convert('RGB')

            # Ensure images have the same size
            if img_gen_pil.size != img_gt_pil.size:
                print(f"Warning: Image sizes do not match for '{gen_filename_with_ext}' ({img_gen_pil.size}) and '{gt_filename_to_use}' ({img_gt_pil.size}).")
                print("Resizing generated image to match ground-truth size for evaluation.")
                img_gen_pil = img_gen_pil.resize(img_gt_pil.size, Image.BICUBIC)

            # Convert to NumPy array, value range [0, 255], type uint8
            img_gen_np = np.array(img_gen_pil)
            img_gt_np = np.array(img_gt_pil)

            # 1. PSNR
            # data_range is the maximum possible pixel value.
            # For uint8 images, data_range = 255.
            # For float images in range [0,1], data_range = 1.
            # skimage.metrics.psnr defaults channel_axis=None, which averages over channels for color images.
            # To calculate per-channel and then average, set channel_axis=-1 or 2 (depending on HWC or CHW format)
            val_psnr = psnr(img_gt_np, img_gen_np, data_range=255)
            metrics_results['psnr'].append(val_psnr)

            # 2. SSIM
            # data_range is similar to PSNR.
            # multichannel=True is used for color images (RGB). (Older skimage versions used `multichannel`)
            # `channel_axis=-1` for HWC images.
            val_ssim = ssim(img_gt_np, img_gen_np, data_range=255, channel_axis=-1, win_size=7) # win_size might need adjustment
            metrics_results['ssim'].append(val_ssim)

            # 3. LPIPS
            # LPIPS requires PyTorch Tensors, normalized to [-1, 1]
            img_gen_lpips = preprocess_image_for_lpips(img_gen_pil, device)
            img_gt_lpips = preprocess_image_for_lpips(img_gt_pil, device)

            with torch.no_grad(): # No gradients needed during evaluation
                val_lpips = loss_fn_lpips(img_gen_lpips, img_gt_lpips).item()
            metrics_results['lpips'].append(val_lpips)

            # 4. MSE
            # Uses NumPy images [0, 255]
            val_mse = calculate_mse(img_gt_np, img_gen_np)
            metrics_results['mse'].append(val_mse)

            # 5. MAE
            val_mae = calculate_mae(img_gt_np, img_gen_np)
            metrics_results['mae'].append(val_mae)

            metrics_results['filename'].append(gen_filename_with_ext)

        except FileNotFoundError:
            print(f"Error: Image file not found '{gen_img_path}' or '{gt_img_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing image '{gen_filename_with_ext}': {e}")
            # Add NaN or skip to avoid corrupting average results
            metrics_results['psnr'].append(float('nan'))
            metrics_results['ssim'].append(float('nan'))
            metrics_results['lpips'].append(float('nan'))
            metrics_results['mse'].append(float('nan'))
            metrics_results['mae'].append(float('nan'))
            metrics_results['filename'].append(gen_filename_with_ext + " (Error)")
            continue

    # Calculate average metrics (ignoring NaNs)
    avg_metrics = {}
    if metrics_results['psnr']: # Check if there are any results
        avg_metrics['psnr'] = np.nanmean(metrics_results['psnr'])
        avg_metrics['ssim'] = np.nanmean(metrics_results['ssim'])
        avg_metrics['lpips'] = np.nanmean(metrics_results['lpips'])
        avg_metrics['mse'] = np.nanmean(metrics_results['mse'])
        avg_metrics['mae'] = np.nanmean(metrics_results['mae'])

        print("\n--- Average Evaluation Results ---")
        print(f"Processed images: {len(metrics_results['filename']) - metrics_results['psnr'].count(float('nan'))} / {len(generated_files)}")
        print(f"PSNR:  {avg_metrics['psnr']:.4f} dB")
        print(f"SSIM:  {avg_metrics['ssim']:.4f}")
        print(f"LPIPS: {avg_metrics['lpips']:.4f} (lower is better)")
        print(f"MSE:   {avg_metrics['mse']:.4f} (lower is better)")
        print(f"MAE:   {avg_metrics['mae']:.4f} (lower is better)")
        print("------------------------------------")

        # Save to CSV if requested
        if output_csv:
            try:
                df = pd.DataFrame(metrics_results)
                # Add average row to DataFrame
                avg_row_data = {'filename': 'AVERAGE'}
                for metric_name, avg_val in avg_metrics.items():
                    avg_row_data[metric_name] = avg_val
                # Convert avg_row to DataFrame before concatenating
                avg_df = pd.DataFrame([avg_row_data])
                df = pd.concat([df, avg_df], ignore_index=True)

                df.to_csv(output_csv, index=False, float_format='%.4f')
                print(f"Detailed results saved to: {output_csv}")
            except Exception as e:
                print(f"Error saving CSV file: {e}")
    else:
        print("No images were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image super-resolution models.")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Path to the directory containing super-resolved images generated by the model.")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Path to the directory containing ground-truth (original high-resolution) images.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional: Path to save detailed metric results to a CSV file.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run LPIPS calculations on ('cuda' or 'cpu').")

    args = parser.parse_args()

    evaluate_model(args.generated_dir, args.gt_dir, args.output_csv, args.device)