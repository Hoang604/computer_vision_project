import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse
import traceback

try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    print(f"Adding parent directory to sys.path: {parent_dir}")
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from src.utils.bicubic import upscale_image
except ImportError:
    print("Please ensure bicubic.py is in the PYTHONPATH or the project's root directory.")
    exit(1)

def preprocess_images_with_bicubic(args):
    """
    Preprocesses images by generating LR versions and Bicubic-upscaled HR versions.
    Uses bicubic interpolation for upscaling.
    Assumes input HR images are already at the target args.img_size.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    path_hr_original = os.path.join(args.output_dir, 'hr_original')
    path_lr = os.path.join(args.output_dir, 'lr')
    path_hr_bicubic_upscaled = os.path.join(args.output_dir, 'hr_bicubic_upscaled')

    os.makedirs(path_hr_original, exist_ok=True)
    os.makedirs(path_lr, exist_ok=True)
    os.makedirs(path_hr_bicubic_upscaled, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No images found in {args.input_dir}. Please check the path and file extensions.")
        return

    print(f"Found {len(image_files)} images in {args.input_dir}. Processing...")

    for i, img_name in enumerate(tqdm(image_files, desc="Processing images with bicubic upscaling")):
        img_path = os.path.join(args.input_dir, img_name)
        base_filename_no_ext = os.path.splitext(img_name)[0]  # e.g., "image_001"

        try:
            # --- Load original HR image ---
            image_pil = Image.open(img_path).convert("RGB")

            if image_pil.width != args.img_size or image_pil.height != args.img_size:
                print(f"Warning: Image {img_name} has dimensions {image_pil.size}, "
                      f"but args.img_size is ({args.img_size}, {args.img_size}). "
                      f"Resizing to target size.")
                image_pil = image_pil.resize((args.img_size, args.img_size), Image.BICUBIC)

            # Convert to tensor and normalize to [-1, 1]
            original_hr_tensor_0_1 = TF.to_tensor(image_pil)  # (C, H, W), [0,1]
            original_hr_tensor = original_hr_tensor_0_1 * 2.0 - 1.0  # [-1, 1]

            # --- Create LR image ---
            low_res_h = args.img_size // args.downscale_factor
            low_res_w = args.img_size // args.downscale_factor

            if low_res_h == 0 or low_res_w == 0:
                print(f"Error: Calculated low_res dimension is zero for {img_name}. Skipping.")
                continue

            low_res_tensor = TF.resize(
                original_hr_tensor.clone(),
                [low_res_h, low_res_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            )
            low_res_tensor = low_res_tensor.clamp(-1.0, 1.0)  # Ensure values are in [-1, 1]

            # --- Create Bicubic upscaled image ---
            # Convert LR from [-1,1] to [0,1] for bicubic upscale function
            low_res_tensor_0_1 = (low_res_tensor + 1.0) / 2.0

            # Use bicubic upscale function
            bicubic_upscaled_hwc_0_1 = upscale_image(
                image_source=low_res_tensor_0_1,  # Input in [0,1]
                scale_factor=args.downscale_factor,
                save_image=False
            )

            if bicubic_upscaled_hwc_0_1 is None:
                raise RuntimeError(f"The 'upscale_image' function returned None for image: {img_path}")

            if not isinstance(bicubic_upscaled_hwc_0_1, torch.Tensor):
                print(f"Warning: upscale_image returned {type(bicubic_upscaled_hwc_0_1)}. Converting to tensor.")
                import numpy as np
                if isinstance(bicubic_upscaled_hwc_0_1, np.ndarray):
                    bicubic_upscaled_hwc_0_1 = torch.from_numpy(bicubic_upscaled_hwc_0_1).float()
                else:
                    raise TypeError(f"upscale_image returned an unexpected type: {type(bicubic_upscaled_hwc_0_1)}")

            # Convert from HWC [0,1] to CHW [-1,1]
            bicubic_upscaled_hwc_neg1_1 = bicubic_upscaled_hwc_0_1 * 2.0 - 1.0
            bicubic_hr_tensor = bicubic_upscaled_hwc_neg1_1.permute(2, 0, 1).float()

            # Ensure the upscaled image matches the target dimensions
            if bicubic_hr_tensor.shape[1:] != original_hr_tensor.shape[1:]:
                bicubic_hr_tensor = TF.resize(
                    bicubic_hr_tensor,
                    [args.img_size, args.img_size],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True
                )

            # --- Save processed tensors ---
            torch.save(original_hr_tensor.cpu(), os.path.join(path_hr_original, f"{base_filename_no_ext}.pt"))
            torch.save(low_res_tensor.cpu(), os.path.join(path_lr, f"{base_filename_no_ext}.pt"))
            torch.save(bicubic_hr_tensor.cpu(), os.path.join(path_hr_bicubic_upscaled, f"{base_filename_no_ext}.pt"))

        except Exception as e:
            print(f"Error processing image {img_name} (index {i}): {e}")
            traceback.print_exc()
            continue

    print(f"Preprocessing completed. Processed {len(image_files)} images.")
    print(f"Output directories:")
    print(f"  - HR Original: {path_hr_original}")
    print(f"  - LR: {path_lr}")
    print(f"  - HR Bicubic Upscaled: {path_hr_bicubic_upscaled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images with Bicubic interpolation for upscaling")
    parser.add_argument('--input_dir', type=str, default='data/hr_images',
                        help='Directory containing original high-resolution images.')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/bicubic_processed_train',
                        help='Directory to save preprocessed PyTorch tensors (.pt files).')

    # Common image parameters
    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (height and width).')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Factor to downscale HR to get LR. This is also the scale factor for bicubic upscaling.')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for tensor operations (e.g., cuda:0, cuda:1, cpu).')

    args = parser.parse_args()

    print("--- Preprocessing Configuration (Bicubic Interpolation) ---")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    print("----------------------------------------------------------")

    preprocess_images_with_bicubic(args)

    print("Bicubic preprocessing finished.")
