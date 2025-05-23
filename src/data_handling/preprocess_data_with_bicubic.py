import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse
import traceback # Import traceback for detailed error printing

# Import necessary classes from your repository
try:
    # We need upscale_image from bicubic.py
    from src.utils.bicubic import upscale_image # [bicubic.py is a user uploaded file]
except ImportError:
    print("Please ensure bicubic.py is in the PYTHONPATH or the same directory.") 
    exit(1)

def preprocess_images_bicubic(args):
    """
    Preprocesses images by generating LR versions and Bicubic-upscaled HR versions.
    Assumes input HR images are already at the target args.img_size.
    Ensures Bicubic upscaled output is clamped to [-1, 1].
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for initial tensor operations.") 
    print("Note: bicubic.upscale_image might use CPU due to OpenCV/NumPy operations.") 

    # 1. Create output directories
    path_hr_original = os.path.join(args.output_dir, 'hr_original')
    path_lr = os.path.join(args.output_dir, 'lr')
    path_hr_bicubic_upscaled = os.path.join(args.output_dir, 'hr_bicubic_upscaled') # Renamed

    os.makedirs(path_hr_original, exist_ok=True)
    os.makedirs(path_lr, exist_ok=True)
    os.makedirs(path_hr_bicubic_upscaled, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No images found in {args.input_dir}. Please check the path and file extensions.") 
        return
        
    print(f"Found {len(image_files)} images in {args.input_dir}. Processing one by one.") 

    for img_name in tqdm(image_files, desc="Preprocessing images"):
        img_path = os.path.join(args.input_dir, img_name)
        base_filename = os.path.splitext(img_name)[0]
        
        try:
            # --- Load original HR image ---
            image_pil = Image.open(img_path).convert("RGB")

            if image_pil.width != args.img_size or image_pil.height != args.img_size:
                print(f"Warning: Image {img_name} has dimensions {image_pil.size}, "
                      f"but args.img_size is ({args.img_size}, {args.img_size}). "
                      f"Ensure input HR images are pre-resized to the target img_size.") 
            
            original_hr_tensor_0_1 = TF.to_tensor(image_pil) # (C, H, W), [0,1]
            # Transform to [-1, 1] range, move to device
            original_hr_tensor = (original_hr_tensor_0_1 * 2.0 - 1.0).to(device) 

            # --- Create LR image ---
            low_res_h = args.img_size // args.downscale_factor
            low_res_w = args.img_size // args.downscale_factor

            if low_res_h == 0 or low_res_w == 0:
                print(f"Error: Calculated low_res dimension is zero for {img_name}. Skipping.") 
                continue

            # Create LR tensor on the specified device, range should be maintained around [-1,1]
            low_res_tensor = TF.resize(
                original_hr_tensor.clone(), 
                [low_res_h, low_res_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            ) 
            # Explicitly clamp LR tensor to ensure it's within [-1, 1] after resize
            low_res_tensor_clamped = torch.clamp(low_res_tensor, -1.0, 1.0)


            # --- Upscale LR image using bicubic.upscale_image ---
            # 1. Convert LR tensor from [-1,1] to [0,1] for upscale_image function
            low_res_tensor_0_1_for_upscale = (low_res_tensor_clamped.cpu() + 1.0) / 2.0 # Move to CPU for upscale_image

            # 2. Call upscale_image. It expects CHW tensor [0,1] and returns HWC tensor [0,1] (on CPU)
            hr_bicubic_upscaled_hwc_0_1 = upscale_image(
                image_source=low_res_tensor_0_1_for_upscale, # CHW, [0,1], CPU
                scale_factor=args.downscale_factor,
                save_image=False # We are saving tensors, not image files here
            )
            if hr_bicubic_upscaled_hwc_0_1 is None:
                print(f"Error: bicubic.upscale_image returned None for {img_name}. Skipping.") 
                continue
            
            # Ensure it's a tensor if it's not already (upscale_image can return ndarray or tensor)
            if not isinstance(hr_bicubic_upscaled_hwc_0_1, torch.Tensor):
                hr_bicubic_upscaled_hwc_0_1 = torch.from_numpy(hr_bicubic_upscaled_hwc_0_1).float()


            # 3. Permute HWC [0,1] tensor to CHW [0,1]
            hr_bicubic_upscaled_chw_0_1 = hr_bicubic_upscaled_hwc_0_1.permute(2, 0, 1)

            # 4. Convert CHW [0,1] tensor back to [-1,1] range
            hr_bicubic_upscaled_chw_neg1_1 = hr_bicubic_upscaled_chw_0_1 * 2.0 - 1.0
            
            # 5. Clamp the final bicubic upscaled tensor to [-1,1]
            hr_bicubic_upscaled_final = torch.clamp(hr_bicubic_upscaled_chw_neg1_1, -1.0, 1.0)


            # --- Save processed tensors (all should be on CPU before saving) ---
            torch.save(original_hr_tensor.cpu(), os.path.join(path_hr_original, f"{base_filename}.pt"))
            torch.save(low_res_tensor_clamped.cpu(), os.path.join(path_lr, f"{base_filename}.pt"))
            torch.save(hr_bicubic_upscaled_final.cpu(), os.path.join(path_hr_bicubic_upscaled, f"{base_filename}.pt"))
                
        except Exception as e:
            print(f"Error processing image {img_name}: {e}") 
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images with Bicubic Upscaling for Diffusion Model Training")
    parser.add_argument('--input_dir', type=str, default='data/hr_images',
                        help='Directory containing original high-resolution images (assumed to be at target img_size).')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/bicubic_processed_train',
                        help='Directory to save preprocessed PyTorch tensors (.pt files).')
    
    # Removed --batch_size as processing is per image for bicubic

    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (height and width). Input HR images should already be this size.')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels (e.g., 3 for RGB).')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Factor to downscale HR to get LR, and to upscale LR with bicubic.')

    parser.add_argument('--device', type=str, default='cuda:0', # Still used for initial HR load and LR creation
                        help='Device for initial tensor operations (e.g., cuda:0, cpu).')

    args = parser.parse_args()
    
    print("--- Preprocessing Configuration (Bicubic Upscaling) ---") 
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}") 
    print("-------------------------------------------------------") 

    preprocess_images_bicubic(args)

    print("Bicubic preprocessing finished.") 
