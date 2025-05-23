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
    from rrdb_trainer import BasicRRDBNetTrainer
except ImportError:
    print("Please ensure rrdb_trainer.py is in the PYTHONPATH or the project's root directory.")
    exit(1)

def preprocess_images_batched_rrdb_no_features(args):
    """
    Preprocesses images by generating LR versions and RRDB-upscaled HR versions.
    DOES NOT save LR features. Features will be extracted on-the-fly during training.
    Uses batched inference for RRDBNet upscaling.
    Assumes input HR images are already at the target args.img_size.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the pre-trained RRDBNet model for upscaling
    rrdb_model_config = {
        'in_nc': args.img_channels,
        'out_nc': args.img_channels, 
        'num_feat': args.rrdb_num_feat,
        'num_block': args.rrdb_num_block,
        'gc': args.rrdb_gc,
        'sr_scale': args.downscale_factor
    }
    try:
        rrdb_upscaler_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path_for_upscaling,
            model_config=rrdb_model_config,
            device=device
        )
    except Exception as e:
        print(f"Error loading the RRDBNet upscaler model: {e}")
        print("Please check the RRDBNet upscaler model path, config, and the BasicRRDBNetTrainer implementation.")
        exit(1)
    # rrdb_upscaler_model.eval()   # load_model_for_evaluation put the model on eval mode by default
    print(f"RRDBNet upscaler model loaded from {args.rrdb_weights_path_for_upscaling}")
    print(f"This model will be used ONLY for upscaling to create hr_rrdb_upscaled.")

    # 2. Create output directories
    path_hr_original = os.path.join(args.output_dir, 'hr_original')
    path_lr = os.path.join(args.output_dir, 'lr')
    path_hr_rrdb_upscaled = os.path.join(args.output_dir, 'hr_rrdb_upscaled')

    os.makedirs(path_hr_original, exist_ok=True)
    os.makedirs(path_lr, exist_ok=True)
    os.makedirs(path_hr_rrdb_upscaled, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No images found in {args.input_dir}. Please check the path and file extensions.")
        return

    print(f"Found {len(image_files)} images in {args.input_dir}. Processing in batches of {args.batch_size}.")

    # Lists to hold data for the current batch
    current_batch_lr_tensors_gpu = []
    current_batch_hr_original_tensors_cpu = [] # Store HR original on CPU to save GPU memory
    current_batch_basenames = []

    for i, img_name in enumerate(tqdm(image_files, desc="Reading images and preparing LR")):
        img_path = os.path.join(args.input_dir, img_name)
        base_filename_no_ext = os.path.splitext(img_name)[0] # e.g., "image_001"

        try:
            # --- Load original HR image ---
            image_pil = Image.open(img_path).convert("RGB")

            if image_pil.width != args.img_size or image_pil.height != args.img_size:
                print(f"Warning: Image {img_name} has dimensions {image_pil.size}, "
                      f"but args.img_size is ({args.img_size}, {args.img_size}). "
                      f"Ensure input HR images are pre-resized to the target img_size.")

            original_hr_tensor_0_1 = TF.to_tensor(image_pil) # (C, H, W), [0,1]
            original_hr_tensor_gpu = (original_hr_tensor_0_1 * 2.0 - 1.0).to(device) # to GPU for LR creation

            # --- Create LR image ---
            low_res_h = args.img_size // args.downscale_factor
            low_res_w = args.img_size // args.downscale_factor

            if low_res_h == 0 or low_res_w == 0:
                print(f"Error: Calculated low_res dimension is zero for {img_name}. Skipping.")
                continue

            low_res_tensor_gpu = TF.resize(
                original_hr_tensor_gpu.clone(), # Use original_hr_tensor_gpu for LR creation
                [low_res_h, low_res_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            ) # Stays on GPU
            low_res_tensor_gpu = low_res_tensor_gpu.clamp(-1.0, 1.0) # Ensure values are in the range [-1, 1]

            current_batch_lr_tensors_gpu.append(low_res_tensor_gpu)
            current_batch_hr_original_tensors_cpu.append(original_hr_tensor_gpu.cpu()) # Move HR original to CPU
            current_batch_basenames.append(base_filename_no_ext)

            # If batch is full or it's the last image, process the batch
            if len(current_batch_lr_tensors_gpu) == args.batch_size or \
               (i == len(image_files) - 1 and len(current_batch_lr_tensors_gpu) > 0):

                lr_batch_gpu = torch.stack(current_batch_lr_tensors_gpu) # (B, C, H_lr, W_lr)

                # --- Run RRDBNet for Upscaling ---
                with torch.no_grad():
                    rrdb_hr_batch_gpu = rrdb_upscaler_model(lr_batch_gpu, get_fea=False)
                    # rrdb_hr_batch_gpu is (B, C, H_hr, W_hr) - the upscaled image

                # --- Save processed tensors for this batch ---
                for j in range(rrdb_hr_batch_gpu.shape[0]): # Iterate through images in the batch
                    hr_original_to_save = current_batch_hr_original_tensors_cpu[j] # Already on CPU
                    lr_to_save = current_batch_lr_tensors_gpu[j].cpu() # Move LR to CPU for saving
                    rrdb_hr_to_save = rrdb_hr_batch_gpu[j].cpu() # Move Upscaled HR to CPU for saving
                    basename_to_save = current_batch_basenames[j]

                    # Save image tensors
                    torch.save(hr_original_to_save, os.path.join(path_hr_original, f"{basename_to_save}.pt"))
                    torch.save(lr_to_save, os.path.join(path_lr, f"{basename_to_save}.pt"))
                    torch.save(rrdb_hr_to_save, os.path.join(path_hr_rrdb_upscaled, f"{basename_to_save}.pt"))

                # Clear lists for the next batch
                current_batch_lr_tensors_gpu = []
                current_batch_hr_original_tensors_cpu = []
                current_batch_basenames = []

        except Exception as e:
            print(f"Error processing image {img_name} (index {i}): {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images with a single RRDBNet for Upscaling ONLY (No Feature Saving)")
    parser.add_argument('--input_dir', type=str, default='/media/tuannl1/heavy_weight/data/cv_data/celeba160x160/test',
                        help='Directory containing original high-resolution images (assumed to be at target img_size).')
    parser.add_argument('--output_dir', type=str, default='/media/tuannl1/heavy_weight/data/cv_data/celeba160x160/test/rrdb',
                        help='Directory to save preprocessed PyTorch tensors (.pt files), excluding features.')

    # Common image parameters
    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (height and width). Input HR images should already be this size.')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels (e.g., 3 for RGB).')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Factor to downscale HR to get LR. This is also the sr_scale for the RRDBNet upscaler.')

    # RRDBNet configuration (used for upscaling ONLY)
    parser.add_argument('--rrdb_weights_path_for_upscaling', type=str, default='/home/hoangdv/cv_project/checkpoints_rrdb/rrdb_20250521-141800/rrdb_model_best.pth',
                        help='Path to the pre-trained RRDBNet weights (.pth file) used for upscaling LR to HR_RRDB.')
    parser.add_argument('--rrdb_num_feat', type=int, default=64,
                        help='Number of features (nf) in the RRDBNet upscaler.')
    parser.add_argument('--rrdb_num_block', type=int, default=17,
                        help='Number of RRDB blocks (nb) in the RRDBNet upscaler.')
    parser.add_argument('--rrdb_gc', type=int, default=32,
                        help='Growth channel (gc) in the RRDBNet upscaler.')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of images to process in a batch for RRDBNet upscaling.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run RRDBNet upscaler on (e.g., cuda:0, cuda:1, cpu).')

    args = parser.parse_args()

    print("--- Preprocessing Configuration (RRDBNet for Upscaling ONLY, No Feature Saving) ---")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    print("------------------------------------------------------------------------------------")

    preprocess_images_batched_rrdb_no_features(args)

    print("Batched preprocessing with RRDBNet (upscaling only, no feature saving) finished.")
