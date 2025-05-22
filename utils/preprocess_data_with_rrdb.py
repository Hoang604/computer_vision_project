import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse
import traceback # Import traceback for detailed error printing

try:
    # Assuming rrdb_trainer.py is in the parent directory or PYTHONPATH
    # Adjust the import path if your project structure is different
    import sys
    # Add the parent directory to sys.path to find rrdb_trainer
    # This is a common way to handle imports when scripts are in subdirectories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from rrdb_trainer import BasicRRDBNetTrainer
except ImportError:
    print("Please ensure rrdb_trainer.py is in the PYTHONPATH or the project's root directory.") 
    exit(1)

def preprocess_images_batched_single_rrdb(args):
    """
    Preprocesses images by generating LR versions, RRDB-upscaled HR versions,
    and extracting features from LR images, ALL using a SINGLE RRDBNet model.
    Uses batched inference.
    Assumes input HR images are already at the target args.img_size.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    # 1. Load the single pre-trained RRDBNet model
    # This model will be used for both upscaling and feature extraction.
    rrdb_model_config = {
        'in_nc': args.img_channels,
        'out_nc': args.img_channels, # Output channels for upscaling
        'num_feat': args.rrdb_num_feat,
        'num_block': args.rrdb_num_block,
        'gc': args.rrdb_gc,
        'sr_scale': args.downscale_factor # sr_scale for upscaling
    }
    try:
        rrdb_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path,
            model_config=rrdb_model_config,
            device=device
        )
    except Exception as e:
        print(f"Error loading the RRDBNet model: {e}") 
        print("Please check the RRDBNet model path, config, and the BasicRRDBNetTrainer implementation.") 
        exit(1)
    rrdb_model.eval() # Ensure the model is in evaluation mode
    print(f"Single RRDBNet model loaded from {args.rrdb_weights_path}") 
    print(f"This model will be used for both upscaling and feature extraction.") 

    # 2. Create output directories
    path_hr_original = os.path.join(args.output_dir, 'hr_original')
    path_lr = os.path.join(args.output_dir, 'lr')
    path_hr_rrdb_upscaled = os.path.join(args.output_dir, 'hr_rrdb_upscaled')
    path_lr_features = os.path.join(args.output_dir, 'lr_features') # Directory for features

    os.makedirs(path_hr_original, exist_ok=True)
    os.makedirs(path_lr, exist_ok=True)
    os.makedirs(path_hr_rrdb_upscaled, exist_ok=True)
    os.makedirs(path_lr_features, exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No images found in {args.input_dir}. Please check the path and file extensions.") 
        return

    print(f"Found {len(image_files)} images in {args.input_dir}. Processing in batches of {args.batch_size}.") 

    # Lists to hold data for the current batch
    current_batch_lr_tensors_gpu = []
    current_batch_hr_original_tensors_cpu = []
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
                original_hr_tensor_gpu.clone(),
                [low_res_h, low_res_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            ) # Stays on GPU
            low_res_tensor_gpu = low_res_tensor_gpu.clamp(-1.0, 1.0) # Ensure values are in the range [-1, 1]

            current_batch_lr_tensors_gpu.append(low_res_tensor_gpu)
            current_batch_hr_original_tensors_cpu.append(original_hr_tensor_gpu.cpu()) # Move to CPU for batching before save
            current_batch_basenames.append(base_filename_no_ext)

            # If batch is full or it's the last image, process the batch
            if len(current_batch_lr_tensors_gpu) == args.batch_size or \
               (i == len(image_files) - 1 and len(current_batch_lr_tensors_gpu) > 0):

                lr_batch_gpu = torch.stack(current_batch_lr_tensors_gpu) # (B, C, H_lr, W_lr)

                # --- Run RRDBNet for Upscaling and Feature Extraction ---
                with torch.no_grad():
                    # The model's forward with get_fea=True returns: out_img, feas_list
                    rrdb_hr_batch_gpu, context_features_list_batch_gpu = rrdb_model(lr_batch_gpu, get_fea=True)
                    # rrdb_hr_batch_gpu is (B, C, H_hr, W_hr) - the upscaled image
                    # context_features_list_batch_gpu is a list of tensors (features).
                    # Each tensor in the list has shape (B, Channels_feat, H_feat, W_feat).

                # --- Save processed tensors and features for this batch ---
                for j in range(rrdb_hr_batch_gpu.shape[0]): # Iterate through images in the batch
                    hr_original_to_save = current_batch_hr_original_tensors_cpu[j]
                    lr_to_save = current_batch_lr_tensors_gpu[j].cpu()
                    rrdb_hr_to_save = rrdb_hr_batch_gpu[j].cpu() # Upscaled HR
                    basename_to_save = current_batch_basenames[j]

                    # Save image tensors
                    torch.save(hr_original_to_save, os.path.join(path_hr_original, f"{basename_to_save}.pt"))
                    torch.save(lr_to_save, os.path.join(path_lr, f"{basename_to_save}.pt"))
                    torch.save(rrdb_hr_to_save, os.path.join(path_hr_rrdb_upscaled, f"{basename_to_save}.pt"))

                    # Extract and save features for the j-th image in the batch
                    single_image_features_cpu = [feat_tensor[j].cpu() for feat_tensor in context_features_list_batch_gpu]
                    torch.save(single_image_features_cpu, os.path.join(path_lr_features, f"{basename_to_save}.pt"))

                # Clear lists for the next batch
                current_batch_lr_tensors_gpu = []
                current_batch_hr_original_tensors_cpu = []
                current_batch_basenames = []

        except Exception as e:
            print(f"Error processing image {img_name} (index {i}): {e}") 
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images with a single RRDBNet for Upscaling and Context Feature Extraction (Batched)")
    parser.add_argument('--input_dir', type=str, default='/media/tuannl1/heavy_weight/data/cv_data/160/train/bicubic/hr_original_images',
                        help='Directory containing original high-resolution images (assumed to be at target img_size).')
    parser.add_argument('--output_dir', type=str, default='/media/tuannl1/heavy_weight/data/cv_data/160/train/rrdb',
                        help='Directory to save preprocessed PyTorch tensors (.pt files), including features.')

    # Common image parameters
    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (height and width). Input HR images should already be this size.')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels (e.g., 3 for RGB).')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Factor to downscale HR to get LR. This is also the sr_scale for the RRDBNet.')

    # Single RRDBNet configuration (used for both upscaling and feature extraction)
    parser.add_argument('--rrdb_weights_path', type=str, required=True, default='/home/hoangdv/cv_project/checkpoints_rrdb/rrdb_20250520_135000/rrdb_model_best.pth',
                        help='Path to the pre-trained RRDBNet weights (.pth file).')
    parser.add_argument('--rrdb_num_feat', type=int, default=64,
                        help='Number of features (nf) in the RRDBNet.')
    parser.add_argument('--rrdb_num_block', type=int, default=17,
                        help='Number of RRDB blocks (nb) in the RRDBNet.')
    parser.add_argument('--rrdb_gc', type=int, default=32,
                        help='Growth channel (gc) in the RRDBNet.')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of images to process in a batch for RRDBNet inference.')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to run RRDBNet on (e.g., cuda:0, cuda:1, cpu).')

    args = parser.parse_args()

    print("--- Preprocessing Configuration (Single RRDBNet for Upscaling & Features) ---") 
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}") 
    print("---------------------------------------------------------------------------") 

    preprocess_images_batched_single_rrdb(args) # Call the updated function

    print("Batched preprocessing with single RRDBNet (upscaling and feature extraction) finished.") 
