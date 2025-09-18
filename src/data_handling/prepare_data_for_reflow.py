import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import traceback

try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from src.trainers.rectified_flow_trainer import RectifiedFlowTrainer, ImageGenerator
    from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
    from src.data_handling.dataset import ImageDatasetRRDB
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

class ReflowGenerationDataset(Dataset):
    """A wrapper dataset to get necessary items for reflow data generation."""
    def __init__(self, rrdb_dataset: ImageDatasetRRDB):
        self.rrdb_dataset = rrdb_dataset
    def __len__(self): return len(self.rrdb_dataset)
    def __getitem__(self, idx):
        # Return None on error, which will be handled by the safe_collate in the main script
        sample = self.rrdb_dataset[idx]
        if sample is None:
            return None
        low_res, _, original_hr, _ = sample
        return low_res, original_hr

def save_chunk(chunk_data, chunk_idx, output_dir):
    """Saves a chunk of collected tensors to disk."""
    if not chunk_data["x0"]: # Skip if chunk is empty
        return
    
    # Stack lists of tensors into single large tensors
    x0_tensor = torch.stack(chunk_data["x0"])
    x1_tensor = torch.stack(chunk_data["x1"])
    lr_tensor = torch.stack(chunk_data["lr"])
    hr_original_tensor = torch.stack(chunk_data["hr_original"])

    # Save each large tensor to a file
    chunk_name = f"reflow_chunk_{chunk_idx:05d}"
    torch.save(x0_tensor, os.path.join(output_dir, 'x0', f"{chunk_name}_x0.pt"))
    torch.save(x1_tensor, os.path.join(output_dir, 'x1', f"{chunk_name}_x1.pt"))
    torch.save(lr_tensor, os.path.join(output_dir, 'lr', f"{chunk_name}_lr.pt"))
    torch.save(hr_original_tensor, os.path.join(output_dir, 'hr_original', f"{chunk_name}_hr.pt"))
    print(f"Saved chunk {chunk_idx} with {len(chunk_data['x0'])} samples.")


def prepare_data_for_reflow(args):
    """
    Prepares data for 2-RF training with an I/O-optimized chunking strategy.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    output_path_x0 = os.path.join(args.output_dir, 'x0')
    output_path_x1 = os.path.join(args.output_dir, 'x1')
    output_path_lr = os.path.join(args.output_dir, 'lr')
    output_path_hr = os.path.join(args.output_dir, 'hr_original') # For validation logging
    os.makedirs(output_path_x0, exist_ok=True)
    os.makedirs(output_path_x1, exist_ok=True)
    os.makedirs(output_path_lr, exist_ok=True)
    os.makedirs(output_path_hr, exist_ok=True)

    # Load pre-trained models
    flow_model_stage1 = RectifiedFlowTrainer.load_rectified_flow_unet(args.flow_model_path, device=device)
    rrdb_config = {
        'in_nc': args.img_channels, 'out_nc': args.img_channels,
        'num_feat': args.rrdb_num_feat, 'num_block': args.rrdb_num_block,
        'gc': args.rrdb_gc, 'sr_scale': args.downscale_factor
    }
    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(args.context_extractor_path, rrdb_config, device)
    generator = ImageGenerator(img_channels=args.img_channels, img_size=args.img_size, device=device)

    # Initialize dataset
    source_dataset_rrdb = ImageDatasetRRDB(args.input_dir, apply_hflip=False)
    generation_dataset = ReflowGenerationDataset(source_dataset_rrdb)
    dataloader = DataLoader(generation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Chunking logic
    chunk_data = {"lr": [], "x0": [], "x1": [], "hr_original": []}
    chunk_idx = 0

    for batch_data in tqdm(dataloader, desc="Preparing data for Reflow"):
        try:
            if batch_data is None or batch_data[0] is None or batch_data[0].shape[0] == 0:
                print("Warning: Skipping a corrupted or empty batch from dataloader.")
                continue

            low_res_batch, original_hr_batch = batch_data
            low_res_batch, original_hr_batch = low_res_batch.to(device), original_hr_batch.to(device)

            with torch.no_grad():
                _, condition_features = context_extractor(low_res_batch, get_fea=True)
                
                # --- Flexible Data Generation Loop ---
                # This loop generates N reflow samples for each original image
                for _ in range(args.reflow_samples_per_image):
                    # 1. Generate new random noise for this iteration
                    x0_new_batch = torch.randn_like(original_hr_batch)

                    # 2. Generate the new target image from noise
                    x1_prime_batch_gpu = generator.generate_images(
                        model=flow_model_stage1, features=condition_features,
                        num_images=low_res_batch.shape[0], num_inference_steps=args.inference_steps,
                        initial_noise=x0_new_batch
                    )

                    # 3. Accumulate results in CPU memory
                    chunk_data["lr"].extend(list(low_res_batch.cpu()))
                    chunk_data["hr_original"].extend(list(original_hr_batch.cpu()))
                    chunk_data["x0"].extend(list(x0_new_batch.cpu()))
                    chunk_data["x1"].extend(list(x1_prime_batch_gpu.cpu()))
                # --- End of Flexible Loop ---

            # Save chunk to disk when it's full
            if len(chunk_data["x0"]) >= args.chunk_size:
                save_chunk(chunk_data, chunk_idx, args.output_dir)
                chunk_idx += 1
                chunk_data = {"lr": [], "x0": [], "x1": [], "hr_original": []} # Reset

        except Exception as e:
            print(f"\nERROR processing a batch: {e}")
            traceback.print_exc()

    # Save the final, partially filled chunk
    save_chunk(chunk_data, chunk_idx, args.output_dir)

    print(f"\nReflow data preparation finished. Data saved in chunks in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I/O-Optimized Data Preparation for 2-RF (Reflow) Training.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to RRDB-preprocessed data.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the final chunked data.')
    parser.add_argument('--flow_model_path', type=str, required=True, help="Path to the trained 1-RF model checkpoint.")
    parser.add_argument('--context_extractor_path', type=str, required=True, help="Path to the RRDBNet context extractor weights.")
    parser.add_argument('--img_size', type=int, default=160, help='Target HR image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor of the images.')
    parser.add_argument('--inference_steps', type=int, default=50, help="ODE steps for image generation.")
    parser.add_argument('--batch_size', type=int, default=32, help='GPU batch size for generation.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Number of samples to bundle into a single file.')
    
    parser.add_argument('--reflow_samples_per_image', type=int, default=2, 
                        help='Số lần tạo mẫu reflow cho mỗi ảnh gốc. Mặc định là 2 để nhân đôi dữ liệu.')
    
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for generation.')
    parser.add_argument('--rrdb_num_feat', type=int, default=64, help='Number of features (nf) in RRDBNet.')
    parser.add_argument('--rrdb_num_block', type=int, default=17, help='Number of blocks (nb) in RRDBNet.')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='Growth channel (gc) in RRDBNet.')
    args = parser.parse_args()
    prepare_data_for_reflow(args)
