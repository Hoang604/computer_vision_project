import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import traceback
import shutil

# --- Handle project-level imports ---
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    print(f"Adding parent directory to sys.path: {parent_dir}")
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from src.trainers.rectified_flow_trainer import RectifiedFlowTrainer, ImageGenerator
    from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
    from src.data_handling.dataset import ImageDatasetRRDB

except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

class ReflowInputDataset(Dataset):
    """
    A PyTorch Dataset for loading data required for the 'reflow' process.

    This dataset reads tensors (x0, x1, and lr) from the output directory of a
    previous Rectified Flow training stage. It assumes a directory structure where
    tensors are organized into 'x0', 'x1', and 'lr' subdirectories.

    Attributes:
        root_dir (str): The root directory containing the prepared data.
        tensor_files_basenames (list): A sorted list of tensor filenames.
    """
    def __init__(self, prepared_data_folder: str):
        """
        Initializes the ReflowInputDataset.

        Args:
            prepared_data_folder (str): Path to the directory containing the 'x0', 'x1',
                                        and 'lr' subdirectories.
        """
        self.root_dir = prepared_data_folder
        self.path_x0 = os.path.join(self.root_dir, 'x0')
        self.path_x1 = os.path.join(self.root_dir, 'x1')
        self.path_lr = os.path.join(self.root_dir, 'lr')

        if not all(os.path.isdir(p) for p in [self.path_x0, self.path_x1, self.path_lr]):
            raise FileNotFoundError(f"One or more required subdirectories ('x0', 'x1', 'lr') not found in '{self.root_dir}'")

        # Assume filenames are consistent across subdirectories
        self.tensor_files_basenames = sorted([f for f in os.listdir(self.path_x0) if f.endswith('.pt')])
        print(f"Found {len(self.tensor_files_basenames)} tensor files in {self.root_dir}")

    def __len__(self):
        return len(self.tensor_files_basenames)

    def __getitem__(self, idx):
        basename = self.tensor_files_basenames[idx]
        
        path_x0 = os.path.join(self.path_x0, basename)
        path_x1 = os.path.join(self.path_x1, basename)
        path_lr = os.path.join(self.path_lr, basename)

        x0 = torch.load(path_x0)
        x1_old = torch.load(path_x1) # x1 from the previous stage
        lr = torch.load(path_lr)

        return x0, x1_old, lr, basename

def prepare_data_for_flow_training(args):
    """
    Prepares paired data (x0, x1) for training a Rectified Flow model.

    This script supports two modes:
    - 'rectified_flow': Prepares data for the initial training stage. It pairs a
      random Gaussian noise tensor (x0) with a ground-truth high-resolution image (x1).
    - 'reflow': Prepares data for a subsequent training stage (reflowing). It uses a
      pre-trained flow model to generate a new, "straighter" target image (the new x1)
      by starting from the exact same noise (the old x0) used in the previous stage.
      This process aims to straighten the learned transportation path.

    Args:
        args (argparse.Namespace): An object containing the script's command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    output_path_x0 = os.path.join(args.output_dir, 'x0')
    output_path_x1 = os.path.join(args.output_dir, 'x1')
    output_path_lr = os.path.join(args.output_dir, 'lr')
    output_path_hr_rrdb = os.path.join(args.output_dir, 'hr_rrdb_upscaled')

    os.makedirs(output_path_x0, exist_ok=True)
    os.makedirs(output_path_x1, exist_ok=True)
    os.makedirs(output_path_lr, exist_ok=True)
    os.makedirs(output_path_hr_rrdb, exist_ok=True)

    # ===================================================================
    # Mode 'rectified_flow': (x0=noise, x1=real_image)
    # ===================================================================
    if args.mode == 'rectified_flow':
        print("Preparing data in 'rectified_flow' mode...")
        try:
            dataset = ImageDatasetRRDB(
                preprocessed_folder_path=args.input_dir,
                img_size=args.img_size,
                downscale_factor=args.downscale_factor,
                apply_hflip=False
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error initializing dataset: {e}")
            exit(1)
        
        for i in tqdm(range(len(dataset)), desc="Processing for Rectified Flow"):
            try:
                base_filename_pt = dataset.tensor_files_basenames[i]
                low_res_image, _, original_hr_image, _ = dataset[i]

                # x1 is the ground-truth image
                x1 = original_hr_image
                # x0 is random noise with the same shape as x1
                x0 = torch.randn_like(x1)

                torch.save(x0, os.path.join(output_path_x0, base_filename_pt))
                torch.save(x1, os.path.join(output_path_x1, base_filename_pt))
                
                # Copy auxiliary tensors (lr, hr_rrdb) for conditioning and other purposes
                shutil.copy(os.path.join(dataset.path_lr, base_filename_pt), os.path.join(output_path_lr, base_filename_pt))
                if os.path.exists(os.path.join(dataset.path_hr_rrdb_upscaled, base_filename_pt)):
                    shutil.copy(os.path.join(dataset.path_hr_rrdb_upscaled, base_filename_pt), os.path.join(output_path_hr_rrdb, base_filename_pt))

            except Exception as e:
                print(f"Error processing item at index {i}: {e}")
                traceback.print_exc()

    # ===================================================================
    # Mode 'reflow': (x0=old_noise, x1=generated_image_from_old_noise)
    # ===================================================================
    elif args.mode == 'reflow':
        print("Preparing data in 'reflow' mode...")
        if not args.flow_model_path or not args.context_extractor_path:
            print("Error: --flow_model_path and --context_extractor_path are required for 'reflow' mode.")
            exit(1)

        # 1. Load the necessary pre-trained models
        flow_model_stage1 = RectifiedFlowTrainer.load_rectified_flow_unet(args.flow_model_path, device=device)
        rrdb_config = {
            'in_nc': args.img_channels, 'out_nc': args.img_channels, 'num_feat': args.rrdb_num_feat,
            'num_block': args.rrdb_num_block, 'gc': args.rrdb_gc, 'sr_scale': args.downscale_factor
        }
        context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.context_extractor_path, model_config=rrdb_config, device=device
        )
        generator = ImageGenerator(img_channels=args.img_channels, img_size=args.img_size, device=device)

        # 2. Use ReflowInputDataset to load data (x0, lr) from the previous stage
        print(f"Initializing ReflowInputDataset from: {args.input_dir}")
        try:
            dataset = ReflowInputDataset(prepared_data_folder=args.input_dir)
        except FileNotFoundError as e:
            print(f"Error initializing dataset: {e}")
            print("Please ensure the input directory points to a valid flow-prepared data folder.")
            exit(1)
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        for batch_data in tqdm(dataloader, desc="Processing for Reflow"):
            try:
                # 3. Get data from the new dataloader
                # We need the old noise (x0) and the low-res image (lr)
                x0_old_batch, _, low_res_batch, batch_basenames = batch_data
                
                x0_old_batch = x0_old_batch.to(device)
                low_res_batch = low_res_batch.to(device)
                actual_batch_size = low_res_batch.shape[0]

                with torch.no_grad():
                    # Extract conditioning features from the LR image
                    _, condition_features = context_extractor(low_res_batch, get_fea=True)
                    
                    # 4. Generate the NEW x1 using the OLD x0 as the initial noise
                    x1_new_batch_gpu = generator.generate_images(
                        model=flow_model_stage1,
                        features=condition_features,
                        num_images=actual_batch_size,
                        num_inference_steps=args.inference_steps,
                        initial_noise=x0_old_batch
                    )

                # 5. Save the new pair: (old_x0, new_x1)
                for j in range(actual_batch_size):
                    basename = batch_basenames[j]
                    # Save the old noise (x0)
                    torch.save(x0_old_batch[j].cpu(), os.path.join(output_path_x0, basename))
                    # Save the newly generated image (x1)
                    torch.save(x1_new_batch_gpu[j].cpu(), os.path.join(output_path_x1, basename))
                    
                    # Copy the corresponding lr image for the next training stage
                    original_lr_path = os.path.join(dataset.path_lr, basename)
                    shutil.copy(original_lr_path, os.path.join(output_path_lr, basename))

            except Exception as e:
                print(f"Error processing batch: {e}")
                traceback.print_exc()

    else:
        print(f"Error: Unknown mode '{args.mode}'. Please choose 'rectified_flow' or 'reflow'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Rectified Flow training.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Mode 'rectified_flow': Path to RRDB processed data. Mode 'reflow': Path to the output of the previous flow stage.")
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/flow_data_train',
                        help='Directory to save the final flow-ready tensors (x0, x1, etc.).')
    parser.add_argument('--mode', type=str, required=True, choices=['rectified_flow', 'reflow'],
                        help="The mode for data preparation: 'rectified_flow' for Stage-1, 'reflow' for subsequent stages.")
    parser.add_argument('--img_size', type=int, default=160, help='Target HR image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor used.')
    parser.add_argument('--flow_model_path', type=str,
                        help="[Reflow-mode ONLY] Path to the pre-trained Rectified Flow model from the previous stage (.pth).")
    parser.add_argument('--context_extractor_path', type=str,
                        help="[Reflow-mode ONLY] Path to the RRDBNet weights used as a context/feature extractor.")
    parser.add_argument('--inference_steps', type=int, default=50,
                        help="[Reflow-mode ONLY] Number of ODE steps to generate x1 images.")
    parser.add_argument('--rrdb_num_feat', type=int, default=64, help='Number of features (nf) in the RRDBNet.')
    parser.add_argument('--rrdb_num_block', type=int, default=17, help='Number of RRDB blocks (nb) in the RRDBNet.')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='Growth channel (gc) in the RRDBNet.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of images to process in a batch.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run generation on (e.g., cuda:0, cpu).')
    
    args = parser.parse_args()

    print("--- Flow Data Preparation Configuration ---")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    print("-------------------------------------------------------------------")

    prepare_data_for_flow_training(args)

    print(f"Flow data preparation in '{args.mode}' mode finished.")
    print(f"Output data is ready in: {args.output_dir}")
