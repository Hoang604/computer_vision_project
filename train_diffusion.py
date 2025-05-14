import torch
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset
import os
import argparse
from diffusion_modules import Unet, RRDBNet # Assuming RRDBNet is correctly imported/defined elsewhere
from diffusion_trainer import DiffusionTrainer

def train_diffusion(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """

    context = args.context
    assert context in ['LR', 'HR'], "Context must be either 'LR' or 'HR'." # Ensure context is valid
    # --- Setup Device ---
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {args.device} requested but CUDA not available. Using CPU.") # Warning message
        device = torch.device("cpu")
    elif not args.device.startswith("cuda") and args.device != "cpu":
        print(f"Invalid device specified: {args.device}. Using CPU if CUDA not available, else cuda:0.") # Warning message
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}") # Log the device being used

    # --- Setup Dataset and DataLoader ---
    # Create the dataset with the specified folder
    folder_path = args.image_folder # Use image_folder from args
    train_dataset = ImageDataset(folder_path=folder_path, img_size=args.img_size, downscale_factor=args.downscale_factor)
    print(f"Loaded {len(train_dataset)} images from {folder_path}") # Log dataset size and path

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- Initialize DiffusionTrainer ---
    diffusion_trainer = DiffusionTrainer(
        timesteps=args.timesteps,
        device=device,
        mode=args.diffusion_mode, # Use diffusion_mode from args
    )

    # --- Initialize UNet Model ---
    unet_model = Unet(
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults), # Ensure dim_mults is a tuple
        use_attention=args.use_attention # Pass use_attention argument
    ).to(device)
    print(f"UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}, use_attention={args.use_attention}") # Log UNet config

    # --- Initialize Optimizer ---
    # Use standard AdamW optimizer
    optimizer = torch.optim.AdamW(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}") # Log optimizer config

    # --- Initialize Scheduler ---
    scheduler = None
    if args.scheduler_type != "none":
        if args.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_decay_epochs,
                gamma=args.lr_gamma
            )
            print(f"Using StepLR scheduler with step_size={args.lr_decay_epochs}, gamma={args.lr_gamma}") # Log StepLR config
        elif args.scheduler_type == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.t_max_epochs,
                eta_min=args.eta_min_lr
            )
            print(f"Using CosineAnnealingLR scheduler with T_max={args.t_max_epochs}, eta_min={args.eta_min_lr}") # Log CosineAnnealingLR config
        elif args.scheduler_type == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.lr_gamma # ExponentialLR uses gamma directly
            )
            print(f"Using ExponentialLR scheduler with gamma={args.lr_gamma}") # Log ExponentialLR config
        # Add other schedulers here as elif blocks
        # Example:
        # elif args.scheduler_type == "multisteplr":
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         optimizer,
        #         milestones=args.lr_milestones, # Assuming lr_milestones is a list of ints from args
        #         gamma=args.lr_gamma
        #     )
        #     print(f"Using MultiStepLR scheduler with milestones={args.lr_milestones}, gamma={args.lr_gamma}")
        else:
            print(f"Warning: Unknown scheduler type '{args.scheduler_type}'. No scheduler will be used.") # Warning for unknown scheduler
            args.scheduler_type = "none"

    # --- Load Checkpoint if available ---
    start_epoch = 0
    best_loss = float('inf')
    # Path to pre-trained model weights, can be None
    weights_path = args.weights_path if args.weights_path and os.path.exists(args.weights_path) else None
    if weights_path:
        print(f"Attempting to load checkpoint from: {weights_path}") # Log checkpoint loading attempt
        # Pass scheduler for loading its state if available in checkpoint
        start_epoch, best_loss = diffusion_trainer.load_checkpoint_for_resume(
            device=device, # Pass device
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_type != "none" else None, # Pass scheduler
            checkpoint_path=weights_path
        )
        print(f"Loaded checkpoint. Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}") # Log resume info
    else:
        print("No valid pre-trained weights path found or specified. Starting training from scratch.") # Log starting from scratch

    # --- Initialize the context_extractor RRDBNet ---
    # Ensure RRDBNet is defined or imported correctly
    # This part assumes RRDBNet is used as a context extractor as in the original script
    context_extractor_model = RRDBNet(
        in_channels=args.img_channels, # Number of input channels
        out_channels=args.img_channels, # Number of output channels
        rrdb_in_channels=args.rrdb_in_channels, # Number of features in RRDB
        number_of_rrdb_blocks=args.number_of_rrdb_blocks, # Number of RRDB blocks
        sr_scale=args.downscale_factor # sr_scale for RRDBNet, typically matches downscale_factor
    ).to(device)
    # Load pre-trained weights for context_extractor if provided
    if args.rrdb_weights_path and os.path.exists(args.rrdb_weights_path):
        try:
            # Assuming a simple state_dict load for RRDBNet for now
            # You might need a more sophisticated loading function similar to DiffusionTrainer.load_model_weights
            # or the one in BasicRRDBNetTrainer.load_model_for_evaluation
            rrdb_checkpoint = torch.load(args.rrdb_weights_path, map_location=device)
            if 'model_state_dict' in rrdb_checkpoint: # Common pattern for checkpoints
                context_extractor_model.load_state_dict(rrdb_checkpoint['model_state_dict'])
            elif 'state_dict' in rrdb_checkpoint and 'model' in rrdb_checkpoint['state_dict']: # Another common pattern
                 context_extractor_model.load_state_dict(rrdb_checkpoint['state_dict']['model'])
            else: # Assume it's a raw state_dict
                context_extractor_model.load_state_dict(rrdb_checkpoint)
            print(f"Loaded pre-trained weights for RRDBNet context extractor from: {args.rrdb_weights_path}") # Log RRDB weights loading
        except Exception as e:
            print(f"Error loading RRDBNet weights from {args.rrdb_weights_path}: {e}. Using randomly initialized RRDBNet.") # Log error
    else:
        print("No pre-trained RRDBNet weights path specified or found. Using randomly initialized RRDBNet for context extraction.") # Log random init

    context_extractor_model.eval() # Set context extractor to evaluation mode

    # --- Start Training ---
    print("\nStarting diffusion model training process...") # Log start of training
    try:
        diffusion_trainer.train(
            dataset=train_loader,
            model=unet_model,
            context_extractor=context_extractor_model, # Pass the initialized context extractor
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_type != "none" else None, # Pass the scheduler
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            start_epoch=start_epoch,
            best_loss=best_loss,
            context_selection_mode=context, # Pass context selection mode
            log_dir_param=args.continue_log_dir if args.continue_log_dir else None,
            checkpoint_dir_param=args.continue_checkpoint_dir if args.continue_checkpoint_dir else None,
            log_dir_base=args.base_log_dir,
            checkpoint_dir_base=args.base_checkpoint_dir
        )
    except Exception as train_error:
        # Catch potential errors during training
        print(f"\nERROR occurred during training: {train_error}") # Log training error
        print("This might be due to issues reading image files, CUDA memory, or shape mismatches.") # Helpful message
        raise # Re-raise the exception for debugging

# --- Script Entry Point ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Diffusion Model with LR Scheduler Support")
    # Dataset args
    parser.add_argument('--img_size', type=int, default=160, help='Target image size for HR')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--image_folder', type=str, default="/media/tuannl1/heavy_weight/data/cv_data/images160x160", help='Path to the image folder (HR images)')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor for LR images, also SR scale for RRDBNet')

    # Training args
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs') # Increased default
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device') # Adjusted default
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)') # Adjusted default
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')

    # Scheduler args
    parser.add_argument('--scheduler_type', type=str, default='cosineannealinglr', choices=['none', 'steplr', 'cosineannealinglr', 'exponentiallr'], help='Type of LR scheduler')
    parser.add_argument('--lr_decay_epochs', type=int, default=30, help='StepLR: decay LR every N epochs') # For StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='StepLR/ExponentialLR: LR decay factor') # For StepLR, ExponentialLR
    parser.add_argument('--t_max_epochs', type=int, default=100, help='CosineAnnealingLR: T_max in epochs') # For CosineAnnealingLR (usually total epochs)
    parser.add_argument('--eta_min_lr', type=float, default=1e-6, help='CosineAnnealingLR: minimum learning rate') # For CosineAnnealingLR
    # Example for MultiStepLR if you add it:
    # parser.add_argument('--lr_milestones', type=int, nargs='+', default=[50, 80], help='MultiStepLR: epoch milestones for LR decay')


    # Diffusion args
    parser.add_argument('--context', type=str, default='LR', choices=['LR', 'HR'], help='Context type for conditioning (LR or HR image features)')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--diffusion_mode', type=str, default='v_prediction', choices=['v_prediction', 'noise'], help='Diffusion model prediction mode')

    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Channel multipliers for UNet down/up blocks')
    parser.add_argument('--use_attention', action='store_true', help='Whether to use attention in the UNet mid block')


    # RRDBNet (Context Extractor) args
    parser.add_argument('--rrdb_in_channels', type=int, default=64, help='Number of input features for RRDBNet trunk (nf)')
    parser.add_argument('--number_of_rrdb_blocks', type=int, default=8, help='Number of RRDB blocks in RRDBNet trunk (nb)')
    parser.add_argument('--rrdb_weights_path', type=str, default=None, help='Path to pre-trained RRDBNet weights (optional)')


    # Logging/Saving args
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pre-trained UNet model weights to resume training')
    parser.add_argument('--base_log_dir', type=str, default='./cv_logs_diffusion', help='Base directory for TensorBoard logging')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./cv_checkpoints_diffusion', help='Base directory for saving model checkpoints')
    parser.add_argument('--continue_log_dir', type=str, default=None, help='Specific log directory to continue (resumes experiment)')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None, help='Specific checkpoint directory to continue (resumes experiment)')

    # Loading model args (verbose for checkpoint loading)
    parser.add_argument('--verbose_load', action='store_true', help='Print detailed information about weight loading from checkpoint')

    args = parser.parse_args()

    # --- Print Configuration ---
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"--- Diffusion Model Training Configuration ---") # Header for config
    print(f"Image Folder: {args.image_folder}") # Log image folder
    print(f"Image Size (HR): {args.img_size}x{args.img_size}") # Log image size
    print(f"Image Channels: {args.img_channels}") # Log image channels
    print(f"Downscale Factor / SR Scale: {args.downscale_factor}") # Log downscale factor
    print(f"Device: {args.device}") # Log device
    print(f"Epochs: {args.epochs}") # Log epochs
    print(f"Batch Size (per device): {args.batch_size}") # Log batch size
    print(f"Accumulation Steps: {args.accumulation_steps}") # Log accumulation steps
    print(f"Effective Batch Size: {effective_batch_size}") # Log effective batch size
    print(f"Initial Learning Rate: {args.learning_rate}") # Log LR
    print(f"Weight Decay: {args.weight_decay}") # Log weight decay
    print(f"Scheduler Type: {args.scheduler_type}") # Log scheduler type
    if args.scheduler_type == "steplr":
        print(f"  StepLR: Decay Epochs: {args.lr_decay_epochs}, Gamma: {args.lr_gamma}") # Log StepLR params
    elif args.scheduler_type == "cosineannealinglr":
        print(f"  CosineAnnealingLR: T_max Epochs: {args.t_max_epochs}, Eta_min LR: {args.eta_min_lr}") # Log CosineAnnealingLR params
    elif args.scheduler_type == "exponentiallr":
        print(f"  ExponentialLR: Gamma: {args.lr_gamma}") # Log ExponentialLR params
    print(f"Num Workers: {args.num_workers}") # Log num workers
    print(f"Context Type: {args.context}") # Log context type
    print(f"Diffusion Timesteps: {args.timesteps}") # Log timesteps
    print(f"Diffusion Mode: {args.diffusion_mode}") # Log diffusion mode
    print(f"UNet Base Dim: {args.unet_base_dim}") # Log UNet base dim
    print(f"UNet Dim Mults: {tuple(args.unet_dim_mults)}") # Log UNet dim mults
    print(f"UNet Use Attention: {args.use_attention}") # Log UNet attention
    print(f"RRDBNet nf: {args.rrdb_in_channels}, nb: {args.number_of_rrdb_blocks}") # Log RRDBNet config
    if args.rrdb_weights_path:
        print(f"RRDBNet Weights Path: {args.rrdb_weights_path}") # Log RRDB weights path
    if args.weights_path:
        print(f"UNet Weights Path (for resume): {args.weights_path}") # Log UNet weights path
    print(f"Base Log Dir: {args.base_log_dir}") # Log base log dir
    print(f"Base Checkpoint Dir: {args.base_checkpoint_dir}") # Log base checkpoint dir
    if args.continue_log_dir:
        print(f"Resuming Log Dir: {args.continue_log_dir}") # Log resume log dir
    if args.continue_checkpoint_dir:
        print(f"Resuming Checkpoint Dir: {args.continue_checkpoint_dir}") # Log resume checkpoint dir
    print(f"Verbose Checkpoint Loading: {args.verbose_load}") # Log verbose load
    print(f"-------------------------------------------") # Footer for config

    # Call the main function
    train_diffusion(args)
