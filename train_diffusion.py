import torch
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset
import os
import argparse
from diffusion_modules import Unet
from diffusion_trainer import DiffusionTrainer
from rrdb_trainer import BasicRRDBNetTrainer 

def train_diffusion(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """

    context_mode_for_trainer = args.context_type
    assert context_mode_for_trainer in ['LR', 'HR'], "Context type must be either 'LR' or 'HR'." # Ensure context is valid
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
    train_dataset = ImageDataset(folder_path=args.image_folder, 
                                 img_size=args.img_size, 
                                 downscale_factor=args.downscale_factor)
    print(f"Loaded {len(train_dataset)} images from {args.image_folder}") # Log dataset size and path

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
        dim_mults=tuple(args.unet_dim_mults),
        use_attention=args.use_attention,
        cond_dim=args.rrdb_num_feat,
        rrdb_num_blocks=args.number_of_rrdb_blocks
    ).to(device)
    print(f"UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}, use_attention={args.use_attention}") # Log UNet config

    # --- Initialize Optimizer ---
    optimizer = torch.optim.AdamW(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}") # Log optimizer config

    # --- Initialize Scheduler ---
    scheduler = None
    if args.scheduler_type.lower() != "none":
        if args.scheduler_type.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_decay_epochs, # Note: StepLR steps per epoch here
                gamma=args.lr_gamma
            )
            print(f"Using StepLR scheduler with step_size_epochs={args.lr_decay_epochs}, gamma={args.lr_gamma}") # Log StepLR config
        elif args.scheduler_type.lower() == "cosineannealinglr":
            # T_max is typically total number of epochs for CosineAnnealingLR
            t_max_epochs_for_scheduler = args.cosine_t_max_epochs if args.cosine_t_max_epochs is not None else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max_epochs_for_scheduler,
                eta_min=args.eta_min_lr
            )
            print(f"Using CosineAnnealingLR scheduler with T_max_epochs={t_max_epochs_for_scheduler}, eta_min={args.eta_min_lr}") # Log CosineAnnealingLR config
        elif args.scheduler_type.lower() == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.lr_gamma
            )
            print(f"Using ExponentialLR scheduler with gamma={args.lr_gamma}") # Log ExponentialLR config
        else:
            print(f"Warning: Unknown scheduler type '{args.scheduler_type}'. No scheduler will be used.") # Warning for unknown scheduler
            args.scheduler_type = "none" 

    # --- Load Checkpoint if available ---
    start_epoch = 0
    best_loss = float('inf')
    weights_path_unet = args.weights_path_unet if args.weights_path_unet and os.path.exists(args.weights_path_unet) else None
    if weights_path_unet:
        print(f"Attempting to load UNet checkpoint from: {weights_path_unet}") # Log checkpoint loading attempt
        start_epoch, best_loss = DiffusionTrainer.load_checkpoint_for_resume(
            device=device,
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_type.lower() != "none" else None,
            checkpoint_path=weights_path_unet,
            verbose_load=args.verbose_load
        )
        print(f"Loaded UNet checkpoint. Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}") # Log resume info
    else:
        print("No valid pre-trained UNet weights path found or specified. Starting UNet training from scratch.") # Log starting from scratch

    # --- Initialize the context_extractor RRDBNet ---
    # This part assumes RRDBNet is used as a context extractor as in the original script
    model_config = {
        'in_nc': args.img_channels, # Number of input channels
        'out_nc': args.img_channels, # Number of output channels
        'num_feat': args.rrdb_num_feat, # Number of features in RRDBNet
        'num_block': args.number_of_rrdb_blocks, # Number of RRDB blocks
        'gc': args.rrdb_gc, # Growth channel in RRDBNet
        'sr_scale': args.downscale_factor # Downscale factor for LR images
    }
    try:
        context_extractor_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path, 
            model_config=model_config, 
            device=device)
    except Exception as e:
        print(f"Error initializing RRDBNet context extractor: {e}")
        print("Ensure --rrdb_weights_path is set correctly and model config matches.")
        raise

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
            context_selection_mode=context_mode_for_trainer,
            log_dir_param=args.continue_log_dir if args.continue_log_dir else None,
            checkpoint_dir_param=args.continue_checkpoint_dir if args.continue_checkpoint_dir else None,
            log_dir_base=args.base_log_dir,
            checkpoint_dir_base=args.base_checkpoint_dir
        )
    except Exception as train_error:
        print(f"\nERROR occurred during training: {train_error}") # Log training error
        import traceback
        traceback.print_exc()
        print("This might be due to issues like CUDA memory, shape mismatches, or data loading.") # Helpful message
        raise

# --- Script Entry Point ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Diffusion Model with LR Scheduler Support")
    # Dataset args
    parser.add_argument('--image_folder', type=str, 
                        default="/media/tuannl1/heavy_weight/data/cv_data/images160x160", 
                        help='Path to the image folder (HR images)')
    parser.add_argument('--img_size', type=int, default=160, 
                        help='Target image size for HR')
    parser.add_argument('--img_channels', type=int, default=3, 
                        help='Number of image channels')
    parser.add_argument('--downscale_factor', type=int, default=4, 
                        help='Downscale factor for LR images, also SR scale for RRDBNet')

    # Training args
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='Device to train on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size per device')
    parser.add_argument('--accumulation_steps', type=int, default=4, 
                        help='Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)') # Adjusted default
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Initial optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, 
                        help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                         help='DataLoader worker processes')

    # Scheduler args
    parser.add_argument('--scheduler_type', type=str, default='cosineannealinglr', 
                        choices=['none', 'steplr', 'cosineannealinglr', 'exponentiallr'], 
                        help='Type of LR scheduler')
    parser.add_argument('--lr_decay_epochs', type=int, default=30, 
                        help='StepLR: decay LR every N epochs')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
                        help='StepLR/ExponentialLR: LR decay factor')
    parser.add_argument('--cosine_t_max_epochs', type=int, default=None, 
                        help='CosineAnnealingLR: T_max in epochs')
    parser.add_argument('--eta_min_lr', type=float, default=1e-6, 
                        help='CosineAnnealingLR: minimum learning rate') # For CosineAnnealingLR


    # Diffusion args
    parser.add_argument('--context_type', type=str, default='LR', choices=['LR', 'HR'], 
                        help='Context type for conditioning (LR or HR image features)')
    parser.add_argument('--timesteps', type=int, default=1000, 
                        help='Number of diffusion timesteps')
    parser.add_argument('--diffusion_mode', type=str, default='noise', choices=['v_prediction', 'noise'],
                        help='Diffusion model prediction mode')

    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=64, 
                        help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], 
                        help='Channel multipliers for UNet down/up blocks')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Whether to use attention in the UNet mid block')


    # RRDBNet (Context Extractor) args
    parser.add_argument('--rrdb_weights_path', type=str, default=None, 
                        help='Path to pre-trained RRDBNet weights (optional)')
    parser.add_argument('--number_of_rrdb_blocks', type=int, default=8, 
                        help='Number of RRDB blocks in RRDBNet trunk (nb)')
    parser.add_argument('--rrdb_num_feat', type=int, default=64, 
                        help='Number of features (nf) in RRDBNet')
    parser.add_argument('--rrdb_gc', type=int, default=32, 
                        help='Growth channel (gc) in RRDBNet')
    

    # Logging/Saving args
    parser.add_argument('--weights_path_unet', type=str, default=None, 
                        help='Path to pre-trained UNet model weights to resume training')
    parser.add_argument('--base_log_dir', type=str, default='./cv_logs_diffusion', 
                        help='Base directory for TensorBoard logging')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./cv_checkpoints_diffusion', 
                        help='Base directory for saving model checkpoints')
    parser.add_argument('--continue_log_dir', type=str, default=None, 
                        help='Specific log directory to continue (resumes experiment)')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None, 
                        help='Specific checkpoint directory to continue (resumes experiment)')
    parser.add_argument('--verbose_load', action='store_true', 
                        help='Print detailed information about weight loading from checkpoint')

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
        print(f"  CosineAnnealingLR: T_max Epochs: {args.cosine_t_max_epochs}, Eta_min LR: {args.eta_min_lr}") # Log CosineAnnealingLR params
    elif args.scheduler_type == "exponentiallr":
        print(f"  ExponentialLR: Gamma: {args.lr_gamma}") # Log ExponentialLR params
    print(f"Num Workers: {args.num_workers}") # Log num workers
    print(f"Context Type: {args.context_type}") # Log context type
    print(f"Diffusion Timesteps: {args.timesteps}") # Log timesteps
    print(f"Diffusion Mode: {args.diffusion_mode}") # Log diffusion mode
    print(f"UNet Base Dim: {args.unet_base_dim}") # Log UNet base dim
    print(f"UNet Dim Mults: {tuple(args.unet_dim_mults)}") # Log UNet dim mults
    print(f"UNet Use Attention: {args.use_attention}") # Log UNet attention
    print(f"RRDBNet nf: {args.rrdb_num_feat}, nb: {args.number_of_rrdb_blocks}") # Log RRDBNet config
    if args.rrdb_weights_path:
        print(f"RRDBNet Weights Path: {args.rrdb_weights_path}") # Log RRDB weights path
    if args.weights_path:
        print(f"UNet Weights Path (for resume): {args.weights_path_unet}") # Log UNet weights path
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
