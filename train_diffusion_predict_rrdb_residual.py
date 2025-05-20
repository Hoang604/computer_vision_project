import torch
from torch.utils.data import DataLoader
from utils.dataset import ImageDatasetRRDB
import os
import argparse
from diffusion_modules import Unet
from diffusion_trainer import DiffusionTrainer

def train_diffusion(args):
    """
    Main function to set up and run the diffusion model training
    using preprocessed data.
    """

    context_mode_for_trainer = args.context
    assert context_mode_for_trainer in ['LR', 'HR'], "Context mode for trainer must be either 'LR' or 'HR'." 

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
    # Create the dataset using the preprocessed data folder
    print(f"Loading preprocessed data from: {args.preprocessed_data_folder}") 
    train_dataset = ImageDatasetRRDB(
        preprocessed_folder_path=args.preprocessed_data_folder,
        img_size=args.img_size, # Passed for potential verification within ImageDataset
        downscale_factor=args.downscale_factor, # Passed for potential verification
        apply_hflip=args.apply_hflip # Control data augmentation
    )
    print(f"Loaded {len(train_dataset)} samples for training.") # Log dataset size

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = None
    if args.val_preprocessed_data_folder:
        print(f"Loading validation data from: {args.val_preprocessed_data_folder}") 
        val_dataset = ImageDatasetRRDB(
            preprocessed_folder_path=args.val_preprocessed_data_folder,
            img_size=args.img_size,
            downscale_factor=args.downscale_factor,
            apply_hflip=args.apply_hflip # Use specific arg for validation hflip (usually False)
        )
        print(f"Loaded {len(val_dataset)} samples for validation.") 
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, # Use specific batch size for validation
            shuffle=False, # No need to shuffle validation data
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False # Usually False for validation to evaluate all samples
        )
    else:
        print("No validation data folder provided. Skipping validation.") 


    # --- Initialize DiffusionTrainer ---
    diffusion_trainer = DiffusionTrainer(
        timesteps=args.timesteps,
        device=device,
        mode=args.diffusion_mode, # 'v_prediction' or 'noise'
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
            args.scheduler_type = "none" # Ensure it's marked as none

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

    # --- Start Training ---
    print(f"\nStarting diffusion model training (mode: {args.diffusion_mode}, UNet conditioned on RRDBNet features from LR)...") # Log start of training
    try:
        diffusion_trainer.train(
            dataset=train_loader,
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_type.lower() != "none" else None,
            val_dataset=val_loader,
            val_every_n_epochs=args.val_every_n_epochs,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            start_epoch=start_epoch,
            best_loss=best_loss,
            log_dir_param=args.continue_log_dir,
            checkpoint_dir_param=args.continue_checkpoint_dir,
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
    parser = argparse.ArgumentParser(description="Train Diffusion Model with Preprocessed Data and LR Scheduler Support")
    
    # Dataset args
    parser.add_argument('--preprocessed_data_folder', type=str, 
                        default='/media/tuannl1/heavy_weight/data/cv_data/images160x160/processed_val/train',
                        help='Path to the folder containing preprocessed tensors (.pt files from preprocess_data_with_rrdb.py).')
    parser.add_argument('--val_preprocessed_data_folder', type=str, 
                        default='/media/tuannl1/heavy_weight/data/cv_data/images160x160/processed_val/validation', # New
                        help='Path to the folder containing preprocessed validation tensors (optional).')

    parser.add_argument('--img_size', type=int, default=160,
                        help='Target HR image size (used for verification in ImageDataset).')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='Number of image channels.')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='Downscale factor used during preprocessing (for verification and RRDBNet sr_scale).')
    parser.add_argument('--apply_hflip', action='store_true',
                        help='Apply horizontal flipping augmentation to the preprocessed data.')

    # Training args
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to train on (e.g., cuda:0, cuda:1, cpu).')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per device.')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps).')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial optimizer learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, # Adjusted default
                        help='Optimizer weight decay.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes.')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, # New
                        help='Frequency (in epochs) to run validation.')


    # Scheduler args
    parser.add_argument('--scheduler_type', type=str, default='cosineannealinglr',
                        choices=['none', 'steplr', 'cosineannealinglr', 'exponentiallr'],
                        help='Type of LR scheduler.')
    parser.add_argument('--lr_decay_epochs', type=int, default=30,
                        help='StepLR: decay LR every N epochs.')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='StepLR/ExponentialLR: LR decay factor.')
    parser.add_argument('--cosine_t_max_epochs', type=int, default=None, # Default to total epochs if None
                        help='CosineAnnealingLR: T_max in epochs (defaults to total epochs if None).')
    parser.add_argument('--eta_min_lr', type=float, default=1e-6, # Adjusted default
                        help='CosineAnnealingLR: minimum learning rate.')

    # Diffusion args
    parser.add_argument('--context', type=str, default='LR', choices=['LR', 'HR'],
                        help="Context mode for DiffusionTrainer (influences internal processing, not feature source for U-Net's condition).")
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps.')
    parser.add_argument('--diffusion_mode', type=str, default='noise', choices=['v_prediction', 'noise'],
                        help='Diffusion model prediction mode (v_prediction or noise).')

    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=64,
                        help='Base channel dimension for UNet.')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Channel multipliers for UNet down/up blocks.')
    parser.add_argument('--use_attention', action='store_true',
                        help='Whether to use attention in the UNet mid block.')

    # RRDBNet (Context Extractor) args
    parser.add_argument('--rrdb_weights_path_context_extractor', type=str, 
                        default='/home/hoangdv/cv_project/checkpoints_rrdb/rrdb_17_05_16/rrdb_model_best.pth',
                        help='Path to pre-trained RRDBNet weights used as the context extractor for U-Net.')
    parser.add_argument('--number_of_rrdb_blocks', type=int, default=17, # Must match the context extractor RRDBNet
                        help='Number of RRDB blocks in the context extractor RRDBNet.')
    parser.add_argument('--rrdb_num_feat', type=int, default=64, # Must match the context extractor RRDBNet
                        help='Number of features (nf) in the context extractor RRDBNet.')
    parser.add_argument('--rrdb_gc', type=int, default=32, # Must match the context extractor RRDBNet
                        help='Growth channel (gc) in the context extractor RRDBNet.')
    
    # Logging/Saving args
    parser.add_argument('--weights_path_unet', type=str, default=None,
                        help='Path to pre-trained UNet model weights to resume training.')
    parser.add_argument('--base_log_dir', type=str, default='./logs_diffusion_v2', # Changed default
                        help='Base directory for TensorBoard logging.')
    parser.add_argument('--base_checkpoint_dir', type=str, default='./checkpoints_diffusion_v2', # Changed default
                        help='Base directory for saving model checkpoints.')
    parser.add_argument('--continue_log_dir', type=str, default=None,
                        help='Specific log directory to continue (resumes experiment).')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None,
                        help='Specific checkpoint directory to continue (resumes experiment).')
    parser.add_argument('--verbose_load', action='store_true',
                        help='Print detailed information about weight loading from UNet checkpoint.')

    args = parser.parse_args()

    # --- Print Configuration ---
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"--- Diffusion Model Training Configuration (with Preprocessed Data) ---") # Header for config
    print(f"Preprocessed Data Folder: {args.preprocessed_data_folder}") # Log preprocessed data folder
    print(f"Image Size (HR): {args.img_size}x{args.img_size}")
    print(f"Image Channels: {args.img_channels}")
    print(f"Downscale Factor / SR Scale: {args.downscale_factor}")
    print(f"Apply Horizontal Flip Augmentation: {args.apply_hflip}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size (per device): {args.batch_size}")
    print(f"Accumulation Steps: {args.accumulation_steps}")
    print(f"Effective Batch Size: {effective_batch_size}")
    print(f"Initial Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Scheduler Type: {args.scheduler_type}")
    if args.scheduler_type.lower() == "steplr":
        print(f"  StepLR: Decay Epochs: {args.lr_decay_epochs}, Gamma: {args.lr_gamma}")
    elif args.scheduler_type.lower() == "cosineannealinglr":
        t_max_display = args.cosine_t_max_epochs if args.cosine_t_max_epochs is not None else args.epochs
        print(f"  CosineAnnealingLR: T_max Epochs: {t_max_display}, Eta_min LR: {args.eta_min_lr}")
    elif args.scheduler_type.lower() == "exponentiallr":
        print(f"  ExponentialLR: Gamma: {args.lr_gamma}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Diffusion Timesteps: {args.timesteps}")
    print(f"Diffusion Prediction Mode: {args.diffusion_mode}")
    print(f"UNet Base Dim: {args.unet_base_dim}")
    print(f"UNet Dim Mults: {tuple(args.unet_dim_mults)}")
    print(f"UNet Use Attention: {args.use_attention}")
    print(f"Context Extractor RRDBNet Weights: {args.rrdb_weights_path_context_extractor}")
    print(f"Context Extractor RRDBNet Config: nf={args.rrdb_num_feat}, nb={args.number_of_rrdb_blocks}, gc={args.rrdb_gc}")
    if args.weights_path_unet:
        print(f"UNet Weights Path (for resume): {args.weights_path_unet}")
    print(f"Base Log Dir: {args.base_log_dir}")
    print(f"Base Checkpoint Dir: {args.base_checkpoint_dir}")
    if args.continue_log_dir: print(f"Resuming Log Dir: {args.continue_log_dir}")
    if args.continue_checkpoint_dir: print(f"Resuming Checkpoint Dir: {args.continue_checkpoint_dir}")
    print(f"Verbose UNet Checkpoint Loading: {args.verbose_load}")
    print(f"--------------------------------------------------------------------") # Footer for config

    train_diffusion(args)
