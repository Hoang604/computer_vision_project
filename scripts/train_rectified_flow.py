import torch
from torch.utils.data import DataLoader
import os
import argparse
import traceback

# Assuming these files are part of the project structure and can be imported
from src.diffusion_modules.unet import Unet 
from src.trainers.rectified_flow_trainer import RectifiedFlowTrainer
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer 
from src.data_handling.dataset import FlowDataset 

def train_flow(args):
    """
    Main function to set up and run the Rectified Flow training process.
    """
    # --- Setup Device ---
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    elif not args.device.startswith("cuda") and args.device != "cpu":
        print(f"Invalid device specified: {args.device}. Using CPU if CUDA not available, else cuda:0.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Setup Dataset and DataLoader ---
    print(f"Loading prepared flow data from: {args.flow_prepared_data_folder}")
    train_dataset = FlowDataset(
        prepared_data_folder=args.flow_prepared_data_folder,
        apply_hflip=args.apply_hflip
    )
    print(f"Loaded {len(train_dataset)} samples for training.")
       
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if args.val_flow_prepared_data_folder:
        print(f"Loading validation data from: {args.val_flow_prepared_data_folder}")
        val_dataset = FlowDataset(
            prepared_data_folder=args.val_flow_prepared_data_folder,
            apply_hflip=args.apply_val_hflip # Usually False for validation
        )
        print(f"Loaded {len(val_dataset)} samples for validation.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        print("No validation data folder provided. Skipping validation.")

    # --- Initialize RRDBNet Context Extractor ---
    # This model extracts features from LR images to condition the UNet.
    rrdb_context_config = {
        'in_nc': args.img_channels, 
        'out_nc': args.img_channels,
        'num_feat': args.rrdb_num_feat_context, # Crucial for UNet's cond_dim
        'num_block': args.rrdb_num_block_context,
        'gc': args.rrdb_gc_context, 
        'sr_scale': args.downscale_factor
    }
    try:
        if not args.rrdb_weights_path_context_extractor or not os.path.exists(args.rrdb_weights_path_context_extractor):
            raise FileNotFoundError(f"RRDBNet weights for context extractor not found at: {args.rrdb_weights_path_context_extractor}")
        
        context_extractor_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path_context_extractor,
            model_config=rrdb_context_config, 
            device=device
        )
        context_extractor_model.eval() # Set to evaluation mode
        print(f"RRDBNet context extractor loaded from {args.rrdb_weights_path_context_extractor}")
        print(f"Context extractor config: nf={args.rrdb_num_feat_context}, nb={args.rrdb_num_block_context}, gc={args.rrdb_gc_context}")
    except Exception as e:
        print(f"Error loading RRDBNet context extractor: {e}")
        print("Please check the path and configuration of the context extractor RRDBNet.")
        return

    # --- Initialize UNet Model ---
    # cond_dim must match rrdb_num_feat_context from the context extractor.
    unet_model = Unet(
        base_dim=args.unet_base_dim, 
        dim_mults=tuple(args.unet_dim_mults),
        use_attention=args.use_attention, 
        cond_dim=args.rrdb_num_feat_context,
        rrdb_num_blocks=args.rrdb_num_block_context
    ).to(device)
    print(f"UNet model initialized with base_dim={args.unet_base_dim}, cond_dim={args.rrdb_num_feat_context}")

    # --- Initialize RectifiedFlowTrainer ---
    flow_trainer = RectifiedFlowTrainer(device=device, mode=args.mode)

    # --- Initialize Optimizer ---
    optimizer = torch.optim.AdamW(
        unet_model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}")

    # --- Initialize Scheduler ---
    scheduler = None
    if args.scheduler_type.lower() != "none":
        if args.scheduler_type.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_gamma)
            print(f"Using StepLR scheduler with step_size_epochs={args.lr_decay_epochs}, gamma={args.lr_gamma}")
        elif args.scheduler_type.lower() == "cosineannealinglr":
            t_max_epochs_for_scheduler = args.cosine_t_max_epochs if args.cosine_t_max_epochs is not None else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_epochs_for_scheduler, eta_min=args.eta_min_lr)
            print(f"Using CosineAnnealingLR scheduler with T_max_epochs={t_max_epochs_for_scheduler}, eta_min={args.eta_min_lr}")
        else:
            print(f"Warning: Unknown scheduler type '{args.scheduler_type}'. No scheduler will be used.")
            args.scheduler_type = "none"
    
    # --- Load Checkpoint if available ---
    start_epoch = 0
    best_loss = float('inf')
    weights_path_unet_resume = args.weights_path_unet_resume if args.weights_path_unet_resume and os.path.exists(args.weights_path_unet_resume) else None
    if weights_path_unet_resume:
        print(f"Attempting to load UNet checkpoint from: {weights_path_unet_resume}")
        start_epoch, best_loss = RectifiedFlowTrainer.load_checkpoint_for_resume(
            device=device, 
            model=unet_model, 
            optimizer=optimizer, 
            scheduler=scheduler if args.scheduler_type.lower() != "none" else None,
            checkpoint_path=weights_path_unet_resume, 
            verbose_load=args.verbose_load
        )
        print(f"Loaded UNet checkpoint. Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")
    else:
        print("No valid pre-trained UNet weights path found or specified. Starting UNet training from scratch.")

    # --- Start Training ---
    print(f"\nStarting Rectified Flow training (mode: {args.mode})...")
    try:
        flow_trainer.train(
            train_dataset=train_loader,
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler_type.lower() != "none" else None,
            context_extractor=context_extractor_model,
            val_dataset=val_loader,
            pretrained_model_path=args.pretrained_model_path_for_reflow if args.mode == 'reflow' else None,
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
        print(f"\nERROR occurred during training: {train_error}")
        traceback.print_exc()
        print("This might be due to issues like CUDA memory, shape mismatches, or data loading.")
        raise

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Rectified Flow model for Image Super-Resolution")

    # --- Dataset Arguments ---
    parser.add_argument('--flow_prepared_data_folder', type=str, required=True, help='Path to the folder containing data processed by prepare_data_for_rectified.py (containing lr, x0, x1 subfolders).')
    parser.add_argument('--val_flow_prepared_data_folder', type=str, default=None, help='Path to the corresponding validation data folder (optional).')
    parser.add_argument('--img_size', type=int, default=160, help='Target HR image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--downscale_factor', type=int, default=4, help='The downscale factor used.')
    parser.add_argument('--apply_hflip', action='store_true', help='Apply horizontal flipping augmentation to the training data.')
    parser.add_argument('--apply_val_hflip', action='store_true', help='Apply horizontal flipping augmentation to the validation data.')

    # --- Training Arguments ---
    parser.add_argument('--mode', type=str, required=True, choices=['rectified_flow', 'reflow'], help="Training mode: 'rectified_flow' for Stage 1, 'reflow' for subsequent stages.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device (e.g., cuda:0, cpu).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size per device.')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps).')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial optimizer learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Optimizer weight decay.')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes.')
    parser.add_argument('--val_every_n_epochs', type=int, default=1, help='Frequency (in epochs) to run validation.')
    
    # --- Scheduler Arguments ---
    parser.add_argument('--scheduler_type', type=str, default='cosineannealinglr', choices=['none', 'steplr', 'cosineannealinglr'], help='Type of LR scheduler.')
    parser.add_argument('--lr_decay_epochs', type=int, default=50, help='StepLR: decay LR every N epochs.')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR: LR decay factor.')
    parser.add_argument('--cosine_t_max_epochs', type=int, default=None, help='CosineAnnealingLR: T_max in epochs (defaults to total epochs if None).')
    parser.add_argument('--eta_min_lr', type=float, default=1e-6, help='CosineAnnealingLR: minimum learning rate.')

    # --- UNet Arguments ---
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base channel dimension for UNet.')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Channel multipliers for UNet down/up blocks.')
    parser.add_argument('--use_attention', action='store_true', help='Whether to use attention in the UNet.')

    # --- RRDBNet (Context Extractor) Arguments ---
    parser.add_argument('--rrdb_weights_path_context_extractor', type=str, required=True, help='Path to pre-trained RRDBNet weights used as the context extractor.')
    parser.add_argument('--rrdb_num_block_context', type=int, default=17, help='Number of RRDB blocks (nb) in the context extractor RRDBNet.')
    parser.add_argument('--rrdb_num_feat_context', type=int, default=64, help='Number of features (nf) in the context extractor RRDBNet. Must match UNet cond_dim.')
    parser.add_argument('--rrdb_gc_context', type=int, default=32, help='Growth channel (gc) in the context extractor RRDBNet.')

    # --- Logging/Saving & Reflow Arguments ---
    parser.add_argument('--weights_path_unet_resume', type=str, default=None, help='Path to a UNet checkpoint to resume training from.')
    parser.add_argument('--pretrained_model_path_for_reflow', type=str, default=None, help="[REFLOW MODE ONLY] Path to the model trained in the previous stage.")
    parser.add_argument('--base_log_dir', type=str, default='logs/rectified_flow', help='Base directory for TensorBoard logging.')
    parser.add_argument('--base_checkpoint_dir', type=str, default='checkpoints/rectified_flow', help='Base directory for saving model checkpoints.')
    parser.add_argument('--continue_log_dir', type=str, default=None, help='Specific log directory to continue (resumes experiment).')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None, help='Specific checkpoint directory to continue (resumes experiment).')
    parser.add_argument('--verbose_load', action='store_true', help='Print detailed information about weight loading from UNet checkpoint.')

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.mode == 'reflow' and not args.pretrained_model_path_for_reflow:
        print("\nWARNING: 'reflow' mode is selected but '--pretrained_model_path_for_reflow' is not provided. The trainer will likely fail.")
        print("Please provide the path to the model from the previous training stage.\n")

    # --- Print Configuration ---
    effective_batch_size = args.batch_size * args.accumulation_steps
    print("--- Rectified Flow Training Configuration ---")
    print(f"Prepared Data Folder: {args.flow_prepared_data_folder}")
    print(f"Image Size (HR): {args.img_size}x{args.img_size}")
    print(f"Device: {args.device}, Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size (per device): {args.batch_size}, Validation Batch Size: {args.val_batch_size}")
    print(f"Accumulation Steps: {args.accumulation_steps}, Effective Batch Size: {effective_batch_size}")
    print(f"Initial Learning Rate: {args.learning_rate}, Weight Decay: {args.weight_decay}")
    print(f"Scheduler Type: {args.scheduler_type}")
    print(f"UNet: base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}, use_attention={args.use_attention}")
    print(f"Context Extractor RRDBNet Weights: {args.rrdb_weights_path_context_extractor}")
    print(f"Context Extractor RRDBNet Config for U-Net condition: nf={args.rrdb_num_feat_context}, nb={args.rrdb_num_block_context}")
    if args.weights_path_unet_resume: 
        print(f"UNet Weights Path (for resume): {args.weights_path_unet_resume}")
    if args.mode == 'reflow':
        print(f"Pre-trained Model for Reflow: {args.pretrained_model_path_for_reflow}")
    print("-------------------------------------------")

    train_flow(args)