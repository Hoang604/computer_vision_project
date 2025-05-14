import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rrdb_trainer import BasicRRDBNetTrainer # Assuming rrdb_trainer.py is in the same directory or accessible in PYTHONPATH
import argparse
from utils.dataset import ImageDataset # Or your appropriate dataset module

def train_rrdb_main(args):
    """
    Main function to set up and run RRDBNet training with configurable schedulers.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    elif not args.device.startswith("cuda") and args.device != "cpu":
        print(f"Invalid device specified: {args.device}. Using CPU if CUDA not available, else cuda:0.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Prepare Dataset and DataLoader ---
    train_dataset = ImageDataset(folder_path=args.image_folder, 
                                 img_size=args.img_size, 
                                 downscale_factor=args.downscale_factor)
    print(f"Loaded {len(train_dataset)} images from {args.image_folder}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- Define configuration dictionaries from args ---
    sr_scale = args.downscale_factor 
    
    model_cfg = {
        'in_nc': args.img_channels,
        'out_nc': args.img_channels,
        'num_feat': args.rrdb_num_feat,
        'num_block': args.rrdb_num_block,
        'gc': args.rrdb_gc,
        'sr_scale': sr_scale 
    }
    optimizer_cfg = {
        'lr': args.learning_rate,
        'beta1': args.adam_beta1,
        'beta2': args.adam_beta2,
        'weight_decay': args.weight_decay
    }
    
    # Build scheduler_cfg based on args.scheduler_type
    scheduler_cfg = {'type': args.scheduler_type.lower()} # Ensure type is lowercase
    if args.scheduler_type.lower() == 'steplr':
        scheduler_cfg['step_lr_step_size'] = args.step_lr_step_size
        scheduler_cfg['step_lr_gamma'] = args.step_lr_gamma
    elif args.scheduler_type.lower() == 'cosineannealinglr':
        scheduler_cfg['cosine_t_max'] = args.cosine_t_max if args.cosine_t_max is not None else args.epochs
        scheduler_cfg['cosine_eta_min'] = args.cosine_eta_min
    elif args.scheduler_type.lower() == 'cosineannealingwarmrestarts':
        scheduler_cfg['cosine_warm_t_0'] = args.cosine_warm_t_0
        scheduler_cfg['cosine_warm_t_mult'] = args.cosine_warm_t_mult
        scheduler_cfg['cosine_warm_eta_min'] = args.cosine_warm_eta_min
    elif args.scheduler_type.lower() == 'reducelronplateau':
        scheduler_cfg['plateau_mode'] = args.plateau_mode
        scheduler_cfg['plateau_factor'] = args.plateau_factor
        scheduler_cfg['plateau_patience'] = args.plateau_patience
        scheduler_cfg['plateau_verbose'] = True # Or make this an arg

    logging_cfg = {
        'exp_name': args.exp_name,
        'log_dir_base': args.base_log_dir,
        'checkpoint_dir_base': args.base_checkpoint_dir
    }

    # --- Initialize and train ---
    rrdb_trainer = BasicRRDBNetTrainer(
        model_config=model_cfg,
        optimizer_config=optimizer_cfg,
        scheduler_config=scheduler_cfg,
        logging_config=logging_cfg,
        device=device
    )

    print("\nStarting RRDBNet training process...")
    try:
        rrdb_trainer.train(
            train_loader=train_loader,
            epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            log_dir_param=args.continue_log_dir,
            checkpoint_dir_param=args.continue_checkpoint_dir,
            resume_checkpoint_path=args.weights_path,
            save_every_n_epochs=args.save_every_n_epochs
        )
    except Exception as train_error:
        print(f"\nERROR occurred during RRDBNet training: {train_error}")
        raise 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RRDBNet Model with Advanced Schedulers")
    
    # Dataset and Model args
    parser.add_argument('--image_folder', type=str, default="/media/tuannl1/heavy_weight/data/cv_data/images160x160", help='Path to the image folder for HR images')
    parser.add_argument('--img_size', type=int, default=160, help='Target HR image size')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Factor to downscale HR to get LR (determines sr_scale)')
    
    parser.add_argument('--rrdb_num_feat', type=int, default=64, help='Number of features (nf) in RRDBNet')
    parser.add_argument('--rrdb_num_block', type=int, default=8, help='Number of RRDB blocks (nb) in RRDBNet')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='Growth channel (gc) in RRDBNet')
    
    # Training args
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on (e.g., cuda:0, cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Optimizer learning rate')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')

    # Scheduler args
    parser.add_argument('--scheduler_type', type=str, default='CosineAnnealingLR', 
                        choices=['none', 'StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau'],
                        help='Type of LR scheduler to use.')
    
    # StepLR specific args
    parser.add_argument('--step_lr_step_size', type=int, default=200000, help='StepLR: decay LR every N optimizer steps')
    parser.add_argument('--step_lr_gamma', type=float, default=0.5, help='StepLR: LR decay factor')
    
    # CosineAnnealingLR specific args
    parser.add_argument('--cosine_t_max', type=int, default=None, help='CosineAnnealingLR: T_max in epochs (defaults to total epochs if None)')
    parser.add_argument('--cosine_eta_min', type=float, default=0.0, help='CosineAnnealingLR: minimum learning rate')
    
    # CosineAnnealingWarmRestarts specific args
    parser.add_argument('--cosine_warm_t_0', type=int, default=10, help='CosineAnnealingWarmRestarts: Number of epochs for the first restart')
    parser.add_argument('--cosine_warm_t_mult', type=int, default=1, help='CosineAnnealingWarmRestarts: Factor to increase T_i after a restart')
    parser.add_argument('--cosine_warm_eta_min', type=float, default=0.0, help='CosineAnnealingWarmRestarts: minimum learning rate')

    # ReduceLROnPlateau specific args
    parser.add_argument('--plateau_mode', type=str, default='min', choices=['min', 'max'], help='ReduceLROnPlateau: mode (min or max)')
    parser.add_argument('--plateau_factor', type=float, default=0.1, help='ReduceLROnPlateau: factor by which LR is reduced')
    parser.add_argument('--plateau_patience', type=int, default=10, help='ReduceLROnPlateau: number of epochs with no improvement')
    
    # Logging/Saving args
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for subdirectories (e.g., rrdb_cosine_run1)')
    parser.add_argument('--base_log_dir', type=str, default='logs_rrdb', help='Base directory for logging')
    parser.add_argument('--base_checkpoint_dir', type=str, default='checkpoints_rrdb', help='Base directory for saving checkpoints')
    parser.add_argument('--continue_log_dir', type=str, default=None, help='Specific directory to continue logging')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None, help='Specific directory to continue saving checkpoints')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pre-trained model weights to resume training')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # --- Print Configuration ---
    print(f"--- RRDBNet Training Configuration ---")
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}")
    print(f"  SR Scale (from downscale_factor): {args.downscale_factor}")
    if args.scheduler_type.lower() == 'cosineannealinglr' and args.cosine_t_max is None:
        print(f"  CosineAnnealingLR T_max will default to total epochs: {args.epochs}")
    print(f"------------------------------------")

    train_rrdb_main(args)
