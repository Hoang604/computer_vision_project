import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rrdb_trainer import BasicRRDBNetTrainer
import argparse
from utils.dataset import ImageDataset # Hoặc dataset phù hợp của bạn

def train_rrdb_main(args):
    """
    Hàm chính để thiết lập và chạy huấn luyện RRDBNet.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Sửa lại logic chọn device một chút cho đơn giản hơn, ưu tiên cuda nếu có
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {args.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    elif not args.device.startswith("cuda") and args.device != "cpu":
        print(f"Invalid device specified: {args.device}. Using CPU if CUDA not available, else cuda:0.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Chuẩn bị Dataset và DataLoader ---
    # Giả sử ImageDataset của bạn trả về (img_lr, img_hr_target, ...)
    # img_size là kích thước của HR target
    # downscale_factor dùng để tạo LR từ HR
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

    # --- Định nghĩa các dictionary cấu hình từ args ---
    # sr_scale được tính từ downscale_factor
    sr_scale = args.downscale_factor 
    # Nếu ImageDataset không dùng downscale_factor mà bạn có sr_scale riêng thì sửa ở đây
    
    model_cfg = {
        'num_feat': args.rrdb_num_feat,
        'num_block': args.rrdb_num_block,
        'gc': args.rrdb_gc,
        'sr_scale': sr_scale 
    }
    optimizer_cfg = {
        'lr': args.learning_rate,
        'beta1': args.adam_beta1,
        'beta2': args.adam_beta2,
        'weight_decay': args.weight_decay # Thêm weight_decay
    }
    scheduler_cfg = {
        'use_scheduler': args.use_scheduler,
        'decay_steps': args.scheduler_decay_steps,
        'gamma': args.scheduler_gamma
    }
    logging_cfg = {
        'exp_name': args.exp_name, # Thêm exp_name vào args
        'log_dir_base': args.base_log_dir,
        'checkpoint_dir_base': args.base_checkpoint_dir
    }

    # --- Khởi tạo và huấn luyện ---
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
            log_dir_param=args.continue_log_dir, # Đã có sẵn
            checkpoint_dir_param=args.continue_checkpoint_dir, # Đã có sẵn
            resume_checkpoint_path=args.weights_path, # Dùng weights_path để resume
            save_every_n_epochs=args.save_every_n_epochs
        )
    except Exception as train_error:
        print(f"\nERROR occurred during RRDBNet training: {train_error}")
        raise # Ném lại lỗi để debug dễ hơn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RRDBNet Model")
    
    parser.add_argument('--image_folder', type=str, default="/media/tuannl1/heavy_weight/data/cv_data/images160x160", help='Path to the image folder for HR images')
    parser.add_argument('--img_size', type=int, default=160, help='Target HR image size (images will be resized to this)')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Factor to downscale HR to get LR (determines sr_scale)')
    
    parser.add_argument('--rrdb_num_feat', type=int, default=64, help='Number of features (nf) in RRDBNet (SRDiff: hidden_size)')
    parser.add_argument('--rrdb_num_block', type=int, default=8, help='Number of RRDB blocks (nb) in RRDBNet (SRDiff: num_block, vd: 17 cho df2k, 8 cho celeba)')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='Growth channel (gc) in RRDBNet')
    
    # Training args
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on (e.g., cuda:0, cpu)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (SRDiff RRDB: max_updates/steps)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (SRDiff RRDB: 64)')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Optimizer learning rate (SRDiff RRDB: 0.0002)')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam optimizer beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam optimizer beta2')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Optimizer weight decay (AdamW, SRDiff RRDB không dùng WD cho Adam)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')

    # Scheduler args (tham khảo SRDiff RRDB)
    parser.add_argument('--use_scheduler', type=bool, default=True, help='Whether to use LR scheduler') # Mặc định là True nếu bạn muốn giống SRDiff
    parser.add_argument('--scheduler_decay_steps', type=int, default=200000, help='StepLR decay steps (SRDiff RRDB: 200000 batches, không phải epochs)') # Cần điều chỉnh nếu dùng theo epoch
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='StepLR gamma (SRDiff RRDB: 0.5)')
    
    # Logging/Saving args
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for subdirectories')
    parser.add_argument('--base_log_dir', type=str, default='logs_rrdb', help='Base directory for logging')
    parser.add_argument('--base_checkpoint_dir', type=str, default='checkpoints_rrdb', help='Base directory for saving checkpoints')
    parser.add_argument('--continue_log_dir', type=str, default=None, help='Specific directory to continue logging (overrides exp_name logic)')
    parser.add_argument('--continue_checkpoint_dir', type=str, default=None, help='Specific directory to continue saving checkpoints')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pre-trained model weights to resume training')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # --- Print Configuration ---
    print(f"--- RRDBNet Training Configuration ---")
    for arg_name, arg_val in vars(args).items():
        print(f"{arg_name}: {arg_val}")
    print(f"SR Scale (from downscale_factor): {args.downscale_factor}")
    print(f"------------------------------------")

    train_rrdb_main(args)