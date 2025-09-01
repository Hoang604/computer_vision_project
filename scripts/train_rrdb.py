import torch
from torch.utils.data import DataLoader
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
import yaml
from src.data_handling.dataset import ImageDataset
import os
from types import SimpleNamespace

def train_rrdb_main(config):
    """
    Main function to set up and run RRDBNet training with configurable schedulers
    and optional validation.
    """
    # --- Setup Device ---
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA device {config.device} requested but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    elif not config.device.startswith("cuda") and config.device != "cpu":
        print(f"Invalid device specified: {config.device}. Using CPU if CUDA not available, else cuda:0.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")

    # --- Prepare Training Dataset and DataLoader ---
    train_dataset = ImageDataset(folder_path=config.image_folder,
                                 img_size=config.img_size,
                                 downscale_factor=config.downscale_factor)
    print(f"Loaded {len(train_dataset)} training images from {config.image_folder}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- Prepare Validation Dataset and DataLoader (Optional) ---
    val_loader = None
    if config.val_image_folder:
        if not os.path.isdir(config.val_image_folder):
            print(f"Warning: Validation image folder not found: {config.val_image_folder}. Proceeding without validation.")
        else:
            val_dataset = ImageDataset(preprocessed_folder_path=config.val_image_folder,
                                       img_size=config.img_size,
                                       downscale_factor=config.downscale_factor)
            if len(val_dataset) > 0:
                print(f"Loaded {len(val_dataset)} validation images from {config.val_image_folder}")
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.val_batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                    drop_last=False
                )
            else:
                print(f"Warning: No images found in validation folder: {config.val_image_folder}. Proceeding without validation.")
    else:
        print("No validation image folder provided. Proceeding without validation.")

    # --- Define configuration dictionaries from config ---
    sr_scale = config.downscale_factor

    model_cfg = {
        'in_nc': config.img_channels,
        'out_nc': config.img_channels,
        'num_feat': config.rrdb_num_feat,
        'num_block': config.rrdb_num_block,
        'gc': config.rrdb_gc,
        'sr_scale': sr_scale
    }
    optimizer_cfg = {
        'lr': config.learning_rate,
        'beta1': config.adam_beta1,
        'beta2': config.adam_beta2,
        'weight_decay': config.weight_decay
    }

    scheduler_cfg = {'type': config.scheduler_type.lower()}
    if config.scheduler_type.lower() == 'steplr':
        scheduler_cfg['step_lr_step_size'] = config.step_lr_step_size
        scheduler_cfg['step_lr_gamma'] = config.step_lr_gamma
    elif config.scheduler_type.lower() == 'cosineannealinglr':
        scheduler_cfg['cosine_t_max'] = config.cosine_t_max if config.cosine_t_max is not None else config.epochs
        scheduler_cfg['cosine_eta_min'] = config.cosine_eta_min
    elif config.scheduler_type.lower() == 'cosineannealingwarmrestarts':
        scheduler_cfg['cosine_warm_t_0'] = config.cosine_warm_t_0
        scheduler_cfg['cosine_warm_t_mult'] = config.cosine_warm_t_mult
        scheduler_cfg['cosine_warm_eta_min'] = config.cosine_warm_eta_min
    elif config.scheduler_type.lower() == 'reducelronplateau':
        scheduler_cfg['plateau_mode'] = config.plateau_mode
        scheduler_cfg['plateau_factor'] = config.plateau_factor
        scheduler_cfg['plateau_patience'] = config.plateau_patience
        scheduler_cfg['plateau_verbose'] = True

    logging_cfg = {
        'exp_name': config.exp_name,
        'log_dir_base': config.base_log_dir,
        'checkpoint_dir_base': config.base_checkpoint_dir
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
            epochs=config.epochs,
            val_loader=val_loader,
            val_every_n_epochs=config.val_every_n_epochs,
            accumulation_steps=config.accumulation_steps,
            log_dir_param=config.continue_log_dir,
            checkpoint_dir_param=config.continue_checkpoint_dir,
            resume_checkpoint_path=config.weights_path,
            save_every_n_epochs=config.save_every_n_epochs,
            predict_residual=config.predict_residual
        )
    except Exception as train_error:
        print(f"\nERROR occurred during RRDBNet training: {train_error}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
    
    config = SimpleNamespace(**config_dict)

    # --- Print Configuration ---
    print(f"--- RRDBNet Training Configuration ---")
    for arg_name, arg_val in vars(config).items():
        print(f"  {arg_name}: {arg_val}")
    print(f"  SR Scale (from downscale_factor): {config.downscale_factor}")
    if config.scheduler_type.lower() == 'cosineannealinglr' and config.cosine_t_max is None:
        print(f"  CosineAnnealingLR T_max will default to total epochs: {config.epochs}")
    print(f"------------------------------------")

    train_rrdb_main(config)
