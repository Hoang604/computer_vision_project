import torch
from torch.utils.data import DataLoader
from src.data_handling.dataset import ImageDatasetRRDB
import os
import yaml
from src.diffusion_modules.unet import Unet
from src.trainers.diffusion_NoisetoHR_trainer import DiffusionTrainer
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
from types import SimpleNamespace

def run_diffusion_no_refine_training(config):
    """
    Main function to set up and run the diffusion model training
    using preprocessed data (LR, HR_RRDB, HR_Original).
    LR features are extracted on-the-fly by a context_extractor RRDBNet.
    """

    context_mode_for_trainer = config.context
    assert context_mode_for_trainer in ['LR', 'HR'], "Context mode for trainer must be either 'LR' or 'HR'."

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

    # --- Setup Dataset and DataLoader ---
    print(f"Loading preprocessed data (LR, HR_RRDB, HR_Orig) from: {config.preprocessed_data_folder}")
    train_dataset = ImageDatasetRRDB(
        preprocessed_folder_path=config.preprocessed_data_folder,
        img_size=config.img_size,
        downscale_factor=config.downscale_factor,
        apply_hflip=config.apply_hflip
    )
    print(f"Loaded {len(train_dataset)} samples for training.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = None
    if config.val_preprocessed_data_folder:
        print(f"Loading validation data from: {config.val_preprocessed_data_folder}")
        val_dataset = ImageDatasetRRDB(
            preprocessed_folder_path=config.val_preprocessed_data_folder,
            img_size=config.img_size,
            downscale_factor=config.downscale_factor,
            apply_hflip=config.apply_val_hflip
        )
        print(f"Loaded {len(val_dataset)} samples for validation.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        print("No validation data folder provided. Skipping validation.")

    # --- Initialize RRDBNet Context Extractor ---
    rrdb_context_config = {
        'in_nc': config.img_channels,
        'out_nc': config.img_channels,
        'num_feat': config.rrdb_num_feat_context,
        'num_block': config.rrdb_num_block_context,
        'gc': config.rrdb_gc_context,
        'sr_scale': config.downscale_factor
    }
    try:
        if not config.rrdb_weights_path_context_extractor or not os.path.exists(config.rrdb_weights_path_context_extractor):
            raise FileNotFoundError(f"RRDBNet weights for context extractor not found at: {config.rrdb_weights_path_context_extractor}")
        context_extractor_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=config.rrdb_weights_path_context_extractor,
            model_config=rrdb_context_config,
            device=device
        )
        context_extractor_model.eval()
        print(f"RRDBNet context extractor loaded from {config.rrdb_weights_path_context_extractor}")
        print(f"Context extractor config: nf={config.rrdb_num_feat_context}, nb={config.rrdb_num_block_context}, gc={config.rrdb_gc_context}")
    except Exception as e:
        print(f"Error loading RRDBNet context extractor: {e}")
        print("Please check the path and configuration of the context extractor RRDBNet.")
        return

    # --- Initialize DiffusionTrainer ---
    diffusion_trainer = DiffusionTrainer(
        timesteps=config.timesteps,
        device=device,
        mode=config.diffusion_mode,
    )

    # --- Initialize UNet Model ---
    unet_model = Unet(
        base_dim=config.unet_base_dim,
        dim_mults=tuple(config.unet_dim_mults),
        use_attention=config.use_attention,
        cond_dim=config.rrdb_num_feat_context,
        rrdb_num_blocks=config.rrdb_num_block_context
    ).to(device)
    print(f"UNet model initialized with base_dim={config.unet_base_dim}, dim_mults={tuple(config.unet_dim_mults)}, use_attention={config.use_attention}")
    print(f"UNet cond_dim={config.rrdb_num_feat_context}, cond_rrdb_num_blocks={config.rrdb_num_block_context}")

    # --- Initialize Optimizer ---
    optimizer = torch.optim.AdamW(
        unet_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {config.learning_rate}, Weight Decay: {config.weight_decay}")

    # --- Initialize Scheduler ---
    scheduler = None
    if config.scheduler_type.lower() != "none":
        if config.scheduler_type.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.lr_decay_epochs,
                gamma=config.lr_gamma
            )
            print(f"Using StepLR scheduler with step_size_epochs={config.lr_decay_epochs}, gamma={config.lr_gamma}")
        elif config.scheduler_type.lower() == "cosineannealinglr":
            t_max_epochs_for_scheduler = config.cosine_t_max_epochs if config.cosine_t_max_epochs is not None else config.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max_epochs_for_scheduler,
                eta_min=config.eta_min_lr
            )
            print(f"Using CosineAnnealingLR scheduler with T_max_epochs={t_max_epochs_for_scheduler}, eta_min={config.eta_min_lr}")
        elif config.scheduler_type.lower() == "exponentiallr":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.lr_gamma
            )
            print(f"Using ExponentialLR scheduler with gamma={config.lr_gamma}")
        else:
            print(f"Warning: Unknown scheduler type '{config.scheduler_type}'. No scheduler will be used.")
            config.scheduler_type = "none"

    # --- Load Checkpoint if available ---
    start_epoch = 0
    best_loss = float('inf')
    weights_path_unet = config.weights_path_unet if config.weights_path_unet and os.path.exists(config.weights_path_unet) else None
    if weights_path_unet:
        print(f"Attempting to load UNet checkpoint from: {weights_path_unet}")
        start_epoch, best_loss = DiffusionTrainer.load_checkpoint_for_resume(
            device=device,
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if config.scheduler_type.lower() != "none" else None,
            checkpoint_path=weights_path_unet,
            verbose_load=config.verbose_load
        )
        print(f"Loaded UNet checkpoint. Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")
    else:
        print("No valid pre-trained UNet weights path found or specified. Starting UNet training from scratch.")

    # --- Start Training ---
    print(f"\nStarting diffusion model training (mode: {config.diffusion_mode}, UNet conditioned on on-the-fly RRDBNet features from LR)...")
    try:
        diffusion_trainer.train(
            train_dataset=train_loader,
            model=unet_model,
            optimizer=optimizer,
            scheduler=scheduler if config.scheduler_type.lower() != "none" else None,
            context_extractor=context_extractor_model,
            val_dataset=val_loader,
            val_every_n_epochs=config.val_every_n_epochs,
            accumulation_steps=config.accumulation_steps,
            epochs=config.epochs,
            start_epoch=start_epoch,
            best_loss=best_loss,
            log_dir_param=config.continue_log_dir,
            checkpoint_dir_param=config.continue_checkpoint_dir,
            log_dir_base=config.base_log_dir,
            checkpoint_dir_base=config.base_checkpoint_dir,
        )
    except Exception as train_error:
        print(f"\nERROR occurred during training: {train_error}")
        import traceback
        traceback.print_exc()
        print("This might be due to issues like CUDA memory, shape mismatches, or data loading.")
        raise
