# Image Super-Resolution Project using Diffusion Models and RRDBNet

## Overview

This project focuses on implementing and exploring image super-resolution techniques, a crucial area in Computer Vision. The primary goal is to enhance image quality by increasing resolution, employing a combination of advanced deep learning models including Diffusion Models and the RRDBNet (Residual-in-Residual Dense Block Network) architecture. The project also includes the traditional Bicubic interpolation method for comparison.

## Key Features

* **Multiple Super-Resolution Methods Implemented**:
    * Bicubic Interpolation.
    * RRDBNet Model.
    * Conditional Diffusion Model (U-Net) based on features from RRDBNet.
* **Model Training**: Provides scripts to train RRDBNet and U-Net (Diffusion) models from scratch or resume from saved checkpoints.
* **Inference**: Offers scripts to use pre-trained models for super-resolving new images.
* **Flexible Configuration**: Supports customization of various parameters for training processes and model architectures.
* **Logging and Monitoring**: Integrates TensorBoard for tracking the training progress.
* **Data Handling**: Includes an `ImageDataset` module for efficient loading, preprocessing, and augmentation of image data.

## Implemented Models and Techniques

1.  **Bicubic Interpolation (`bicubic.py`)**: A classic interpolation method used as a baseline.
2.  **RRDBNet (`diffusion_modules.py`, `rrdb_trainer.py`, `train_rrdb.py`, `rrdb_infer.py`)**:
    * A powerful deep learning architecture based on Residual-in-Residual Dense Blocks.
    * Can be trained independently for direct super-resolution or to predict a residual.
    * Used as a context extractor for the Diffusion Model.
3.  **Diffusion Model (U-Net based on DDPM/DDIM) (`diffusion_modules.py`, `diffusion_trainer.py`, `train_diffusion.py`, `diffusion_infer.py`)**:
    * Utilizes a U-Net architecture to learn the reverse of a noise-adding process on the image "residual."
    * The residual is calculated as the difference between the high-resolution (HR) image and the low-resolution (LR) image upscaled via Bicubic interpolation.
    * The U-Net model is conditioned on features extracted from the LR image by an RRDBNet.
    * Supports `noise` (predicting noise $\epsilon$) and `v_prediction` modes.
    * Uses a `DDIMScheduler` for the inference (sampling) process.

## Directory Structure (Overview)

```
computer_vision_project/
│
├── data/                     # (User-created) Directory for HR image data
├── bicubic_output/           # (Auto-generated) Directory for results from bicubic.py
│
├── bicubic.py                # Script for Bicubic super-resolution
├── diffusion_modules.py      # Defines network architectures (U-Net, RRDBNet, etc.)
├── diffusion_trainer.py      # Training logic for the Diffusion Model (U-Net)
├── diffusion_infer.py        # Inference script for the Diffusion Model
├── train_diffusion.py        # Main script to train the Diffusion Model
│
├── rrdb_trainer.py           # Training logic for RRDBNet (standalone)
├── rrdb_infer.py             # Inference script for RRDBNet
├── train_rrdb.py             # Main script to train RRDBNet
│
├── utils/
│   ├── dataset.py            # Defines ImageDataset
│   └── network_components.py # Auxiliary network components
│
├── cv_logs_diffusion/        # (Auto-generated) TensorBoard logs for Diffusion Model
├── cv_checkpoints_diffusion/ # (Auto-generated) Checkpoints for Diffusion Model
├── logs_rrdb/                # (Auto-generated) TensorBoard logs for RRDBNet
├── checkpoints_rrdb/         # (Auto-generated) Checkpoints for RRDBNet
│
└── README.md                 # This file
```

## Installation and Environment Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/Hoang604/computer_vision_project.git
    cd computer_vision_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install necessary libraries:**
    Create a `requirements.txt` file with the following content (or based on imports in the code):
    ```txt
    torch
    torchvision
    torchaudio
    numpy
    opencv-python-headless
    Pillow
    matplotlib
    tqdm
    tensorboard
    diffusers
    # bitsandbytes # If you plan to use 8-bit Adam (currently commented out)
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *Note*: Ensure your PyTorch version is compatible with CUDA if you are using a GPU.

## Data Preparation

1.  Create a directory to store your high-resolution (HR) images, e.g., `data/my_hr_images`.
2.  The `ImageDataset` class in `utils/dataset.py` will automatically handle the creation of low-resolution (LR) images, Bicubic upscaled images, and residual images during data loading.

## How to Train

### 1. Train RRDBNet (as Context Extractor or Standalone SR)

Run the `train_rrdb.py` script with appropriate arguments.

**Example command:**
```bash
python train_rrdb.py \
    --image_folder data/my_hr_images \
    --img_size 160 \
    --downscale_factor 4 \
    --rrdb_num_feat 64 \
    --rrdb_num_block 8 \
    --epochs 50 \
    --batch_size 32 \
    --accumulation_steps 1 \
    --learning_rate 2e-4 \
    --scheduler_type CosineAnnealingLR \
    --cosine_t_max 50 \
    --device cuda:0 \
    --exp_name rrdb_model_v1 \
    --base_log_dir ./logs_rrdb \
    --base_checkpoint_dir ./checkpoints_rrdb
    # --weights_path ./checkpoints_rrdb/some_experiment/rrdb_model_best.pth # To resume training
    # --predict_residual # If you want to train to predict the residual
```

**Key RRDBNet training arguments:**

* `--image_folder`: Path to the HR image directory.
* `--img_size`: Target size of the HR images (e.g., 160 for 160x160 images).
* `--downscale_factor`: Factor to create LR images (e.g., 4 for 40x40 LR from 160x160 HR). This is also the `sr_scale`.
* `--rrdb_num_feat`: Number of features in RRDBNet (nf).
* `--rrdb_num_block`: Number of RRDB blocks.
* `--rrdb_gc`: Growth channel in RRDBNet.
* `--epochs`: Number of training epochs.
* `--batch_size`: Batch size per device.
* `--accumulation_steps`: Gradient accumulation steps.
* `--learning_rate`: Initial learning rate.
* `--scheduler_type`: Type of learning rate scheduler (e.g., `none`, `StepLR`, `CosineAnnealingLR`).
    * For `StepLR`: `--step_lr_step_size`, `--step_lr_gamma`.
    * For `CosineAnnealingLR`: `--cosine_t_max`, `--cosine_eta_min`.
    * For `CosineAnnealingWarmRestarts`: `--cosine_warm_t_0`, `--cosine_warm_t_mult`, `--cosine_warm_eta_min`.
    * For `ReduceLROnPlateau`: `--plateau_mode`, `--plateau_factor`, `--plateau_patience`.
* `--device`: Training device (e.g., `cuda:0`, `cpu`).
* `--exp_name`: Experiment name for log and checkpoint subdirectories.
* `--base_log_dir`, `--base_checkpoint_dir`: Base directories for logs and checkpoints.
* `--weights_path` (optional): Path to a checkpoint to resume training.
* `--continue_log_dir`, `--continue_checkpoint_dir` (optional): Specific directories to resume logging/checkpointing for an existing experiment.
* `--save_every_n_epochs`: Frequency for saving checkpoints.
* `--predict_residual` (optional): If set, the model learns to predict the residual instead of the full HR image.

### 2. Train Diffusion Model (U-Net)

**Prerequisite**: You need a pre-trained RRDBNet model to act as the context extractor.

Run the `train_diffusion.py` script.

**Example command:**
```bash
python train_diffusion.py \
    --image_folder data/my_hr_images \
    --img_size 160 \
    --downscale_factor 4 \
    --epochs 100 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --learning_rate 1e-4 \
    --scheduler_type cosineannealinglr \
    --t_max_epochs 100 \
    --eta_min_lr 1e-6 \
    --device cuda:0 \
    --diffusion_mode v_prediction \
    --timesteps 1000 \
    --unet_base_dim 64 \
    --unet_dim_mults 1 2 4 8 \
    # --use_attention # Enable if you want to use attention in the U-Net's mid-block
    --rrdb_weights_path ./checkpoints_rrdb/rrdb_model_v1/rrdb_model_best.pth \
    --rrdb_num_feat 64 \
    --number_of_rrdb_blocks 8 \
    --rrdb_gc 32 \
    --base_log_dir ./cv_logs_diffusion \
    --base_checkpoint_dir ./cv_checkpoints_diffusion \
    # --weights_path ./cv_checkpoints_diffusion/some_experiment/diffusion_model_best.pth # To resume U-Net training
    # --continue_log_dir ./cv_logs_diffusion/existing_experiment_name # To resume logging
    # --continue_checkpoint_dir ./cv_checkpoints_diffusion/existing_experiment_name # To resume checkpointing
```

**Key Diffusion Model training arguments:**

* Dataset arguments (`--image_folder`, `--img_size`, `--downscale_factor`, `--img_channels`) are similar to RRDBNet training.
* `--epochs`, `--batch_size`, `--accumulation_steps`, `--learning_rate`, `--weight_decay`, `--num_workers`: Standard training parameters.
* `--scheduler_type`: Type of LR scheduler.
    * For `steplr`: `--lr_decay_epochs`, `--lr_gamma`.
    * For `cosineannealinglr`: `--t_max_epochs`, `--eta_min_lr`.
    * For `exponentiallr`: `--lr_gamma`.
* `--device`: Training device.
* `--context`: Context type for conditioning (usually `LR`).
* `--timesteps`: Number of diffusion timesteps.
* `--diffusion_mode`: `v_prediction` or `noise`.
* `--unet_base_dim`, `--unet_dim_mults`: U-Net architecture configuration.
* `--use_attention`: Flag to enable attention in the U-Net.
* `--rrdb_weights_path`: **Crucial**. Path to the pre-trained RRDBNet checkpoint.
* `--rrdb_num_feat`, `--number_of_rrdb_blocks`, `--rrdb_gc`: Configuration of the loaded RRDBNet (must match the pre-trained model).
* `--base_log_dir`, `--base_checkpoint_dir`: Base directories for Diffusion Model logs and checkpoints.
* `--weights_path` (optional): Path to a U-Net checkpoint to resume training.
* `--continue_log_dir`, `--continue_checkpoint_dir` (optional): Specific directories to resume an existing experiment.
* `--verbose_load`: Print detailed info about weight loading.

## How to Perform Inference

### 1. Inference with RRDBNet

Modify and run the `rrdb_infer.py` script.

* **Required modifications in `rrdb_infer.py`**:
    * `model_path`: Path to your trained RRDBNet checkpoint (`.pth` file).
    * `config`: Dictionary containing the RRDBNet configuration (must match training).
    * `img_size`, `downscale_factor` for `ImageDataset`.
    * `predict_residual`: Set to `True` if the RRDBNet was trained to predict residuals, `False` otherwise.
* **Run**:
    ```bash
    python rrdb_infer.py
    ```

### 2. Inference with Diffusion Model

Modify and run the `diffusion_infer.py` script.

* **Required modifications in `diffusion_infer.py`**:
    * `rrdb_model_path`: Path to the trained RRDBNet checkpoint (context extractor).
    * `config`: Configuration for the RRDBNet.
    * `weight_path`: Path to your trained U-Net (Diffusion Model) checkpoint.
    * `img_size`, `downscale_factor` for `ImageDataset`.
    * `unet = Unet(use_attention=...)`: `use_attention` must match the U-Net training configuration.
    * `generator = ResidualGenerator(..., predict_mode='...')`: `predict_mode` must match the U-Net training configuration.
* **Run**:
    ```bash
    python diffusion_infer.py
    ```

## Key Scripts

* **Training**:
    * `train_rrdb.py`: Main script for training the RRDBNet.
    * `train_diffusion.py`: Main script for training the Diffusion (U-Net) model.
* **Inference**:
    * `rrdb_infer.py`: Script for performing inference with a trained RRDBNet.
    * `diffusion_infer.py`: Script for performing inference with a trained Diffusion model.
* **Core Modules**:
    * `rrdb_trainer.py`: Contains the `BasicRRDBNetTrainer` class.
    * `diffusion_trainer.py`: Contains the `DiffusionTrainer` and `ResidualGenerator` classes.
    * `diffusion_modules.py`: Defines network architectures like `RRDBNet` and `Unet`.
    * `utils/dataset.py`: Defines the `ImageDataset` for data loading.
    * `utils/network_components.py`: Contains various building blocks for networks.
    * `bicubic.py`: Implements Bicubic upscaling.

## Dependencies

Ensure you have the libraries listed in `requirements.txt` installed. Key dependencies include:

* PyTorch
* torchvision
* NumPy
* OpenCV (cv2)
* Pillow (PIL)
* Matplotlib
* tqdm
* TensorBoard
* Diffusers

## Future Work / TODO

* [ ] Implement evaluation metrics (PSNR, SSIM, LPIPS).
* [ ] Experiment with different noise schedulers for the Diffusion Model.
* [ ] Explore more advanced U-Net architectures or conditioning mechanisms.
* [ ] Optimize inference speed.
* [ ] Create a more user-friendly interface for inference.

## Acknowledgements

* This project builds upon concepts from prominent research in super-resolution and diffusion models.
* Inspired by the paper: "SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models" by Li, H., Liu, Y., Zhan, F., Lu, S., Xing, E. P., & Miao, C. (2021). ([arXiv:2104.14951](https://arxiv.org/pdf/2104.14951.pdf))
