import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard
import numpy as np
from tqdm import tqdm
import os
# import bitsandbytes as bnb # Commented out as it's not used in the provided snippet, can be re-enabled if 8-bit Adam is used
import datetime
from diffusers import DDIMScheduler # Keep this import
from torch.utils.data import DataLoader # For type hinting
# from diffusion_modules import Unet # For type hinting, assuming Unet is defined elsewhere

# Assuming ResidualGenerator is defined in this file or imported.
# If it's in another file, ensure it's imported correctly.
# For this modification, we primarily focus on DiffusionTrainer.

class DiffusionTrainer:
    """
    Implements a Denoising Diffusion Probabilistic Model (DDPM) trainer.

    This class handles training logic. It uses a cosine
    noise schedule by default. The model can be trained to predict either
    the noise added during the forward process or a 'v-prediction' target.
    Learning rate scheduler can be integrated into the training loop.

    Attributes:
        timesteps (int): The total number of timesteps in the diffusion process.
        device (str or torch.device): The device on which to perform computations ('cuda' or 'cpu').
        mode (str): The prediction mode, either "v_prediction" or "noise".
        betas (torch.Tensor): Noise schedule (variance of noise added at each step).
        alphas (torch.Tensor): 1.0 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        alphas_cumprod_shift_right (torch.Tensor): Cumulative product of alphas, shifted right by one.
        sqrt_alphas_cumprod (torch.Tensor): Square root of alphas_cumprod.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Square root of (1.0 - alphas_cumprod).
        sqrt_recip_alphas (torch.Tensor): Square root of the reciprocal of alphas.
        posterior_variance (torch.Tensor): Variance of the posterior distribution q(x_{t-1} | x_t, x_0).
    """
    def __init__(
        self,
        timesteps=1000,
        device='cuda',
        mode="v_prediction" # Added mode attribute
    ):
        """
        Initializes the DiffusionTrainer with a specified number of timesteps, device, and prediction mode.

        It pre-computes various parameters of the diffusion process based on a
        cosine noise schedule.

        Args:
            timesteps (int, optional): Number of diffusion steps. Defaults to 1000.
            device (str or torch.device, optional): Device for tensor operations ('cuda' or 'cpu').
                                                   Defaults to 'cuda'.
            mode (str, optional): The prediction mode for the model during training.
                                  Can be "v_prediction" or "noise".
                                  Defaults to "v_prediction".
        Raises:
            ValueError: If the provided `mode` is not "v_prediction" or "noise".
        """
        self.timesteps = timesteps
        self.device = device
        
        if mode not in ["v_prediction", "noise"]:
            raise ValueError("Mode must be 'v_prediction' or 'noise'") # Validate mode
        self.mode = mode
        print(f"DiffusionTrainer initialized in '{self.mode}' mode.") # Log initialization mode


        # Define cosine noise schedule using PyTorch tensors
        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            """
            Generates a cosine noise schedule (betas).

            Args:
                timesteps (int): The number of timesteps.
                s (float, optional): Small offset to prevent beta_t from being too small near t=0.
                                     Defaults to 0.008.
                dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.float32.

            Returns:
                torch.Tensor: The beta schedule.
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) # Use PyTorch's clip

        self.betas = cosine_beta_schedule(timesteps).to(self.device)

        # Pre-calculate diffusion parameters using PyTorch tensors
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        # Use torch.cat instead of np.append for PyTorch tensors
        self.alphas_cumprod_shift_right = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)

        # Parameters for sampling
        self.posterior_variance = (self.betas *
            (1.0 - self.alphas_cumprod_shift_right) /
            (1.0 - self.alphas_cumprod)).to(self.device)
        # Ensure no NaN from division by zero (can happen at t=0 if not careful)
        self.posterior_variance = torch.nan_to_num(self.posterior_variance, nan=0.0, posinf=0.0, neginf=0.0)


    def _extract(self, a, t, x_shape):
        """
        Helper function to extract specific coefficients at given timesteps `t` and reshape
        them to match the batch shape of `x_shape`.

        This is used to gather the appropriate alpha, beta, etc., values for a batch
        of images at different timesteps.

        Args:
            a (torch.Tensor): The tensor to extract coefficients from (e.g., alphas_cumprod).
            t (torch.Tensor): A 1D tensor of timesteps for each item in the batch. Shape [B,].
            x_shape (torch.Size): The shape of the data tensor x (e.g., [B, C, H, W]).

        Returns:
            torch.Tensor: A tensor of extracted coefficients, reshaped to [B, 1, 1, 1]
                          (or [B, 1, ..., 1] to match x_shape's dimensions) for broadcasting.
        """
        batch_size = t.shape[0]
        # Use tensor indexing instead of tf.gather
        out = a[t]
        # Use view for reshaping
        return out.view(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t):
        """
        Performs the forward diffusion process (noising) q(x_t | x_0).

        It takes an initial clean image `x_0` and a timestep `t`, and returns
        a noised version `x_t` along with the noise that was added.
        The formula used is: x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise.

        Args:
            x_0 (torch.Tensor): Input clean images, shape [B, C, H, W].
                                Assumed to be in the [-1, 1] range initially.
            t (torch.Tensor): Timesteps for each image in the batch, shape [B,].

        Returns:
            tuple:
                - torch.Tensor: Noisy images `x_t` at timestep `t`, range [-1, 1].
                - torch.Tensor: The noise `epsilon` added to the images.
        """
        x_0 = x_0.to(self.device) # Move to correct device
        x_0 = x_0.float() # Ensure float type

        # Create random noise
        noise = torch.randn_like(x_0, device=x_0.device)

        # Get pre-calculated parameters using the helper function
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Forward diffusion equation: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def _setup_training_directories_and_writer(self, log_dir_base, checkpoint_dir_base,
                                                log_dir_param, checkpoint_dir_param):
        """Set up log and checkpoint directories and TensorBoard writer."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Determine experiment name based on mode and timestamp if not resuming
        experiment_name = f"{self.mode}_{timestamp}"

        if log_dir_param:
            log_dir = log_dir_param # Use the provided log_dir to resume
            # Try to infer experiment name if resuming, for consistent checkpoint naming
            if checkpoint_dir_param:
                 experiment_name = os.path.basename(checkpoint_dir_param)
            else: # Fallback if only log_dir is given for resume
                 experiment_name = os.path.basename(log_dir_param)
        else:
            log_dir = os.path.join(log_dir_base, experiment_name)


        if checkpoint_dir_param:
            checkpoint_dir = checkpoint_dir_param # Use the provided checkpoint_dir to resume
        else:
            checkpoint_dir = os.path.join(checkpoint_dir_base, experiment_name)


        os.makedirs(log_dir, exist_ok=True) # Create log directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory if it doesn't exist

        writer = SummaryWriter(log_dir) # Initialize TensorBoard writer
        # Consistent naming for best checkpoint
        best_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_model_{os.path.basename(checkpoint_dir)}_best.pth')


        return writer, checkpoint_dir, log_dir, best_checkpoint_path

    def _initialize_training_steps(self, start_epoch, dataset_len, accumulation_steps):
        """Initialize step counters for resuming training."""
        # Calculate effective batches per epoch considering accumulation steps
        effective_batches_per_epoch = dataset_len // accumulation_steps
        if dataset_len % accumulation_steps != 0:
            effective_batches_per_epoch += 1

        global_step_optimizer = start_epoch * effective_batches_per_epoch # Global optimizer steps
        batch_step_counter = start_epoch * dataset_len # Total batches processed across all epochs
        current_accumulation_idx = 0 # Counter for current accumulation cycle
        return global_step_optimizer, batch_step_counter, current_accumulation_idx

    def _get_training_target(self, noise_added, original_residual_batch, t):
        """Determine the model target based on self.mode."""
        if self.mode == "v_prediction":
            # For v-prediction, target is sqrt(alphas_cumprod_t) * noise - sqrt(1 - alphas_cumprod_t) * x_0
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, original_residual_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, original_residual_batch.shape)
            target = sqrt_alphas_cumprod_t * noise_added - sqrt_one_minus_alphas_cumprod_t * original_residual_batch
        elif self.mode == "noise":
            # For noise prediction (epsilon-prediction), target is the noise itself
            target = noise_added
        else:
            raise ValueError(f"Unsupported training mode: {self.mode}") # Should not happen if validated in __init__
        return target

    def _perform_batch_step(self, model, context_extractor, batch_data, context_selection_mode, optimizer,
                            accumulation_steps, current_accumulation_idx, global_step_optimizer):
        """Process a training batch: forward pass, loss calculation, backward pass, and optimizer step (if enough accumulation)."""
        # Unpack batch data - assuming (low_res, upscaled_bicubic, original_hr, original_residual)
        low_res_image_batch, _, _, residual_image_batch = batch_data

        low_res_image_batch = low_res_image_batch.to(self.device) # Move LR images to device
        # Extract condition features using the context_extractor (e.g., RRDBNet)
        # The context_selection_mode might determine if LR or HR (or other) is used by context_extractor
        # For now, assuming context_extractor always takes low_res_image_batch
        with torch.no_grad(): # Context extraction should not be part of gradient computation for the diffusion model
            _, condition_features_list = context_extractor(low_res_image_batch, get_fea=True) # get_fea=True if RRDBNet returns features

        residual_image_batch = residual_image_batch.to(self.device) # Move target residual images to device

        actual_batch_size = residual_image_batch.shape[0] # Get actual batch size (can be smaller for last batch)
        # Sample random timesteps t for each image in the batch
        t = torch.randint(0, self.timesteps, (actual_batch_size,), device=self.device, dtype=torch.long)
        # Perform forward diffusion (q_sample) to get noisy image x_t and the added noise
        residual_image_batch_t, noise_added = self.q_sample(residual_image_batch, t)

        # Determine the target for the model based on the prediction mode
        target = self._get_training_target(noise_added, residual_image_batch, t)

        # Get model prediction (e.g., U-Net predicts noise or v)
        # The model (U-Net) takes the noisy image x_t, timestep t, and condition features
        predicted_output = model(residual_image_batch_t, t, condition=condition_features_list) # Pass condition_features_list as context
        loss = F.mse_loss(predicted_output, target) # Calculate Mean Squared Error loss
        scaled_loss = loss / accumulation_steps # Scale loss for gradient accumulation
        scaled_loss.backward() # Perform backward pass

        current_accumulation_idx += 1 # Increment accumulation counter
        updated_optimizer_this_step = False # Flag to check if optimizer was updated
        if current_accumulation_idx >= accumulation_steps:
            # Optional: Gradient clipping can be added here if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # Update model parameters
            optimizer.zero_grad() # Reset gradients
            current_accumulation_idx = 0 # Reset accumulation counter
            global_step_optimizer +=1 # Increment global optimizer step counter
            updated_optimizer_this_step = True # Mark optimizer as updated

        return loss.detach().item(), current_accumulation_idx, global_step_optimizer, updated_optimizer_this_step

    def _log_and_checkpoint_epoch_end(self, epoch, model, optimizer, scheduler, epoch_losses, current_best_loss,
                                        global_step_optimizer, best_checkpoint_path, writer, dataset): # Added scheduler
        """Handle end-of-epoch tasks: log loss, save checkpoint, step scheduler, generate sample images."""
        mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan') # Calculate mean loss for the epoch
        print(f"Epoch {epoch+1} Average Loss ({self.mode}): {mean_loss:.4f}") # Log epoch average loss
        writer.add_scalar(f'Loss_{self.mode}/epoch_avg', mean_loss, epoch + 1) # Log to TensorBoard

        # Log learning rate
        if optimizer.param_groups:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'LearningRate/epoch', current_lr, epoch + 1) # Log LR to TensorBoard
            print(f"Current Learning Rate: {current_lr:.2e}") # Log current LR

        new_best_loss = current_best_loss
        if mean_loss < current_best_loss:
            new_best_loss = mean_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': new_best_loss,
                'global_optimizer_steps': global_step_optimizer, # Save global optimizer steps
                'mode': self.mode # Save training mode
            }
            if scheduler: # Save scheduler state if it exists
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, best_checkpoint_path) # Save checkpoint
            print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Mode: {self.mode}, OptSteps: {global_step_optimizer})") # Log checkpoint save

        # Step the scheduler (typically after an epoch)
        if scheduler:
            # Some schedulers like ReduceLROnPlateau need the metric (e.g., validation loss)
            # For simplicity, assuming epoch-based schedulers here.
            # If using ReduceLROnPlateau, you'd do: scheduler.step(mean_loss_on_validation_set)
            scheduler.step()
            print(f"Scheduler stepped. New LR (from optimizer): {optimizer.param_groups[0]['lr']:.2e}") # Log scheduler step

        if (epoch + 1) % 5 == 0: # Generate sample images every 5 epochs (configurable)
            self._generate_and_log_samples(model, dataset, epoch, writer) # Generate and log samples

        return new_best_loss

    def _generate_and_log_samples(self, model, dataset, epoch, writer):
        """Generate sample images and log to TensorBoard."""

        print(f"Generating sample images for TensorBoard at epoch {epoch+1}...") # Log sample generation start
        try:
            # Assuming ResidualGenerator is defined and accessible
            generator = ResidualGenerator(
                img_channels=model.final_conv[-1].out_channels if hasattr(model, 'final_conv') else 3, # Infer channels if possible
                img_size=dataset.dataset.img_size if hasattr(dataset, 'dataset') else 160, # Infer img_size
                device=self.device,
                num_train_timesteps=self.timesteps,
                predict_mode=self.mode # Use the trainer's mode for the generator
            )
        except NameError:
            print("Warning: ResidualGenerator class not found or cannot infer parameters. Skipping sample generation.") # Log warning
            return
        except Exception as e:
            print(f"Error initializing ResidualGenerator: {e}. Skipping sample generation.") # Log error
            return


        sample_images_data = [] # List to store image data for logging
        # Get a few samples from the dataset to generate images
        num_samples_to_generate = min(3, len(dataset.dataset) if hasattr(dataset, 'dataset') else 3) # Generate a few samples
        if num_samples_to_generate == 0:
            print("Warning: Dataset is empty or too small. Skipping sample generation.") # Log warning
            return

        # Need context_extractor for generation if model expects it
        # This part requires careful handling of how context is passed during generation
        # For simplicity, this example might need adjustment based on your exact model's forward pass for generation.
        # Let's assume for now the model's `generate_residuals` or similar method handles context internally or doesn't need it explicitly here.
        # If context_extractor is needed, it should be passed or made accessible.

        dataset_iter = iter(dataset) # Create an iterator for the dataset
        for _ in range(num_samples_to_generate):
            try:
                # Unpack batch data - assuming (low_res, upscaled_bicubic, original_hr, original_residual)
                low_res_b, up_scale_b, original_b, residual_b_cpu = next(dataset_iter)
            except StopIteration:
                print("Warning: Not enough data in dataset to generate all samples.") # Log warning
                break

            # Prepare single image samples for generation
            low_res_img = low_res_b[0].unsqueeze(0).to(self.device) # LR image for conditioning
            up_scale_img = up_scale_b[0].unsqueeze(0).to(self.device) # Bicubic upscaled for reconstruction base
            original_img_cpu = original_b[0].cpu().numpy() # Original HR image (CPU, numpy)
            residual_img_for_recon = residual_b_cpu[0].unsqueeze(0).to(self.device) # Ground truth residual

            # This is a placeholder for how context features would be extracted for generation.
            # It needs to match how the model expects `condition` during its forward pass.
            # If the U-Net's forward pass during training is `model(x, t, condition=condition_features_list)`
            # then `condition_features_list` needs to be generated here too.
            # For simplicity, we'll assume `generate_residuals` handles this or takes raw `low_res_img`.
            # You might need to pass `context_extractor` to `ResidualGenerator` or call it here.
            # Example if context_extractor is available:
            # with torch.no_grad():
            #     _, condition_features_list_gen = context_extractor(low_res_img, get_fea=True)

            # Generate residual using the generator
            # The `generate_residuals` method needs to be adapted if it requires explicit context features.
            # Current `ResidualGenerator.generate_residuals` takes `model` and `low_resolution_image` (which might be the context).
            generated_residual = generator.generate_residuals(model=model, low_resolution_image=low_res_img, num_images=1)
            reconstructed_image = up_scale_img + generated_residual # Reconstruct HR image
            reconstructed_image = torch.clamp(reconstructed_image, -1.0, 1.0) # Clamp to [-1, 1]
            reconstructed_image_norm = (reconstructed_image + 1.0) / 2.0 # Normalize to [0, 1] for logging
            reconstructed_image_np = reconstructed_image_norm.cpu().numpy().squeeze(0) # To CPU, numpy, (C,H,W)

            # Original image reconstructed from original residual (for comparison)
            original_reconstructed_image = up_scale_img + residual_img_for_recon # Reconstruct using GT residual
            original_reconstructed_image = torch.clamp(original_reconstructed_image, -1.0, 1.0) # Clamp
            original_reconstructed_image_norm = (original_reconstructed_image + 1.0) / 2.0 # Normalize
            original_reconstructed_image_np = original_reconstructed_image_norm.cpu().numpy().squeeze(0) # To CPU, numpy

            # Prepare images for logging (CHW format, [0,1] range)
            low_res_log = (low_res_img.squeeze(0).cpu().numpy() + 1.0) / 2.0 # LR image, normalized

            sample_images_data.append({
                "low_res": low_res_log, # (C,H_lr,W_lr), range [0,1]
                "generated_hr": reconstructed_image_np, # (C,H_hr,W_hr), range [0,1]
                "original_hr": (original_img_cpu + 1.0) / 2.0, # (C,H_hr,W_hr), range [0,1]
                "reconstructed_original_hr": original_reconstructed_image_np # (C,H_hr,W_hr), range [0,1]
            })

        # Log images to TensorBoard
        for i, imgs_dict in enumerate(sample_images_data):
            writer.add_image(f'Sample_{i}/01_LowRes_Input', imgs_dict["low_res"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/02_Generated_HR', imgs_dict["generated_hr"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/03_Original_HR_GroundTruth', imgs_dict["original_hr"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/04_Reconstructed_from_GT_Residual', imgs_dict["reconstructed_original_hr"], epoch + 1, dataformats='CHW')
        print(f"Logged {len(sample_images_data)} sample images to TensorBoard.") # Log completion


    def train(self,
                dataset: DataLoader, # Type hint for dataset
                model: torch.nn.Module, # Type hint for model
                context_extractor:torch.nn.Module, # Type hint for context_extractor
                optimizer, # Optimizer instance
                scheduler=None, # Optional: Learning rate scheduler
                accumulation_steps=1, # Gradient accumulation steps
                epochs=100, # Total number of epochs
                start_epoch=0, # Starting epoch (for resuming)
                best_loss=float('inf'), # Best loss recorded (for resuming)
                context_selection_mode="LR", # Context selection mode (e.g., "LR", "HR")
                log_dir_param=None, # Optional: Specific log directory for resuming
                checkpoint_dir_param=None, # Optional: Specific checkpoint directory for resuming
                log_dir_base="/media/hoangdv/cv_logs_diffusion", # Base directory for logs
                checkpoint_dir_base="/media/hoangdv/cv_checkpoints_diffusion" # Base directory for checkpoints
             ) -> None:

        model.to(self.device) # Move model to device
        context_extractor.to(self.device) # Move context_extractor to device
        context_extractor.eval() # Set context_extractor to evaluation mode

        # Setup directories and TensorBoard writer
        writer, checkpoint_dir, log_dir, best_checkpoint_path = self._setup_training_directories_and_writer(
            log_dir_base, checkpoint_dir_base, log_dir_param, checkpoint_dir_param
        )

        # Initialize step counters for training
        global_step_optimizer, batch_step_counter, current_accumulation_idx = self._initialize_training_steps(
            start_epoch, len(dataset), accumulation_steps # Use len(dataset) which is num_batches if drop_last=True
        )

        print(f"Starting training in '{self.mode}' mode on device: {self.device} with {accumulation_steps} accumulation steps.") # Log training start
        print(f"Logging to: {log_dir}") # Log logging directory
        print(f"Saving checkpoints to: {checkpoint_dir}") # Log checkpoint directory
        print(f"Initial global optimizer steps: {global_step_optimizer}, initial batch steps: {batch_step_counter}") # Log initial steps

        if start_epoch == 0: # Only zero_grad if not resuming an optimizer state that might have pending grads
             optimizer.zero_grad() # Initialize gradients to zero

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}") # Log current epoch
            model.train() # Set model to training mode
            progress_bar = tqdm(total=len(dataset), desc=f"Training ({self.mode}, Epoch {epoch+1})") # Initialize progress bar
            epoch_losses = [] # List to store losses for this epoch

            for batch_idx, batch_data in enumerate(dataset): # Iterate over batches
                loss_value, current_accumulation_idx, global_step_optimizer, updated_optimizer = self._perform_batch_step(
                    model, context_extractor, batch_data, context_selection_mode, optimizer,
                    accumulation_steps, current_accumulation_idx, global_step_optimizer
                )

                epoch_losses.append(loss_value) # Append batch loss
                writer.add_scalar(f'Loss_{self.mode}/batch_step_raw', loss_value, batch_step_counter) # Log raw batch loss
                if updated_optimizer: # Log LR if optimizer was updated
                    if optimizer.param_groups:
                         writer.add_scalar(f'LearningRate/optimizer_step', optimizer.param_groups[0]['lr'], global_step_optimizer)


                progress_bar.update(1) # Update progress bar
                progress_bar.set_postfix(loss=f"{loss_value:.4f}", opt_steps=global_step_optimizer, lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                batch_step_counter += 1 # Increment batch step counter

            # Handle end of epoch tasks
            best_loss = self._log_and_checkpoint_epoch_end(
                epoch, model, optimizer, scheduler, epoch_losses, best_loss, # Pass scheduler
                global_step_optimizer, best_checkpoint_path, writer, dataset # Pass dataset for sampling
            )
            progress_bar.close() # Close progress bar for the epoch

        # After all epochs, process remaining gradients (if any) from the last partial accumulation
        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} remaining accumulated gradients...") # Log final step
            optimizer.step() # Perform final optimizer step
            optimizer.zero_grad() # Zero gradients
            global_step_optimizer +=1 # Increment global optimizer step counter
            print(f"Final gradients applied. Total optimizer steps: {global_step_optimizer}") # Log completion

        writer.close() # Close TensorBoard writer
        print(f"Training finished for mode '{self.mode}'. Final best loss: {best_loss:.4f}") # Log training completion

    @staticmethod
    def load_model_weights(device, model, model_path, verbose=False):
        """
        Loads model weights from a saved checkpoint file.

        This function can load either a full checkpoint dictionary (extracting
        'model_state_dict') or a raw state_dict file. It handles partial loading
        (missing/unexpected keys) gracefully.

        Args:
            device (str or torch.device): The device to map the loaded weights to.
            model (torch.nn.Module): The PyTorch model instance to load weights into.
            model_path (str): Path to the model checkpoint file (.pth or .pt).
            verbose (bool, optional): If True, prints detailed information about
                                    missing and unexpected keys. Defaults to False.
        """
        if not os.path.exists(model_path):
            print(f"Warning: Model weights path not found: {model_path}. Model weights not loaded.") # Log warning
            return

        print(f"Loading model weights from: {model_path} onto device: {device}") # Log loading attempt
        # Load checkpoint, ensuring it's mapped to the specified device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False) # weights_only=False is safer for general checkpoints

        state_dict_to_load = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict_to_load = checkpoint["model_state_dict"]
                print("Found 'model_state_dict' in checkpoint.") # Log found key
            elif "state_dict" in checkpoint and "model" in checkpoint["state_dict"]: # common in some frameworks
                state_dict_to_load = checkpoint["state_dict"]["model"]
                print("Found 'state_dict' -> 'model' in checkpoint.") # Log found key
            else: # Checkpoint is likely a state_dict itself
                state_dict_to_load = checkpoint
                print("Checkpoint appears to be a raw state_dict.") # Log raw state_dict
                                
        elif isinstance(checkpoint, torch.nn.Module): # If the entire model was saved
            state_dict_to_load = checkpoint.state_dict()
            print("Checkpoint was a full model object; extracting state_dict.") # Log full model saved
        else:
            print(f"Warning: Checkpoint format not recognized or is not a dictionary/model. Type: {type(checkpoint)}") # Log warning
            return


        if state_dict_to_load:
            # Ensure model is on the correct device before loading state_dict
            model.to(device)
            incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False) # Load with strict=False
            if incompatible_keys.missing_keys:
                print(f"Warning: {len(incompatible_keys.missing_keys)} keys in the current model were not found in the checkpoint.") # Log missing keys
                if verbose: print(f"Missing keys: {incompatible_keys.missing_keys}") # Verbose log
            if incompatible_keys.unexpected_keys:
                print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in the checkpoint were not used by the current model.") # Log unexpected keys
                if verbose: print(f"Unused (unexpected) keys: {incompatible_keys.unexpected_keys}") # Verbose log

            num_loaded_params = sum(1 for k_model in model.state_dict() if k_model in state_dict_to_load and k_model not in incompatible_keys.unexpected_keys)
            print(f"Successfully loaded {num_loaded_params} compatible parameters into the model on device {device}.") # Log success
        else:
            print(f"Warning: Could not extract a loadable state_dict from {model_path}.") # Log failure

    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, scheduler, checkpoint_path, verbose_load=False): # Added scheduler and verbose_load
        """
        Loads a checkpoint for resuming training, including model state, optimizer state,
        scheduler state, epoch number, loss, global optimizer steps, and training mode.

        Args:
            device (str or torch.device): The device to load the model and checkpoint onto.
            model (torch.nn.Module): The model instance to load state into.
            optimizer (torch.optim.Optimizer): The optimizer instance to load state into.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler instance
                                                                        to load state into. Can be None.
            checkpoint_path (str): Path to the checkpoint file.
            verbose_load (bool, optional): If True, prints detailed info about missing/unexpected keys. Defaults to False.


        Returns:
            tuple:
                - int: `start_epoch` (the epoch to resume training from).
                - float: `loaded_loss` (the loss value from the checkpoint).
        """
        start_epoch = 0
        loaded_loss = float('inf')
        # global_optimizer_steps = 0 # This will be loaded from checkpoint if available
        # loaded_mode = None # This will be loaded from checkpoint if available

        model.to(device) # Ensure model is on the target device
        print(f"Attempting to load checkpoint for resume from: {checkpoint_path} onto device: {device}") # Log attempt

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint file not found at {checkpoint_path}. Training will start from scratch.") # Log not found
            return start_epoch, loaded_loss

        try:
            # Load checkpoint, ensuring it's mapped to the specified device
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False for complex checkpoints
            print(f"Checkpoint dictionary loaded successfully from {checkpoint_path}.") # Log success

            # Load model state
            if 'model_state_dict' in checkpoint:
                incompatible_model_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if incompatible_model_keys.missing_keys: print(f"Resume Warning: Missing keys in model state_dict: {incompatible_model_keys.missing_keys}")
                if incompatible_model_keys.unexpected_keys: print(f"Resume Info: Unexpected keys in model state_dict from checkpoint: {incompatible_model_keys.unexpected_keys}")
                if verbose_load and (incompatible_model_keys.missing_keys or incompatible_model_keys.unexpected_keys):
                    print(f"Verbose Model Load Details: Missing={len(incompatible_model_keys.missing_keys)}, Unexpected={len(incompatible_model_keys.unexpected_keys)}")
                print(f"Model state loaded successfully for resume.") # Log model state load
            else:
                print("Warning: 'model_state_dict' not found in checkpoint. Model weights not loaded for resume.") # Log warning

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Ensure optimizer states are on the correct device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print(f"Optimizer state loaded successfully for resume and moved to {device}.") # Log optimizer state load
                except Exception as optim_load_err:
                     print(f"Error loading optimizer state for resume: {optim_load_err}. Optimizer will start from scratch.") # Log error
            elif optimizer is None: print("Warning: Optimizer not provided, skipping optimizer state loading for resume.") # Log warning
            else: print("Warning: 'optimizer_state_dict' not found in checkpoint. Optimizer starts from scratch for resume.") # Log warning

            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"Scheduler state loaded successfully for resume.") # Log scheduler state load
                except Exception as sched_load_err:
                    print(f"Error loading scheduler state for resume: {sched_load_err}. Scheduler may start from scratch or use loaded optimizer LR.") # Log error
            elif scheduler is None: print("Info: Scheduler not provided, skipping scheduler state loading for resume.") # Log info
            else: print("Info: 'scheduler_state_dict' not found in checkpoint. Scheduler may start from scratch.") # Log info


            # Load epoch, loss, and other metadata
            if 'epoch' in checkpoint:
                saved_epoch = checkpoint['epoch']
                start_epoch = saved_epoch + 1 # Resume from the next epoch
                print(f"Resuming training from epoch: {start_epoch}") # Log resume epoch
            else: print("Warning: Epoch number not found in checkpoint. Starting from epoch 0 for resume.") # Log warning

            if 'loss' in checkpoint:
                loaded_loss = checkpoint['loss']
                print(f"Loaded loss from checkpoint for resume: {loaded_loss:.6f}") # Log loaded loss
            else: print("Info: Loss value not found in checkpoint. Using default best_loss for resume.") # Log info

            if 'global_optimizer_steps' in checkpoint:
                # global_optimizer_steps = checkpoint['global_optimizer_steps'] # This is handled by _initialize_training_steps
                print(f"Checkpoint saved with global_optimizer_steps: {checkpoint['global_optimizer_steps']}") # Log info
            if 'mode' in checkpoint:
                loaded_mode_from_ckpt = checkpoint['mode']
                print(f"Checkpoint was saved with training mode: '{loaded_mode_from_ckpt}'. Current trainer mode is '{model.mode if hasattr(model, 'mode') else 'N/A (model has no mode attr)'}'. Ensure consistency.") # Log mode info

        except Exception as e:
            print(f"Error loading checkpoint for resume: {e}. Training will start from scratch.") # Log error
            # Reset to default start values if full checkpoint load fails
            start_epoch = 0
            loaded_loss = float('inf')
            # global_optimizer_steps = 0 # Reset

        return start_epoch, loaded_loss


# --- ResidualGenerator Class (Copied from original, ensure it's consistent with your project structure) ---
# This class is used for generating samples during training validation.
# No direct changes for LR scheduler, but its usage in _generate_and_log_samples might need context.
class ResidualGenerator:
    """
    A class for generating image residuals using a pre-trained diffusion model and a scheduler.

    This class can be configured to work with models that predict either 'v' (v-prediction)
    or 'noise' (epsilon-prediction) by setting the `predict_mode` during
    initialization. It uses a `diffusers.SchedulerMixin` (like DDIMScheduler)
    to perform the reverse diffusion process.

    Attributes:
        img_channels (int): Number of channels in the image (e.g., 3 for RGB).
        img_size (int): Height and width of the image.
        device (str or torch.device): Device for tensor operations.
        num_train_timesteps (int): The number of timesteps the diffusion model was trained for.
                                   This is used to initialize the scheduler correctly.
        predict_mode (str): The prediction type the model is expected to output,
                            and how the scheduler should interpret it ("v_prediction" or "noise").
        betas (torch.Tensor): The beta schedule used for the scheduler.
        scheduler (diffusers.SchedulerMixin): The scheduler instance (e.g., DDIMScheduler)
                                              used for the denoising steps, configured according
                                              to `predict_mode`.
    """
    def __init__(self,
                 img_channels=3,
                 img_size=256, # Default, will be overridden if inferred
                 device='cuda',
                 num_train_timesteps=1000,
                 predict_mode='v_prediction'):
        """
        Initializes the ResidualGenerator.

        Sets up image parameters, device, and a DDIMScheduler. The scheduler is
        configured based on the `predict_mode` (either "v_prediction"
        or "noise") and uses a cosine beta schedule.

        Args:
            img_channels (int, optional): Number of image channels. Defaults to 3.
            img_size (int, optional): Size (height and width) of the image. Defaults to 256.
            device (str or torch.device, optional): Device for computations. Defaults to 'cuda'.
            num_train_timesteps (int, optional): Number of training timesteps for the
                                                 diffusion model this generator will use.
                                                 Defaults to 1000.
            predict_mode (str, optional): Specifies what the diffusion model is expected to predict.
                                          Can be "v_prediction" or "noise".
                                          Defaults to "v_prediction".
        Raises:
            ValueError: If `predict_mode` is not "v_prediction" or "noise".
        """
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        if predict_mode not in ["v_prediction", "noise"]:
            raise ValueError("Prediction mode must be 'v_prediction' or 'noise'") # Validate mode
        self.predict_mode = predict_mode

        # Helper function to generate cosine beta schedule
        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) # Clip betas

        self.betas = cosine_beta_schedule(self.num_train_timesteps).to(self.device)

        # Initialize the DDIM scheduler based on the predict_mode
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=self.betas.cpu().numpy(), # DDIMScheduler might expect numpy array
            beta_schedule="trained_betas", # Indicate use of pre-computed betas
            prediction_type=self.predict_mode, # Crucial: sets how scheduler interprets model output
            clip_sample=False, # Manual clipping is often preferred
            set_alpha_to_one=False, # Standard for cosine schedules
            steps_offset=1, # Common setting
        )
        print(f"ResidualGenerator initialized with {type(self.scheduler).__name__}, "
              f"configured for model prediction_type='{self.predict_mode}'. "
              f"Image size: {self.img_size}x{self.img_size}, Channels: {self.img_channels}.") # Log init

    @torch.no_grad()
    def generate_residuals(self, model, low_resolution_image, num_images=1, num_inference_steps=50):
        """
        Generates image residuals using the provided diffusion model and the configured scheduler.

        The model's output (either 'v' or 'noise') should match the
        `predict_mode` this ResidualGenerator was initialized with.

        Args:
            model (torch.nn.Module): The pre-trained diffusion model (e.g., a U-Net).
                                     It should accept `x_t` (noisy image), `t` (timestep),
                                     and potentially `condition` (e.g., from `low_resolution_image`)
                                     as input.
            low_resolution_image (torch.Tensor): The low-resolution image to condition the generation on.
                                                 Shape: [batch_size, img_channels, H_lr, W_lr].
                                                 This is used to form the `condition` for the model.
            num_images (int, optional): Number of images/residuals to generate. Defaults to 1.
                                        Must match the batch size of `low_resolution_image`.
            num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.

        Returns:
            torch.Tensor: A batch of generated residuals.
                          Shape: [num_images, img_channels, img_size, img_size].
        """
        model.eval() # Set model to evaluation mode
        model.to(self.device) # Ensure model is on correct device

        if num_images != low_resolution_image.shape[0]:
            # This can happen if low_resolution_image is a single sample expanded for batch,
            # or if num_images is meant to control multiple generations from the *same* LR image.
            # For now, assume low_resolution_image batch size dictates num_images.
            # print(f"Warning: num_images ({num_images}) differs from low_resolution_image batch_size ({low_resolution_image.shape[0]}). Using LR batch_size.")
            num_images = low_resolution_image.shape[0]


        print(f"Generating {num_images} residuals using {num_inference_steps} steps "
              f"with {type(self.scheduler).__name__} (model expected to predict: '{self.predict_mode}') "
              f"on device {self.device}...") # Log generation start

        self.scheduler.set_timesteps(num_inference_steps, device=self.device) # Set scheduler timesteps

        # Initialize with random noise (latent space representation for the residual)
        image_latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size), # Shape for residual
            device=self.device
        )
        image_latents = image_latents * self.scheduler.init_noise_sigma # Scale initial noise

        # Prepare condition from low_resolution_image.
        # This step is CRUCIAL and depends on how your `model` and `context_extractor` work.
        # If `model` expects features from a `context_extractor`:
        # 1. You need access to the `context_extractor` here.
        # 2. Extract features: `_, condition_features = context_extractor(low_resolution_image.to(self.device), get_fea=True)`
        # Then `condition_features` would be passed to `model`.
        # For this example, we assume `model` can take `low_resolution_image` directly as part of its `condition`
        # or that the `model`'s forward pass in `Unet` handles this if `condition` is `low_resolution_image`.
        # The `train_diffusion.py` script passes `condition_features_list` from an RRDBNet.
        # A similar extraction is needed here if the U-Net expects that.
        # For simplicity, let's assume the U-Net's forward in `diffusion_modules.py` can handle
        # `condition` being the `low_resolution_image` directly or pre-extracted features.
        # If `model` is the U-Net, and its forward is `forward(self, x, time, condition)`,
        # then `condition` must be prepared appropriately.
        #
        # Placeholder for actual conditioning logic:
        # This would typically involve your context_extractor (e.g., RRDBNet)
        # For now, we pass low_resolution_image as the `condition` argument,
        # assuming the model (U-Net) is structured to use it or its features.
        # If your Unet expects a list of feature maps like in training, you'd need to replicate that here.
        # This is a simplification for the generator.
        condition_input_for_model = low_resolution_image.to(self.device)


        # Iteratively denoise the latents
        for t_step in tqdm(self.scheduler.timesteps, desc="Generating residuals"):
            model_input = image_latents # Current noisy latents for the residual

            t_for_model = t_step.unsqueeze(0).expand(num_images).to(self.device) # Timestep for the batch

            # Model predicts based on its training (either 'v' or 'noise' for the residual)
            # The `condition` argument here needs to match what the U-Net expects.
            # If U-Net's `forward` is `(self, x, time, condition)` where `condition` is a list of features,
            # then `condition_input_for_model` must be that list.
            # This is a common point of mismatch if not handled carefully.
            # For now, assuming it's passed as `condition`.
            model_output = model(model_input, t_for_model, condition=condition_input_for_model) # Pass condition

            # Scheduler step to compute the previous noisy sample
            scheduler_output = self.scheduler.step(model_output, t_step, image_latents)
            image_latents = scheduler_output.prev_sample # Update latents

        generated_residuals = image_latents # Final denoised latents are the generated residuals

        print("Residual generation complete.") # Log completion
        return generated_residuals

if __name__ == "__main__":
    # Example usage for ResidualGenerator (illustrative)
    # This requires a dummy model and a dummy low_resolution_image for testing.
    print("ResidualGenerator example usage (illustrative):")
    try:
        # Create a dummy Unet-like model for the example
        class DummyUnet(torch.nn.Module):
            def __init__(self, in_channels=3, out_channels=3, base_dim=32):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                # A very simple conv layer to simulate the network
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                print(f"DummyUnet initialized with in_channels={in_channels}, out_channels={out_channels}")

            def forward(self, x, time, condition=None): # Added condition
                # x: noisy residual, time: timestep, condition: LR image or its features
                # This dummy model just passes x through a conv layer, ignoring time and condition for simplicity.
                # A real Unet would use time embeddings and incorporate condition.
                # print(f"DummyUnet forward: x.shape={x.shape}, time.shape={time.shape}, condition.shape={condition.shape if condition is not None else 'None'}")
                return self.conv(x)

        img_c = 3
        hr_size = 160 # Example HR size
        lr_size = hr_size // 4 # Example LR size (if downscale_factor=4)

        generator = ResidualGenerator(
            img_channels=img_c,
            img_size=hr_size, # Residuals are generated at HR size
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_train_timesteps=1000,
            predict_mode='v_prediction'
        )
        # Dummy Unet model instance
        unet_example = DummyUnet(in_channels=img_c, out_channels=img_c).to(generator.device)

        # Dummy low-resolution image (batch size 1)
        # This would be the input for conditioning the generation of the residual
        # Its size should match what the context_extractor or Unet expects for conditioning
        dummy_low_res_image = torch.randn(1, img_c, lr_size, lr_size).to(generator.device)

        print(f"Generating residuals with dummy model. LR image shape: {dummy_low_res_image.shape}")
        # Generate residuals. The `condition` for the Unet is derived from `dummy_low_res_image`.
        generated_residuals_example = generator.generate_residuals(
            model=unet_example,
            low_resolution_image=dummy_low_res_image, # This is the conditioning input
            num_images=1, # Should match batch size of low_res_image
            num_inference_steps=10 # Fewer steps for quick example
        )
        print(f"Generated residuals shape: {generated_residuals_example.shape}") # Expected: [1, img_c, hr_size, hr_size]
        assert generated_residuals_example.shape == (1, img_c, hr_size, hr_size)
        print("ResidualGenerator example completed successfully.")

    except Exception as e:
        print(f"Error in ResidualGenerator example: {e}") # Log error
