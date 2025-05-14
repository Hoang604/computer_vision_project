import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard
import numpy as np
from tqdm import tqdm
import os
import bitsandbytes as bnb
import datetime
from diffusers import DDIMScheduler # Keep this import
from torch.utils.data import DataLoader
from diffusion_modules import Unet

class DiffusionTrainer:
    """
    Implements a Denoising Diffusion Probabilistic Model (DDPM).

    This class handles trainning logic. It uses a cosine
    noise schedule by default. The model can be trained to predict either
    the noise added during the forward process or a 'v-prediction' target.

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
        Initializes the DiffusionModel with a specified number of timesteps, device, and prediction mode.

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
            raise ValueError("Mode must be 'v_prediction' or 'noise'")
        self.mode = mode
        print(f"DiffusionModel initialized in '{self.mode}' mode.")


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
        x_0 = x_0.float()

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
        if log_dir_param is None:
            log_dir = os.path.join(log_dir_base, f"{self.mode}_{timestamp}")
        else:
            log_dir = log_dir_param # Use the provided log_dir to resume

        if checkpoint_dir_param is None:
            checkpoint_dir = os.path.join(checkpoint_dir_base, f"{self.mode}_{timestamp}")
        else:
            checkpoint_dir = checkpoint_dir_param # Use the provided checkpoint_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        best_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_model_{self.mode}_best.pth')

        return writer, checkpoint_dir, log_dir, best_checkpoint_path

    def _initialize_training_steps(self, start_epoch, dataset_len, accumulation_steps):
        """Initialize step counters for resuming training."""
        # Calculate effective batches per epoch considering accumulation steps
        effective_batches_per_epoch = dataset_len // accumulation_steps
        if dataset_len % accumulation_steps != 0:
            effective_batches_per_epoch += 1

        global_step_optimizer = start_epoch * effective_batches_per_epoch
        batch_step_counter = start_epoch * dataset_len
        current_accumulation_idx = 0
        return global_step_optimizer, batch_step_counter, current_accumulation_idx

    def _get_training_target(self, noise_added, original_residual_batch, t):
        """Determine the model target based on self.mode."""
        if self.mode == "v_prediction":
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, original_residual_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, original_residual_batch.shape)
            target = sqrt_alphas_cumprod_t * noise_added - sqrt_one_minus_alphas_cumprod_t * original_residual_batch
        elif self.mode == "noise":
            target = noise_added
        else:
            raise ValueError(f"Unsupported training mode: {self.mode}")
        return target

    def _perform_batch_step(self, model, context_extractor, batch_data, context_selection_mode, optimizer,
                            accumulation_steps, current_accumulation_idx, global_step_optimizer):
        """Process a training batch: forward pass, loss calculation, backward pass, and optimizer step (if enough accumulation)."""
        low_res_image_batch, _, _, residual_image_batch = batch_data

        low_res_image_batch = low_res_image_batch.to(self.device)
        _, condition_features_list = context_extractor(low_res_image_batch)

        residual_image_batch = residual_image_batch.to(self.device)

        actual_batch_size = residual_image_batch.shape[0]
        t = torch.randint(0, self.timesteps, (actual_batch_size,), device=self.device, dtype=torch.long)
        residual_image_batch_t, noise_added = self.q_sample(residual_image_batch, t)

        target = self._get_training_target(noise_added, residual_image_batch, t)

        predicted_output = model(residual_image_batch_t, t, context=condition_features_list)
        loss = F.mse_loss(predicted_output, target)
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        current_accumulation_idx += 1
        updated_optimizer_this_step = False
        if current_accumulation_idx >= accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            current_accumulation_idx = 0
            global_step_optimizer +=1
            updated_optimizer_this_step = True

        return loss.detach().item(), current_accumulation_idx, global_step_optimizer, updated_optimizer_this_step

    def _log_and_checkpoint_epoch_end(self, epoch, model, optimizer, epoch_losses, current_best_loss,
                                        global_step_optimizer, best_checkpoint_path, writer, dataset):
        """Handle end-of-epoch tasks: log loss, save checkpoint, generate sample images."""
        mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"Epoch {epoch+1} Average Loss ({self.mode}): {mean_loss:.4f}")
        writer.add_scalar(f'Loss_{self.mode}/epoch', mean_loss, epoch + 1)

        new_best_loss = current_best_loss
        if mean_loss < current_best_loss:
            new_best_loss = mean_loss
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': new_best_loss,
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Mode: {self.mode}, OptSteps: {global_step_optimizer})")

        if (epoch + 1) % 5 == 0: # Generate sample images every 5 epochs
            self._generate_and_log_samples(model, dataset, epoch, writer)

        return new_best_loss

    def _generate_and_log_samples(self, model, dataset, epoch, writer):
        """Generate sample images and log to TensorBoard."""

        print(f"Generating sample images for TensorBoard at epoch {epoch+1}...")
        try:
            generator = ResidualGenerator(
                img_channels=3,
                img_size=160,
                device=self.device,
                num_train_timesteps=self.timesteps,
                predict_mode=self.mode
            )
        except NameError:
            print("Warning: ResidualGenerator class not found. Skipping sample generation.")
            return

        sample_images_data = [] # Renamed to avoid confusion with outer variable
        # Get a few samples from the dataset to generate images
        num_samples_to_generate = 3
        dataset_iter = iter(dataset)
        for _ in range(num_samples_to_generate):
            try:
                low_res_b, up_scale_b, original_b, residual_b_cpu = next(dataset_iter)
            except StopIteration:
                print("Warning: Not enough data in dataset to generate all samples.")
                break

            low_res_img = low_res_b[0].unsqueeze(0).to(self.device)
            up_scale_img = up_scale_b[0].unsqueeze(0).to(self.device)
            original_img_cpu = original_b[0].cpu().numpy() # Keep on CPU, (C,H,W)
            residual_img_for_recon = residual_b_cpu[0].unsqueeze(0).to(self.device)


            generated_residual = generator.generate_residuals(model=model, low_resolution_image=low_res_img, num_images=1)
            reconstructed_image = up_scale_img + generated_residual
            reconstructed_image = torch.clamp(reconstructed_image, -1.0, 1.0)
            reconstructed_image_norm = (reconstructed_image + 1.0) / 2.0 # Normalize to [0, 1]
            reconstructed_image_np = reconstructed_image_norm.cpu().numpy().squeeze(0) # (C,H,W)

            # Original image reconstructed from original residual (for comparison)
            original_reconstructed_image = up_scale_img + residual_img_for_recon
            original_reconstructed_image = torch.clamp(original_reconstructed_image, -1.0, 1.0)
            original_reconstructed_image_norm = (original_reconstructed_image + 1.0) / 2.0
            original_reconstructed_image_np = original_reconstructed_image_norm.cpu().numpy().squeeze(0) # (C,H,W)

            # Prepare images for logging (CHW format, [0,1] range or [0,255] uint8)
            low_res_log = (low_res_img.squeeze(0).cpu().numpy() + 1.0) / 2.0

            sample_images_data.append({
                "low_res": low_res_log, # (C,H,W), range [0,1]
                "generated_hr": reconstructed_image_np, # (C,H,W), range [0,1]
                "original_hr": (original_img_cpu + 1.0) / 2.0, # (C,H,W), range [0,1]
                "reconstructed_original_hr": original_reconstructed_image_np # (C,H,W), range [0,1]
            })

        for i, imgs_dict in enumerate(sample_images_data):
            writer.add_image(f'Sample_{i}/01_LowRes', imgs_dict["low_res"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/02_Generated_HR', imgs_dict["generated_hr"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/03_Original_HR_from_dataset', imgs_dict["original_hr"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/04_Reconstructed_from_GT_Residual', imgs_dict["reconstructed_original_hr"], epoch + 1, dataformats='CHW')


    def train(self,
                dataset: DataLoader,
                model: torch.nn.Module,
                context_extractor:torch.nn.Module,
                optimizer,
                accumulation_steps=32, epochs=30,
                start_epoch=0, best_loss=float('inf'),
                context_selection_mode="LR", # Renamed `context` to avoid confusion
                log_dir_param=None,
                checkpoint_dir_param=None,
                log_dir_base="/media/hoangdv/cv_logs",
                checkpoint_dir_base="/media/hoangdv/cv_checkpoints") -> None:

        model.to(self.device)
        context_extractor.to(self.device)
        context_extractor.eval()

        writer, checkpoint_dir, log_dir, best_checkpoint_path = self._setup_training_directories_and_writer(
            log_dir_base, checkpoint_dir_base, log_dir_param, checkpoint_dir_param
        )

        global_step_optimizer, batch_step_counter, current_accumulation_idx = self._initialize_training_steps(
            start_epoch, len(dataset), accumulation_steps
        )

        print(f"Starting training in '{self.mode}' mode on device: {self.device} with {accumulation_steps} accumulation steps.")
        print(f"Logging to: {log_dir}") # log_dir is already determined in _setup
        print(f"Saving checkpoints to: {checkpoint_dir}") # checkpoint_dir as well
        print(f"Initial global optimizer steps: {global_step_optimizer}, initial batch steps: {batch_step_counter}")

        optimizer.zero_grad()

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            progress_bar = tqdm(total=len(dataset), desc=f"Training ({self.mode}, Epoch {epoch+1})")
            epoch_losses = []

            for batch_data in dataset:
                loss_value, current_accumulation_idx, global_step_optimizer, _ = self._perform_batch_step(
                    model, context_extractor, batch_data, context_selection_mode, optimizer,
                    accumulation_steps, current_accumulation_idx, global_step_optimizer
                )

                epoch_losses.append(loss_value)
                writer.add_scalar(f'Loss_{self.mode}/batch_step', loss_value, batch_step_counter) # Log by batch_step_counter

                progress_bar.update(1)
                progress_bar.set_description(f"Mode: {self.mode} Loss: {loss_value:.4f} OptSteps: {global_step_optimizer}")
                batch_step_counter += 1 # Increment batch_step_counter after each batch

            # Handle end of epoch
            best_loss = self._log_and_checkpoint_epoch_end(
                epoch, model, optimizer, epoch_losses, best_loss,
                global_step_optimizer, best_checkpoint_path, writer, dataset
            )
            progress_bar.close()

        # After all epochs, process remaining gradients (if any)
        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} remaining accumulated gradients...")
            optimizer.step()
            optimizer.zero_grad()
            global_step_optimizer +=1 # Increment global_step_optimizer for the final step
            print(f"Final gradients applied. Total optimizer steps: {global_step_optimizer}")

        writer.close()
        print(f"Training finished for mode '{self.mode}'.")

    @staticmethod
    def load_model_weights(device, model, model_path, verbose=False):
        """
        Loads model weights from a saved checkpoint file.

        This function can load either a full checkpoint dictionary (extracting
        'model_state_dict') or a raw state_dict file. It handles partial loading
        (missing/unexpected keys) gracefully.

        Args:
            model (torch.nn.Module): The PyTorch model instance to load weights into.
            model_path (str): Path to the model checkpoint file (.pth or .pt).
            verbose (bool, optional): If True, prints detailed information about
                                    missing and unexpected keys. Defaults to False.
        """
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            state_dict_to_load = None
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict_to_load = checkpoint["model_state_dict"]
                else: # Checkpoint is a state_dict itself
                    state_dict_to_load = checkpoint
                                
            else: # Loaded object is directly a state_dict
                state_dict_to_load = checkpoint

            if state_dict_to_load:
                incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False)
                if incompatible_keys.missing_keys:
                    print(f"Warning: {len(incompatible_keys.missing_keys)} keys in the current model were not found in the checkpoint.")
                    if verbose: print(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in the checkpoint were not used by the current model.")
                    if verbose: print(f"Unused (unexpected) keys: {incompatible_keys.unexpected_keys}")

                num_loaded_params = sum(1 for k in state_dict_to_load if k not in incompatible_keys.unexpected_keys and k in model.state_dict())
                print(f"Weights loaded from {model_path}")
                print(f"Successfully loaded {num_loaded_params} compatible parameters into the model.")
            else:
                print(f"Warning: Could not extract state_dict from {model_path}.")

        else:
            print(f"Warning: Model weights path not found: {model_path}. Model weights not loaded.")

    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, checkpoint_path):
        """
        Loads a checkpoint for resuming training, including model state, optimizer state,
        epoch number, loss, global optimizer steps, and potentially the training mode.

        If the checkpoint contains a 'mode', it will be printed but not directly set on
        the model instance by this static method. The `DiffusionModel` instance should
        be initialized with the correct mode, or `load_model_weights` (which is an
        instance method) can update the mode if loading weights separately.

        Args:
            device (str or torch.device): The device to load the model and checkpoint onto.
            model (torch.nn.Module): The model instance to load state into.
            optimizer (torch.optim.Optimizer or bnb.optim.Adam8bit): The optimizer instance
                                                                    to load state into.
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            tuple:
                - int: `start_epoch` (the epoch to resume training from).
                - float: `loaded_loss` (the loss value from the checkpoint).
        """
        start_epoch = 0
        loaded_loss = float('inf')
        global_optimizer_steps = 0
        loaded_mode = None # To store the mode from checkpoint

        model.to(device)
        print(f"Ensuring model is on device: {device} for checkpoint loading.")

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint for resume from: {checkpoint_path} directly onto {device}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print(f"Checkpoint dictionary loaded successfully to {device} memory.")

                if 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys: print(f"\nWarning: Missing keys in model state_dict: {missing_keys}")
                    if unexpected_keys: print(f"\nInfo: Unexpected keys in model state_dict from checkpoint: {unexpected_keys}")
                    print(f"Model state loaded successfully onto model on {device}.")
                else:
                    print("Warning: 'model_state_dict' not found in checkpoint. Model weights not loaded.")

                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                        print(f"Optimizer state loaded successfully and moved to {device}.")
                    except Exception as optim_load_err:
                         print(f"Error loading optimizer state: {optim_load_err}. Optimizer will start from scratch.")
                elif optimizer is None: print("Warning: Optimizer not provided, skipping optimizer state loading.")
                else: print("Warning: 'optimizer_state_dict' not found in checkpoint. Optimizer starts from scratch.")

                if 'epoch' in checkpoint:
                    saved_epoch = checkpoint['epoch']
                    start_epoch = saved_epoch + 1
                    print(f"Resuming training from epoch: {start_epoch}")
                else: print("Warning: Epoch number not found in checkpoint. Starting from epoch 0.")

                if 'loss' in checkpoint:
                    loaded_loss = checkpoint['loss']
                    print(f"Loaded loss from checkpoint: {loaded_loss:.6f}")
                else: print("Info: Loss value not found in checkpoint. Using default best_loss.")

            except Exception as e:
                print(f"Error loading checkpoint: {e}. Training will start from scratch.")
                model.to(device)
                start_epoch = 0; loaded_loss = float('inf'); global_optimizer_steps = 0; loaded_mode = None
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Training will start from scratch.")
            model.to(device)
            start_epoch = 0; loaded_loss = float('inf'); global_optimizer_steps = 0; loaded_mode = None

        return start_epoch, loaded_loss

class ResidualGenerator:
    """
    A class for generating images using a pre-trained diffusion model and a scheduler.

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
                 img_size=256, 
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
            img_size (int, optional): Size (height and width) of the image. Defaults to 32.
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
            raise ValueError("Prediction mode must be 'v_prediction' or 'noise'")
        self.predict_mode = predict_mode

        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            # Helper function to generate cosine beta schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999)

        self.betas = cosine_beta_schedule(self.num_train_timesteps).to(self.device)

        # Initialize the DDIM scheduler based on the predict_mode
        # The `prediction_type` parameter of the scheduler is crucial.
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=self.betas.cpu().numpy(), # DDIMScheduler might expect numpy array for custom betas
            beta_schedule="trained_betas", # Indicate that we are providing pre-computed betas
            prediction_type=self.predict_mode, # This directly uses the chosen mode
            clip_sample=False, # Typically set to False for v-prediction, final clipping is manual.
                               # For noise prediction, schedulers often handle clipping if set to True,
                               # but False + manual clamp is also a common pattern.
            set_alpha_to_one=False, # Standard for cosine schedules
            steps_offset=1, # A common setting for many schedulers
        )
        # Corrected print statement to reflect the actual configured mode
        print(f"ResidualGenerator initialized with {type(self.scheduler).__name__}, "
              f"configured for model prediction_type='{self.predict_mode}'.")

    @torch.no_grad()
    def generate_residuals(self, model, low_resolution_image, num_images=1, num_inference_steps=50):
        """
        Generates images using the provided diffusion model and the configured scheduler.

        The model's output (either 'v' or 'noise') should match the
        `predict_mode` this ResidualGenerator was initialized with. The model itself
        (e.g., a U-Net) does not need a '.mode' attribute; this generator handles
        the interpretation of its output based on `self.predict_mode`.

        Args:
            model (torch.nn.Module): The pre-trained diffusion model (e.g., a U-Net).
                                     It should accept `x_t` (noisy image) and `t` (timestep)
                                     as input and output a tensor corresponding to its
                                     training objective (either 'v' or 'noise').
            low_resolution_image (torch.Tensor): The low-resolution image to condition the generation on.
            Should be of shape [batch_size, img_channels, img_size, img_size].
            This image is used as context for the model.
            
            num_images (int, optional): Number of images to generate. Defaults to 1.
            num_inference_steps (int, optional): Number of denoising steps to perform.
                                                 Fewer steps lead to faster generation but
                                                 potentially lower quality. Defaults to 50.

        Returns:
            torch.Tensor: A batch of generated images, normalized to the [0, 1] range.
                          Shape: [num_images, img_channels, img_size, img_size].
        """
        model.eval() # Set the model to evaluation mode
        model.to(self.device) # Ensure model is on the correct device

        # Updated print statement to include the operating mode
        print(f"Generating {num_images} images using {num_inference_steps} steps "
              f"with {type(self.scheduler).__name__} (model expected to predict: '{self.predict_mode}') "
              f"on device {self.device}...")

        # Set the number of inference steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Initialize with random noise (latent space representation)
        # Shape: [batch_size, num_channels, height, width]
        image_latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device
        )

        # Scale the initial noise by the scheduler's init_noise_sigma
        # This is important for some schedulers to ensure the noise is at the right magnitude
        image_latents = image_latents * self.scheduler.init_noise_sigma

        # Iteratively denoise the latents
        for t_step in tqdm(self.scheduler.timesteps, desc="Generating images"):
            # Prepare model input: current noisy latents
            model_input = image_latents

            # The model needs the current latents and the timestep `t_step`
            # Ensure t_step is correctly shaped for the model [batch_size]
            t_for_model = t_step.unsqueeze(0).expand(num_images).to(self.device) # Expand to batch size

            # Model predicts based on its training (either 'v' or 'noise')
            model_output = model(model_input, t_for_model, context=low_resolution_image)

            # Use the scheduler's step function to compute the previous noisy sample
            # The scheduler will interpret `model_output` based on its `prediction_type`
            # (which was set from `self.predict_mode` during initialization).
            scheduler_output = self.scheduler.step(model_output, t_step, image_latents)
            image_latents = scheduler_output.prev_sample

        generated_residuals = image_latents

        print("Image generation complete.")
        return generated_residuals

if __name__ == "__main__":
    # Example usage ResidualGenerator
    generator = ResidualGenerator()
    unet = Unet(base_dim=32)
    low_resolution_image = torch.randn(1, 3, 256, 256) # Example low-res image
    low_resolution_image = low_resolution_image.to('cuda') # Move to GPU if available)
    res = generator.generate_images(unet, low_resolution_image)
    print(res.shape)