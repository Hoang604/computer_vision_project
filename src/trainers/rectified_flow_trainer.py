import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
from torch.utils.data import DataLoader

# Assuming this import path is correct in your project structure
# from src.diffusion_modules.unet import Unet

class ImageGenerator:
    """
    Generates images using a pre-trained Rectified Flow model.

    This class is designed for inference and is decoupled from the training logic,
    providing a clean interface for image generation by solving the probability flow ODE.
    """
    def __init__(self, img_channels=3, img_size=160, device='cuda'):
        """
        Initializes the ImageGenerator.

        Args:
            img_channels (int): Number of channels in the image (e.g., 3 for RGB).
            img_size (int): The height and width of the image.
            device (str or torch.device): The device on which to perform computations.
        """
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        print(f"Rectified Flow ImageGenerator initialized on device {self.device} for image size {img_size}x{img_size}.")

    @torch.no_grad()
    def generate_images(self, model, features, num_images=1, num_inference_steps=100, initial_noise=None):
        """
        Generates images by solving the Ordinary Differential Equation (ODE) with the Euler method.

        This process starts from a noise vector z0 (source distribution) and iteratively
        integrates the velocity field predicted by the model to arrive at the final image z1
        (target distribution).

        Args:
            model (torch.nn.Module): The pre-trained U-Net model.
            features (list[torch.Tensor] or None): A list of conditioning feature tensors.
            num_images (int): The number of images to generate.
            num_inference_steps (int): The number of steps for the ODE solver.
            initial_noise (torch.Tensor, optional): An initial noise tensor to start the process.
                                                    If None, new random noise is generated.

        Returns:
            torch.Tensor: A batch of generated high-resolution images.
        """
        model.eval()
        model.to(self.device)

        # Start from random noise z0 if not provided
        if initial_noise is None:
            z = torch.randn((num_images, self.img_channels, self.img_size, self.img_size), device=self.device)
        else:
            z = initial_noise.to(self.device)

        dt = 1.0 / num_inference_steps

        # ODE solver loop
        for i in tqdm(range(num_inference_steps), desc="Solving ODE", leave=False):
            t = torch.full((num_images,), i * dt, device=self.device)
            # Predict the velocity at the current time t
            v = model(z, t, condition=features)
            # Update z with one Euler step
            z = z + v * dt

        return z

class RectifiedFlowTrainer:
    """
    Implements a trainer for Rectified Flow models.

    This class handles the training logic for learning a velocity field that transports
    samples from a source distribution (e.g., noise) to a target distribution (e.g., images).
    It supports both the initial 'rectified_flow' training and the 'reflow' procedure for
    further straightening the flow paths. Includes utilities for logging, checkpointing,
    validation, and sample generation.

    Attributes:
        device (str or torch.device): The device on which to perform computations ('cuda' or 'cpu').
        mode (str): The training mode, either "rectified_flow" or "reflow".
    """
    def __init__(self, device='cuda', mode='rectified_flow'):
        """
        Initializes the RectifiedFlowTrainer.

        Args:
            device (str or torch.device, optional): Device for tensor operations. Defaults to 'cuda'.
            mode (str, optional): The training mode. Can be 'rectified_flow' for standard training
                                  or 'reflow' for the second stage to straighten the flow.
                                  Defaults to 'rectified_flow'.
        """
        self.device = device
        if mode not in ['rectified_flow', 'reflow']:
            raise ValueError("Mode must be 'rectified_flow' or 'reflow'")
        self.mode = mode
        print(f"RectifiedFlowTrainer initialized in '{self.mode}' mode on device '{self.device}'.")

    def get_interpolated_sample(self, x0, x1, t):
        """Creates a linearly interpolated sample `x_t` between `x0` and `x1`.
        The formula used is: x_t = t * x1 + (1 - t) * x0.
        """
        t_reshaped = t.view(-1, 1, 1, 1)
        return t_reshaped * x1 + (1. - t_reshaped) * x0

    def _setup_training_directories_and_writer(self, log_dir_base, checkpoint_dir_base,
                                                log_dir_param, checkpoint_dir_param):
        """Set up log and checkpoint directories and TensorBoard writer."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{self.mode.capitalize()}_{timestamp}"

        if log_dir_param:
            log_dir = log_dir_param
            if checkpoint_dir_param:
                 experiment_name = os.path.basename(checkpoint_dir_param)
            else:
                 experiment_name = os.path.basename(log_dir_param)
        else:
            log_dir = os.path.join(log_dir_base, experiment_name)

        if checkpoint_dir_param:
            checkpoint_dir = checkpoint_dir_param
        else:
            checkpoint_dir = os.path.join(checkpoint_dir_base, experiment_name)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        best_checkpoint_path = os.path.join(checkpoint_dir, f'model_{os.path.basename(checkpoint_dir)}_best.pth')

        return writer, checkpoint_dir, log_dir, best_checkpoint_path

    def _initialize_training_steps(self, start_epoch, dataset_len):
        """Initialize step counters for resuming training."""
        batch_step_counter = start_epoch * dataset_len
        current_accumulation_idx = 0
        return batch_step_counter, current_accumulation_idx

    def _perform_batch_step(self, model, context_extractor, batch_data, optimizer,
                            accumulation_steps, is_training=True):
        """
        Performs a single batch step of training or validation.

        This function encapsulates the core logic of Rectified Flow training:
        1. Samples a pair of points (x0, x1) from the source and target distributions.
        2. Linearly interpolates between them to get an intermediate point `x_t`.
        3. Calculates the constant target velocity `v = x1 - x0`.
        4. Feeds `x_t`, timestep `t`, and optional conditioning features into the model
           to get the predicted velocity.
        5. Computes the loss (typically MSE) between the predicted and target velocities.
        If in training mode, it performs a backward pass and accumulates gradients. This
        logic is shared between 'rectified_flow' and 'reflow' modes, as the
        difference is handled by the data loader which provides the (x0, x1) pairs.

        Args:
            model (torch.nn.Module): The U-Net model being trained.
            context_extractor (torch.nn.Module or None): The model for extracting conditioning features.
            batch_data (tuple): A tuple containing the batch data, expected to be
                                (low_res_image, x0, x1).
            optimizer (torch.optim.Optimizer): The optimizer used for weight updates.
            accumulation_steps (int): Number of steps to accumulate gradients over.
            is_training (bool): Flag indicating if the model is in training phase. If True,
                                gradients are computed and back-propagated.

        Returns:
            float: The computed loss value for the current batch.
        """
        # Enable/disable gradient calculation based on the is_training flag.
        with torch.set_grad_enabled(is_training):
            # --- Step 1: Prepare Data ---
            # Unpack the data tuple and move necessary tensors to the compute device.
            # This logic is common for both modes as the dataloader always provides (low_res, x0, x1).
            low_res_image_batch, x0, x1 = batch_data
            low_res_image_batch = low_res_image_batch.to(self.device)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            actual_batch_size = x0.shape[0]

            # --- Step 2: Extract Conditioning Features (if applicable) ---
            # Conditional features (e.g., from a low-res image) are extracted
            # without gradient computation to save resources.
            condition_features_list = None
            if context_extractor is not None:
                with torch.no_grad():
                    _, raw_features_list_gpu = context_extractor(low_res_image_batch, get_fea=True)
                    condition_features_list = [feat.detach() for feat in raw_features_list_gpu]

            # --- Step 3: Compute Target and Interpolated Sample ---
            # The target velocity vector is the straight path from the start point x0 to the end point x1.
            target_velocity = x1 - x0

            # Sample a random time t from a uniform distribution [0, 1].
            t = torch.rand(actual_batch_size, device=self.device)

            # Create the interpolated sample xt = t*x1 + (1-t)*x0. This is the point
            # the model will receive as input to predict the velocity.
            xt = self.get_interpolated_sample(x0, x1, t)

            # --- Step 4: Forward Pass and Loss Calculation ---
            # Pass the interpolated sample xt, time t, and conditioning features to the model.
            predicted_velocity = model(xt, t, condition=condition_features_list)

            # The loss is calculated as the Mean Squared Error between the predicted and target velocity.
            loss = F.mse_loss(predicted_velocity, target_velocity)

        # --- Step 5: Backward Pass (only during training) ---
        if is_training:
            # Scale the loss by the number of accumulation steps to normalize the gradient.
            scaled_loss = loss / accumulation_steps
            # Backpropagate to compute gradients.
            scaled_loss.backward()

        # Return the detached loss value (as a float) for logging.
        return loss.detach().item()

    def _run_validation_epoch(self, model, context_extractor, val_loader, writer, epoch):
        """Run a validation epoch and return the average validation loss."""
        model.eval()
        if context_extractor: context_extractor.eval()
        total_val_loss = 0.0
        print(f"\nRunning validation for epoch {epoch+1}...")
        progress_bar_val = tqdm(total=len(val_loader), desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                loss_value = self._perform_batch_step(
                    model, context_extractor, batch_data,
                    optimizer=None, accumulation_steps=1, is_training=False
                )
                total_val_loss += loss_value
                progress_bar_val.update(1)
                progress_bar_val.set_postfix(val_loss_batch=f"{loss_value:.4f}")

        progress_bar_val.close()
        avg_val_loss = total_val_loss / len(val_loader) if val_loader else float('nan')
        print(f"Epoch {epoch+1} Average Validation Loss ({self.mode.capitalize()}): {avg_val_loss:.4f}")

        if writer:
            writer.add_scalar(f'Loss/{self.mode}_validation_epoch_avg', avg_val_loss, epoch + 1)
        return avg_val_loss

    def _log_and_checkpoint_epoch_end(self, epoch, model, optimizer, scheduler, train_epoch_losses,
                                        current_best_val_loss, val_loss_this_epoch,
                                        global_optimizer_steps, best_checkpoint_path, writer,
                                        dataset_for_samples, context_extractor):
        """Handle end-of-epoch tasks: log loss, save checkpoint, step scheduler, generate sample images."""
        mean_train_loss = np.mean(train_epoch_losses) if train_epoch_losses else float('nan')
        print(f"Epoch {epoch+1} Average Training Loss ({self.mode.capitalize()}): {mean_train_loss:.4f}")
        writer.add_scalar(f'Loss/{self.mode}_train_epoch_avg', mean_train_loss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate/epoch', current_lr, epoch + 1)
        print(f"Current Learning Rate at end of epoch {epoch+1}: {current_lr:.2e}")

        new_best_val_loss = current_best_val_loss
        loss_for_comparison = val_loss_this_epoch if val_loss_this_epoch is not None else mean_train_loss

        if not np.isnan(loss_for_comparison) and loss_for_comparison < current_best_val_loss:
            new_best_val_loss = loss_for_comparison
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': new_best_val_loss,
                'global_optimizer_steps': global_optimizer_steps,
                'mode': self.mode,
                'model_config': {
                    'base_dim': getattr(model, 'base_dim', None),
                    'out_dim': getattr(model, 'out_dim', None),
                    'dim_mults': getattr(model, 'dim_mults', None),
                    'cond_dim': getattr(model, 'cond_dim', None),
                    'rrdb_num_blocks': getattr(model, 'rrdb_num_blocks', 8),
                    'sr_scale': getattr(model, 'sr_scale', 4),
                    'use_attention': getattr(model, 'use_attention', False),
                    'use_weight_norm': getattr(model, 'use_weight_norm', True),
                    'weight_init': getattr(model, 'weight_init', True)
                }
            }
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Best Loss: {new_best_val_loss:.4f})")
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_loss_this_epoch if val_loss_this_epoch is not None else mean_train_loss
                if not np.isnan(metric):
                    scheduler.step(metric)
            else:
                scheduler.step()

        if dataset_for_samples:
            self._generate_and_log_samples(model, dataset_for_samples, epoch, writer, context_extractor)
        
        return new_best_val_loss

    def _generate_and_log_samples(self, model, dataset_loader, epoch, writer, context_extractor):
        """Generate sample images from the model and log them to TensorBoard."""
        print(f"Generating sample images for TensorBoard at epoch {epoch+1}...")
        model.eval()

        try:
            sample_batch_data = next(iter(dataset_loader))
            # The fourth element (residual) is not needed for Rectified Flow generation
            low_res_b, up_scale_b, original_b, _ = sample_batch_data
        except StopIteration:
            print("Warning: Not enough data in dataset_loader to generate samples.")
            model.train()
            return

        num_samples = min(3, low_res_b.shape[0])
        low_res_b = low_res_b[:num_samples].to(self.device)
        up_scale_b = up_scale_b[:num_samples].to(self.device)
        original_b = original_b[:num_samples].to(self.device)

        sample_condition_features = None
        if context_extractor:
            with torch.no_grad():
                _, raw_features_list_gpu_sample = context_extractor(low_res_b, get_fea=True)
                sample_condition_features = [feat.to(self.device) for feat in raw_features_list_gpu_sample]

        generator = ImageGenerator(
            img_channels=original_b.shape[1], 
            img_size=original_b.shape[2], 
            device=self.device
        )
        with torch.no_grad():
            generated_hr_batch = generator.generate_images(
                model=model,
                features=sample_condition_features,
                num_images=num_samples,
                num_inference_steps=50 # A reasonable number of steps for a sample
            )

        for i in range(num_samples):
            # Normalize images from [-1, 1] to [0, 1] for display
            lr_img = (low_res_b[i].cpu().numpy() + 1.0) / 2.0
            hr_rrdb = (up_scale_b[i].cpu().numpy() + 1.0) / 2.0
            hr_orig = (original_b[i].cpu().numpy() + 1.0) / 2.0
            pred_hr = (generated_hr_batch[i].cpu().numpy() + 1.0) / 2.0
            
            writer.add_image(f'Sample_{i}/01_LowRes_Input', np.clip(lr_img, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/02_HR_RRDB_Baseline', np.clip(hr_rrdb, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/03_Predicted_HR_{self.mode.capitalize()}', np.clip(pred_hr, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/04_Original_HR_GroundTruth', np.clip(hr_orig, 0, 1), epoch + 1, dataformats='CHW')

        print(f"Logged {num_samples} sample images to TensorBoard.")
        model.train()


    def train(self,
                train_dataset: DataLoader,
                model: torch.nn.Module,
                optimizer,
                scheduler=None,
                context_extractor: torch.nn.Module = None,
                val_dataset: DataLoader = None,
                pretrained_model_path: str = None,
                val_every_n_epochs: int = 1,
                accumulation_steps=1,
                epochs=100,
                start_epoch=0,
                best_loss=float('inf'),
                log_dir_param=None,
                checkpoint_dir_param=None,
                log_dir_base="./logs_rectified_flow",
                checkpoint_dir_base="./checkpoints_rectified_flow",
             ) -> None:
        """
        Main training loop for the Rectified Flow model.
        """
        model.to(self.device)
        if context_extractor:
            context_extractor.to(self.device)
            context_extractor.eval()
        
        if self.mode == 'reflow':
            if not pretrained_model_path:
                raise ValueError("pretrained_model_path must be provided for 'reflow' mode.")
            print(f"Reflow mode requires a pre-trained model. This will be handled by the data loader.")
            # The actual pre-trained model is used inside the ReflowDataset/DataLoader,
            # so we don't need to load it here in the trainer itself.

        writer, _, _, best_checkpoint_path = self._setup_training_directories_and_writer(
            log_dir_base, checkpoint_dir_base, log_dir_param, checkpoint_dir_param
        )
        batch_step_counter, current_accumulation_idx = self._initialize_training_steps(start_epoch, len(train_dataset))
        current_best_val_loss_tracker = best_loss
        global_optimizer_steps = start_epoch * (len(train_dataset) // accumulation_steps)

        print(f"Starting {self.mode.capitalize()} training on device: {self.device} with {accumulation_steps} accumulation steps.")
        if start_epoch == 0: optimizer.zero_grad()

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            progress_bar_train = tqdm(total=len(train_dataset), desc=f"Training ({self.mode.capitalize()}, Epoch {epoch+1})")
            train_epoch_losses = []

            for batch_idx, batch_data in enumerate(train_dataset):
                loss_value = self._perform_batch_step(
                    model, context_extractor, batch_data, optimizer,
                    accumulation_steps, is_training=True
                )
                train_epoch_losses.append(loss_value)
                writer.add_scalar(f'Loss/{self.mode}_train_batch_step_raw', loss_value, batch_step_counter)
                
                current_accumulation_idx += 1
                if current_accumulation_idx >= accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_accumulation_idx = 0
                    global_optimizer_steps += 1
                    writer.add_scalar('LearningRate/optimizer_step', optimizer.param_groups[0]['lr'], global_optimizer_steps)

                progress_bar_train.update(1)
                progress_bar_train.set_postfix(loss=f"{loss_value:.4f}", opt_steps=global_optimizer_steps)
                batch_step_counter += 1
            
            progress_bar_train.close()

            current_val_loss = None
            if val_dataset and (epoch + 1) % val_every_n_epochs == 0:
                current_val_loss = self._run_validation_epoch(model, context_extractor, val_dataset, writer, epoch)

            current_best_val_loss_tracker = self._log_and_checkpoint_epoch_end(
                epoch, model, optimizer, scheduler, train_epoch_losses,
                current_best_val_loss_tracker, current_val_loss,
                global_optimizer_steps, best_checkpoint_path, writer,
                val_dataset if val_dataset else train_dataset, context_extractor
            )

        writer.close()
        print(f"Training finished for mode '{self.mode}'. Final best tracked loss: {current_best_val_loss_tracker:.4f}")

    # ===================================================================
    #                STATIC UTILITY METHODS
    # ===================================================================
    @staticmethod
    def load_rectified_flow_unet(model_path: str, device: torch.device = torch.device("cuda"), verbose: bool = True) -> torch.nn.Module:
        """Load a UNet model from a checkpoint, automatically initializing architecture from config."""
        if verbose: print(f"Loading UNet model from {model_path}...")
        
        # Import UNet here to avoid circular import issues
        from src.diffusion_modules.unet import Unet 

        loaded_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_config' not in loaded_checkpoint:
            raise KeyError("Checkpoint missing 'model_config'. Cannot initialize UNet.")
        if 'model_state_dict' not in loaded_checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict'. Cannot load UNet weights.")

        model_config = loaded_checkpoint['model_config']
        model_state_dict = loaded_checkpoint['model_state_dict']
        
        unet = Unet(**model_config).to(device)
        unet.load_state_dict(model_state_dict, strict=True)
        
        if verbose: print("Model loaded successfully with strict key checking.")
        return unet
    
    @staticmethod
    def load_model_weights(model, model_path, verbose=False, device='cuda'):
        """Loads weights into an already-initialized model instance."""
        if not os.path.exists(model_path):
            print(f"Warning: Model weights path not found: {model_path}. Model weights not loaded.")
            return

        print(f"Loading model weights from: {model_path} onto device: {device}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        
        if state_dict_to_load:
            incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False)
            if incompatible_keys.missing_keys:
                print(f"Warning: {len(incompatible_keys.missing_keys)} keys in the model were not found in the checkpoint.")
                if verbose: print(f"Missing keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in the checkpoint were not used by the model.")
                if verbose: print(f"Unused keys: {incompatible_keys.unexpected_keys}")
            if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                print(f"Successfully loaded all parameters into the model.")
        else:
            print(f"Warning: Could not extract a loadable state_dict from {model_path}.")

    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, scheduler, checkpoint_path, verbose_load=False):
        """Loads a checkpoint for resuming training."""
        start_epoch = 0
        loaded_best_loss = float('inf')

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint file not found at {checkpoint_path}. Training will start from scratch.")
            return start_epoch, loaded_best_loss

        print(f"Attempting to load checkpoint for resume from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model weights
        RectifiedFlowTrainer.load_model_weights(model, checkpoint_path, verbose=verbose_load, device=device)

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and optimizer:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Ensure optimizer states are on the correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Error loading optimizer state: {e}. Optimizer will start from scratch.")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and scheduler:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded successfully.")
            except Exception as e:
                print(f"Error loading scheduler state: {e}. Scheduler may start from scratch.")

        start_epoch = checkpoint.get('epoch', -1) + 1
        loaded_best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"Resuming training from epoch: {start_epoch}")
        print(f"Loaded best loss from checkpoint: {loaded_best_loss:.6f}")

        return start_epoch, loaded_best_loss
