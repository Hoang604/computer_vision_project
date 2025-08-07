import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
from torch.utils.data import DataLoader # For type hinting

class RectifiedFlowTrainer:
    """
    Implements a trainer for a Rectified Flow model.

    This class handles the training logic for Rectified Flow. The model learns a 
    velocity field to approximate the straight path from a source point x0 
    (the low-resolution image) to a target point x1 (the high-resolution image).

    Attributes:
        device (str or torch.device): The device to perform computations on ('cuda' or 'cpu').
    """
    def __init__(self, device='cuda'):
        """
        Initializes the RectifiedFlowTrainer.

        Args:
            device (str or torch.device, optional): Device for tensor operations ('cuda' or 'cpu').
                                                   Defaults to 'cuda'.
        """
        self.device = device
        print(f"RectifiedFlowTrainer initialized on device '{self.device}'.")

    def _perform_batch_step(self, model, context_extractor, batch_data, optimizer,
                            accumulation_steps, current_accumulation_idx, global_step_optimizer, is_training=True):
        """
        Performs a single training or validation step for a batch.
        This function handles the forward pass, loss calculation, and optimizer step if in training mode.

        Args:
            model (torch.nn.Module): The Rectified Flow model (U-Net).
            context_extractor (torch.nn.Module, optional): Model for extracting conditional features.
            batch_data (tuple): The batch data: (low_res, upscaled_rrdb, original_hr).
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            accumulation_steps (int): Number of gradient accumulation steps.
            current_accumulation_idx (int): The current index in the accumulation cycle.
            global_step_optimizer (int): The global optimizer step count.
            is_training (bool): Flag indicating whether it's a training or validation step.

        Returns:
            tuple: Contains:
                - loss (float): The calculated loss value for the batch.
                - current_accumulation_idx (int): The updated accumulation index.
                - global_step_optimizer (int): The updated global optimizer step count.
                - updated_optimizer_this_step (bool): Flag indicating if the optimizer was updated in this step.
        """
        # Unpack batch data
        low_res_image_batch, _, original_hr_image_batch, _ = batch_data
        
        # Move data to the specified device
        low_res_image_batch = low_res_image_batch.to(self.device)
        original_hr_image_batch = original_hr_image_batch.to(self.device)

        # --- Extract conditional features (if applicable) ---
        condition_features_list = None
        if context_extractor is not None:
            context_extractor.eval()
            with torch.no_grad():
                _, raw_features_list_gpu = context_extractor(low_res_image_batch, get_fea=True)
                condition_features_list = [feat.to(self.device) for feat in raw_features_list_gpu]

        # --- Core Logic of Rectified Flow (LR -> HR) ---
        with torch.set_grad_enabled(is_training):
            # The target is the original HR image
            x1 = original_hr_image_batch

            # The starting point x0 is the LR image, upsampled to match x1's dimensions.
            # This upsampling provides a coarse starting point in the high-dimensional space.
            x0 = F.interpolate(low_res_image_batch, size=x1.shape[2:], mode='bilinear', align_corners=False)

            # 1. Sample random time t from [0, 1]
            t = torch.rand(x0.shape[0], device=self.device).view(-1, 1, 1, 1)

            # 2. Create the interpolated point xt on the ideal straight path
            # This will be the input to the model
            xt = (1.0 - t) * x0 + t * x1

            # 3. Calculate the target velocity
            # This is the "ground truth" that the model needs to predict
            target_velocity = x1 - x0

            # 4. Predict the velocity from the model
            # Note: the model needs to receive both xt and t
            predicted_velocity = model(xt, t.squeeze(), condition=condition_features_list)

            # 5. Calculate the loss
            # The loss is the mean squared error between the predicted and target velocities
            loss = F.mse_loss(predicted_velocity, target_velocity)

        # --- Gradient Accumulation and Optimizer Step (if training) ---
        updated_optimizer_this_step = False
        if is_training:
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            current_accumulation_idx += 1
            if current_accumulation_idx >= accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                current_accumulation_idx = 0
                global_step_optimizer += 1
                updated_optimizer_this_step = True

        return loss.detach().item(), current_accumulation_idx, global_step_optimizer, updated_optimizer_this_step

    # ==========================================================================================
    # UTILITY FUNCTIONS (Mostly kept from the original trainer for familiarity)
    # ==========================================================================================

    def _setup_training_directories_and_writer(self, log_dir_base, checkpoint_dir_base,
                                                log_dir_param, checkpoint_dir_param):
        """Sets up log/checkpoint directories and the TensorBoard writer."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"RectifiedFlow_{timestamp}"

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
        best_checkpoint_path = os.path.join(checkpoint_dir, f'rectified_flow_model_{os.path.basename(checkpoint_dir)}_best.pth')

        return writer, checkpoint_dir, log_dir, best_checkpoint_path

    def _initialize_training_steps(self, start_epoch, dataset_len, accumulation_steps):
        """Initializes step counters for resuming training."""
        effective_batches_per_epoch = dataset_len
        global_step_optimizer = start_epoch * effective_batches_per_epoch
        batch_step_counter = start_epoch * dataset_len
        current_accumulation_idx = 0
        return global_step_optimizer, batch_step_counter, current_accumulation_idx

    def _run_validation_epoch(self, model, context_extractor, val_loader, writer, epoch):
        """Runs a validation epoch and returns the average validation loss."""
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        print(f"\nRunning validation for epoch {epoch+1}...")
        progress_bar_val = tqdm(total=len(val_loader), desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                loss_value, _, _, _ = self._perform_batch_step(
                    model, context_extractor, batch_data,
                    optimizer=None,
                    accumulation_steps=1, current_accumulation_idx=0, global_step_optimizer=0,
                    is_training=False
                )
                total_val_loss += loss_value
                num_val_batches += 1
                progress_bar_val.update(1)
                progress_bar_val.set_postfix(val_loss_batch=f"{loss_value:.4f}")

        progress_bar_val.close()
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
        print(f"Epoch {epoch+1} Average Validation Loss (RectifiedFlow): {avg_val_loss:.4f}")

        if writer:
            writer.add_scalar(f'Loss_RectifiedFlow/validation_epoch_avg', avg_val_loss, epoch + 1)
        return avg_val_loss

    def _log_and_checkpoint_epoch_end(self, epoch, model, optimizer, scheduler, train_epoch_losses,
                                        current_best_val_loss,
                                        val_loss_this_epoch,
                                        global_step_optimizer, best_checkpoint_path, writer,
                                        dataset_for_samples, context_extractor):
        """Handles end-of-epoch tasks: logging, checkpointing, scheduler step, and sample generation."""
        mean_train_loss = np.mean(train_epoch_losses) if train_epoch_losses else float('nan')
        print(f"Epoch {epoch+1} Average Training Loss (RectifiedFlow): {mean_train_loss:.4f}")
        writer.add_scalar(f'Loss_RectifiedFlow/train_epoch_avg', mean_train_loss, epoch + 1)

        if optimizer.param_groups:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'LearningRate/epoch', current_lr, epoch + 1)

        new_best_val_loss = current_best_val_loss
        loss_for_comparison = val_loss_this_epoch if val_loss_this_epoch is not None else mean_train_loss

        if loss_for_comparison is not None and not np.isnan(loss_for_comparison) and loss_for_comparison < current_best_val_loss:
            new_best_val_loss = loss_for_comparison
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': new_best_val_loss,
                'global_optimizer_steps': global_step_optimizer,
                # Add model config info if needed
            }
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Best Loss: {new_best_val_loss:.4f})")

        # Save checkpoint for the current epoch
        current_epoch_checkpoint_path = os.path.join(os.path.dirname(best_checkpoint_path), f'rectified_flow_model_epoch_{epoch + 1}.pth')
        # ... (checkpoint saving logic is similar)
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_for_comparison)
            else:
                scheduler.step()

        if dataset_for_samples is not None:
            try:
                self._generate_and_log_samples(model, dataset_for_samples, epoch, writer, context_extractor)
            except Exception as e_sample:
                print(f"Error during sample generation: {e_sample}")

        return new_best_val_loss

    def _generate_and_log_samples(self, model, dataset_loader, epoch, writer, context_extractor):
        """Generates sample images and logs them to TensorBoard."""
        print(f"Generating sample images for TensorBoard at epoch {epoch+1}...")
        model.eval()

        try:
            generator = RectifiedFlowGenerator(device=self.device)
        except NameError:
            print("Warning: RectifiedFlowGenerator class not found. Skipping sample generation.")
            model.train()
            return
        
        sample_batch_data = next(iter(dataset_loader))
        low_res_b, _, original_b, _ = sample_batch_data
        
        num_samples_to_generate = min(3, low_res_b.shape[0])
        low_res_b = low_res_b[:num_samples_to_generate].to(self.device)
        original_b = original_b[:num_samples_to_generate].to(self.device)

        sample_condition_features = None
        if context_extractor is not None:
            with torch.no_grad():
                _, raw_features_list_gpu_sample = context_extractor(low_res_b, get_fea=True)
                sample_condition_features = [feat.to(self.device) for feat in raw_features_list_gpu_sample]

        with torch.no_grad():
            # Use the low-resolution image as the starting point for generation.
            # The generator will handle upsampling it internally.
            generated_hr_images = generator.generate(
                model=model,
                x0_lr=low_res_b, # Start from the LR image
                target_size=original_b.shape[2:], # Specify target HR dimensions
                condition=sample_condition_features,
                num_inference_steps=50 # Number of ODE solver steps
            )

        for i in range(num_samples_to_generate):
            lr_img_display = (low_res_b[i].cpu().numpy() + 1.0) / 2.0
            generated_hr_display = (generated_hr_images[i].cpu().numpy() + 1.0) / 2.0
            original_hr_display = (original_b[i].cpu().numpy() + 1.0) / 2.0
            
            # For comparison, show what the simple upscaled version looks like
            upscaled_base_display = F.interpolate(low_res_b[i].unsqueeze(0), size=original_b.shape[2:], mode='bilinear', align_corners=False)
            upscaled_base_display = (upscaled_base_display.squeeze(0).cpu().numpy() + 1.0) / 2.0


            writer.add_image(f'Sample_{i}/01_LowRes_Input', np.clip(lr_img_display, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/02_Bilinear_Upscaled_Base_(x0)', np.clip(upscaled_base_display, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/03_Generated_HR_RectifiedFlow', np.clip(generated_hr_display, 0, 1), epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/04_Original_HR_GroundTruth_(x1)', np.clip(original_hr_display, 0, 1), epoch + 1, dataformats='CHW')

        print(f"Logged {num_samples_to_generate} sample images to TensorBoard.")
        model.train()

    def train(self,
                train_dataset: DataLoader,
                model: torch.nn.Module,
                optimizer,
                scheduler=None,
                context_extractor: torch.nn.Module = None,
                val_dataset: DataLoader = None,
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
        The main training loop for the Rectified Flow model.
        """
        model.to(self.device)
        if context_extractor:
            context_extractor.to(self.device)
            context_extractor.eval()

        writer, checkpoint_dir, log_dir, best_checkpoint_path = self._setup_training_directories_and_writer(
            log_dir_base, checkpoint_dir_base, log_dir_param, checkpoint_dir_param
        )

        global_step_optimizer, batch_step_counter, current_accumulation_idx = self._initialize_training_steps(
            start_epoch, len(train_dataset), accumulation_steps
        )

        current_best_val_loss_tracker = best_loss

        print(f"Starting Rectified Flow training on device: {self.device}")
        # ... (other log prints)

        if start_epoch == 0: optimizer.zero_grad()

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()

            progress_bar_train = tqdm(total=len(train_dataset), desc=f"Training (RectifiedFlow, Epoch {epoch+1})")
            train_epoch_losses = []

            for batch_idx, batch_data in enumerate(train_dataset):
                loss_value, current_accumulation_idx, global_step_optimizer, updated_optimizer = self._perform_batch_step(
                    model, context_extractor, batch_data, optimizer,
                    accumulation_steps, current_accumulation_idx, global_step_optimizer, is_training=True
                )

                train_epoch_losses.append(loss_value)
                writer.add_scalar(f'Loss_RectifiedFlow/train_batch_step_raw', loss_value, batch_step_counter)
                if updated_optimizer and optimizer.param_groups:
                         writer.add_scalar(f'LearningRate/optimizer_step', optimizer.param_groups[0]['lr'], global_step_optimizer)

                progress_bar_train.update(1)
                progress_bar_train.set_postfix(loss=f"{loss_value:.4f}", opt_steps=global_step_optimizer)
                batch_step_counter += 1

            progress_bar_train.close()

            current_val_loss_for_this_epoch = None
            if val_dataset and (epoch + 1) % val_every_n_epochs == 0:
                current_val_loss_for_this_epoch = self._run_validation_epoch(model, context_extractor, val_dataset, writer, epoch)

            current_best_val_loss_tracker = self._log_and_checkpoint_epoch_end(
                epoch, model, optimizer, scheduler, train_epoch_losses,
                current_best_val_loss_tracker,
                current_val_loss_for_this_epoch,
                global_step_optimizer, best_checkpoint_path, writer,
                val_dataset if val_dataset else train_dataset,
                context_extractor
            )
        
        # ... (final logic after all epochs)
        writer.close()
        print(f"Training finished. Final best tracked loss: {current_best_val_loss_tracker:.4f}")

class RectifiedFlowGenerator:
    """
    A class for generating images using a pre-trained Rectified Flow model.
    Uses a simple ODE solver (Euler method) for image generation.
    """
    def __init__(self, device='cuda'):
        self.device = device

    @torch.no_grad()
    def generate(self, model, x0_lr, target_size, condition, num_inference_steps=50):
        """
        Generates an image by solving the Ordinary Differential Equation (ODE).

        Args:
            model (torch.nn.Module): The pre-trained Rectified Flow model.
            x0_lr (torch.Tensor): The starting low-resolution image.
            target_size (tuple): The target (height, width) for the output image.
            condition (list[torch.Tensor] or None): Conditional features for the model.
            num_inference_steps (int): The number of steps for the ODE solver (more steps = more accurate).

        Returns:
            torch.Tensor: The generated image at the end of the flow.
        """
        model.eval()
        
        # Upsample the low-resolution starting point to the target size to initialize the ODE state 'z'.
        z = F.interpolate(x0_lr.to(self.device), size=target_size, mode='bilinear', align_corners=False)
        
        dt = 1.0 / num_inference_steps

        for i in tqdm(range(num_inference_steps), desc="Generating with Rectified Flow (ODE)"):
            t = torch.full((x0_lr.shape[0],), i / num_inference_steps, device=self.device)
            
            # Predict the velocity at the current point z and time t
            velocity = model(z, t, condition=condition)
            
            # Take a step using the Euler method
            z = z + velocity * dt
        
        # The final result is the point z at t=1
        return torch.clamp(z, -1.0, 1.0)

