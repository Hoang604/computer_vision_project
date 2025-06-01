import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
import numpy as np
from tqdm import tqdm
import os
import datetime
from diffusers import DDIMScheduler 
from torch.utils.data import DataLoader 


class DiffusionTrainer:
    """
    Implements a Denoising Diffusion Probabilistic Model (DDPM) trainer.
    """
    def __init__(
        self,
        timesteps=1000,
        device='cuda',
        mode="v_prediction" 
    ):
        self.timesteps = timesteps
        self.device = device

        if mode not in ["v_prediction", "noise"]:
            raise ValueError("Mode must be 'v_prediction' or 'noise'") 
        self.mode = mode
        print(f"DiffusionTrainer initialized in '{self.mode}' mode.") 


        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) 

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_shift_right = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)
        self.posterior_variance = (self.betas *
            (1.0 - self.alphas_cumprod_shift_right) /
            (1.0 - self.alphas_cumprod)).to(self.device)
        self.posterior_variance = torch.nan_to_num(self.posterior_variance, nan=0.0, posinf=0.0, neginf=0.0)


    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a[t]
        return out.view(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t):
        x_0 = x_0.to(self.device) 
        x_0 = x_0.float() 
        noise = torch.randn_like(x_0, device=x_0.device)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def _setup_training_directories_and_writer(self, log_dir_base, checkpoint_dir_base,
                                                log_dir_param, checkpoint_dir_param):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{self.mode}_{timestamp}"

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
        best_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_model_{os.path.basename(checkpoint_dir)}_best.pth')
        return writer, checkpoint_dir, log_dir, best_checkpoint_path

    def _initialize_training_steps(self, start_epoch, dataset_len, accumulation_steps):
        effective_batches_per_epoch = dataset_len
        global_step_optimizer = start_epoch * effective_batches_per_epoch 
        batch_step_counter = start_epoch * dataset_len 
        current_accumulation_idx = 0 
        return global_step_optimizer, batch_step_counter, current_accumulation_idx

    def _get_training_target(self, noise_added, original_residual_batch, t):
        if self.mode == "v_prediction":
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, original_residual_batch.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, original_residual_batch.shape)
            target = sqrt_alphas_cumprod_t * noise_added - sqrt_one_minus_alphas_cumprod_t * original_residual_batch
        elif self.mode == "noise":
            target = noise_added
        else:
            raise ValueError(f"Unsupported training mode: {self.mode}") 
        return target

    def _perform_batch_step(self, model, context_extractor, batch_data, optimizer,
                            accumulation_steps, current_accumulation_idx, global_step_optimizer, is_training=True):
        low_res_image_batch, _, _, residual_image_batch = batch_data
        low_res_image_batch = low_res_image_batch.to(self.device)
        residual_image_batch = residual_image_batch.to(self.device)
        actual_batch_size = residual_image_batch.shape[0]

        condition_features_list = None
        if context_extractor is not None:
            context_extractor.eval() 
            with torch.no_grad(): 
                _, raw_features_list_gpu = context_extractor(low_res_image_batch, get_fea=True)
                if raw_features_list_gpu is None: # Check if features are None
                    print("Warning: context_extractor returned None for features.")
                    # Handle case where features might be None if U-Net can accept None condition
                else:
                    condition_features_list = [feat.to(self.device) for feat in raw_features_list_gpu]
        else:
            print("Warning: Context extractor is None. U-Net will receive no explicit condition.")

        with torch.set_grad_enabled(is_training): 
            t = torch.randint(0, self.timesteps, (actual_batch_size,), device=self.device, dtype=torch.long)
            residual_image_batch_t, noise_added = self.q_sample(residual_image_batch, t) 
            target_for_unet = self._get_training_target(noise_added, residual_image_batch, t) 
            predicted_output = model(residual_image_batch_t, t, condition=condition_features_list)
            loss = F.mse_loss(predicted_output, target_for_unet)

        updated_optimizer_this_step = False
        if is_training:
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            current_accumulation_idx += 1
            if current_accumulation_idx >= accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                current_accumulation_idx = 0
                global_step_optimizer +=1
                updated_optimizer_this_step = True
        return loss.detach().item(), current_accumulation_idx, global_step_optimizer, updated_optimizer_this_step

    def _run_validation_epoch(self, model, context_extractor, val_loader, writer, epoch):
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
        print(f"Epoch {epoch+1} Average Validation Loss ({self.mode}): {avg_val_loss:.4f}")

        if writer:
            writer.add_scalar(f'Loss_{self.mode}/validation_epoch_avg', avg_val_loss, epoch + 1)
        return avg_val_loss

    def _log_and_checkpoint_epoch_end(self, epoch, model, optimizer, scheduler, train_epoch_losses,
                                        current_best_val_loss,
                                        val_loss_this_epoch,
                                        global_step_optimizer, best_checkpoint_path, writer,
                                        dataset_for_samples, context_extractor): 
        mean_train_loss = np.mean(train_epoch_losses) if train_epoch_losses else float('nan')
        print(f"Epoch {epoch+1} Average Training Loss ({self.mode}): {mean_train_loss:.4f}")
        writer.add_scalar(f'Loss_{self.mode}/train_epoch_avg', mean_train_loss, epoch + 1)

        if optimizer.param_groups:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'LearningRate/epoch', current_lr, epoch + 1)
            print(f"Current Learning Rate at end of epoch {epoch+1}: {current_lr:.2e}")

        new_best_val_loss = current_best_val_loss
        loss_for_comparison = val_loss_this_epoch if val_loss_this_epoch is not None else mean_train_loss
        
        saved_model_config = {
            'base_dim': getattr(model, 'base_dim', None),
            'out_dim': getattr(model, 'out_dim', None), 
            'dim_mults': getattr(model, 'dim_mults', None),
            'cond_dim': getattr(model, 'cond_dim', None),
            'rrdb_num_blocks': getattr(model, 'rrdb_num_blocks', 8),
            'sr_scale': getattr(model, 'sr_scale', 4),
            'use_attention': getattr(model, 'use_attention', False),
            'timesteps': self.timesteps # Save the trainer's timesteps with the model
        }


        if loss_for_comparison is not None and not np.isnan(loss_for_comparison) and loss_for_comparison < current_best_val_loss:
            new_best_val_loss = loss_for_comparison
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': new_best_val_loss, 
                'train_loss_epoch_avg': mean_train_loss, 
                'val_loss_epoch_avg': val_loss_this_epoch, 
                'global_optimizer_steps': global_step_optimizer,
                'mode': self.mode,
                'model_config': saved_model_config
            }
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Best Loss: {new_best_val_loss:.4f}, Train Loss: {mean_train_loss:.4f})")

        current_epoch_checkpoint_path = os.path.join(os.path.dirname(best_checkpoint_path), f'diffusion_model_epoch_{epoch + 1}.pth')
        current_epoch_checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_train_loss, 
            'train_loss_epoch_avg': mean_train_loss,
            'val_loss_epoch_avg': val_loss_this_epoch,
            'global_optimizer_steps': global_step_optimizer,
            'mode': self.mode,
            'model_config': saved_model_config 
        }
        if scheduler:
            current_epoch_checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(current_epoch_checkpoint_data, current_epoch_checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch + 1} to {current_epoch_checkpoint_path}")


        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_for_plateau = val_loss_this_epoch if val_loss_this_epoch is not None else mean_train_loss
                if metric_for_plateau is not None and not np.isnan(metric_for_plateau):
                    scheduler.step(metric_for_plateau)
                    print(f"ReduceLROnPlateau scheduler stepped with metric {metric_for_plateau:.4f}.")
                else:
                    print("ReduceLROnPlateau scheduler not stepped as metric is None or NaN.")
            else: 
                scheduler.step()
                print(f"Epoch-based scheduler stepped. New LR (from optimizer): {optimizer.param_groups[0]['lr']:.2e}")

        if dataset_for_samples is not None: 
            try:
                self._generate_and_log_samples(model, dataset_for_samples, epoch, writer, context_extractor) 
            except Exception as e_sample:
                print(f"Error during sample generation: {e_sample}")
                import traceback
                traceback.print_exc()
        return new_best_val_loss

    def _generate_and_log_samples(self, model, dataset_loader, epoch, writer, context_extractor): 
        print(f"Generating sample images for TensorBoard at epoch {epoch+1}...")
        model.eval() 

        try:
            img_channels_sample = 3 
            img_size_sample = 160   
            if hasattr(dataset_loader, 'dataset') and hasattr(dataset_loader.dataset, 'img_size'):
                img_size_sample = dataset_loader.dataset.img_size
            
            generator = ResidualGenerator(
                img_channels=img_channels_sample,
                img_size=img_size_sample, 
                device=self.device,
                num_train_timesteps=self.timesteps, 
                predict_mode=self.mode 
            )
        except NameError: 
            print("Warning: ResidualGenerator class not found. Skipping sample generation.")
            model.train() 
            return
        except Exception as e_gen_init:
            print(f"Error initializing ResidualGenerator: {e_gen_init}. Skipping sample generation.")
            model.train()
            return

        sample_images_data = []
        num_samples_to_generate = min(3, dataset_loader.batch_size if dataset_loader.batch_size else 3)
        if num_samples_to_generate == 0:
            print("Warning: Cannot generate samples, effective num_samples_to_generate is 0.")
            model.train()
            return
        
        try:
            sample_batch_data = next(iter(dataset_loader))
            low_res_b, up_scale_b, original_b, residual_b_cpu = sample_batch_data
        except StopIteration:
            print("Warning: Not enough data in dataset_loader to generate samples.")
            model.train()
            return
        except Exception as e_data_unpack:
            print(f"Error unpacking sample batch data: {e_data_unpack}")
            model.train()
            return

        low_res_b = low_res_b[:num_samples_to_generate].to(self.device)
        up_scale_b = up_scale_b[:num_samples_to_generate].to(self.device)
        original_b = original_b[:num_samples_to_generate].to(self.device) 
        residual_b_gt = residual_b_cpu[:num_samples_to_generate].to(self.device) 

        sample_condition_features = None
        if context_extractor is not None:
            with torch.no_grad():
                _, raw_features_list_gpu_sample = context_extractor(low_res_b, get_fea=True)
                if raw_features_list_gpu_sample is not None:
                    sample_condition_features = [feat.to(self.device) for feat in raw_features_list_gpu_sample]
                else:
                    print("Warning: Context extractor returned None for features during sample generation.")
        else:
            print("Warning: context_extractor is None during sample generation. Passing None condition to U-Net.")

        with torch.no_grad():
            generated_residuals_batch = generator.generate_residuals(
                model=model,
                features=sample_condition_features,
                num_images=low_res_b.shape[0],
                num_inference_steps=50 
            )

        for i in range(generated_residuals_batch.shape[0]):
            lr_img_display = (low_res_b[i].cpu().numpy() + 1.0) / 2.0
            hr_rrdb_display = (up_scale_b[i].cpu().numpy() + 1.0) / 2.0 
            hr_orig_display = (original_b[i].cpu().numpy() + 1.0) / 2.0
            
            predicted_residual_display = (generated_residuals_batch[i].cpu().numpy() + 1.0) / 2.0
            final_hr_constructed = torch.clamp(up_scale_b[i] + generated_residuals_batch[i], -1.0, 1.0)
            final_hr_constructed_display = (final_hr_constructed.cpu().numpy() + 1.0) / 2.0
            
            true_residual_display = (residual_b_gt[i].cpu().numpy() + 1.0) / 2.0 

            sample_images_data.append({
                "low_res": np.clip(lr_img_display, 0, 1),
                "hr_rrdb": np.clip(hr_rrdb_display, 0, 1),
                "generated_residual": np.clip(predicted_residual_display, 0, 1),
                "final_hr_refined": np.clip(final_hr_constructed_display, 0, 1),
                "original_hr_gt": np.clip(hr_orig_display, 0, 1),
                "true_residual_gt": np.clip(true_residual_display, 0, 1)
            })

        for i, imgs_dict in enumerate(sample_images_data):
            writer.add_image(f'Sample_{i}/01_LowRes_Input', imgs_dict["low_res"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/02_HR_RRDB_Base', imgs_dict["hr_rrdb"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/03_Predicted_Residual_Diffusion', imgs_dict["generated_residual"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/04_Final_HR_Refined', imgs_dict["final_hr_refined"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/05_Original_HR_GroundTruth', imgs_dict["original_hr_gt"], epoch + 1, dataformats='CHW')
            writer.add_image(f'Sample_{i}/06_True_Residual_GT_(HR_orig-HR_rrdb)', imgs_dict["true_residual_gt"], epoch + 1, dataformats='CHW')

        print(f"Logged {len(sample_images_data)} sample images to TensorBoard.")
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
                log_dir_base="./logs_diffusion",
                checkpoint_dir_base="./checkpoints_diffusion",
             ) -> None:

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

        print(f"Starting training in '{self.mode}' mode on device: {self.device} with {accumulation_steps} accumulation steps.")
        if context_extractor:
            print(f"Using on-the-fly context extraction with: {type(context_extractor).__name__}")
        print(f"Logging to: {log_dir}")
        print(f"Saving best checkpoints to: {best_checkpoint_path}")
        if val_dataset: print(f"Validation will be performed every {val_every_n_epochs} epoch(s).")
        print(f"Initial global optimizer steps: {global_step_optimizer}, initial batch steps: {batch_step_counter}")
        print(f"Initial best tracked loss (from resume or inf): {current_best_val_loss_tracker:.6f}")

        if start_epoch == 0: optimizer.zero_grad()

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            progress_bar_train = tqdm(total=len(train_dataset), desc=f"Training ({self.mode}, Epoch {epoch+1})")
            train_epoch_losses = []

            for batch_idx, batch_data in enumerate(train_dataset):
                loss_value, current_accumulation_idx, global_step_optimizer, updated_optimizer = self._perform_batch_step(
                    model, context_extractor, batch_data, optimizer,
                    accumulation_steps, current_accumulation_idx, global_step_optimizer, is_training=True
                )
                train_epoch_losses.append(loss_value)
                writer.add_scalar(f'Loss_{self.mode}/train_batch_step_raw', loss_value, batch_step_counter)
                if updated_optimizer and optimizer.param_groups:
                         writer.add_scalar(f'LearningRate/optimizer_step', optimizer.param_groups[0]['lr'], global_step_optimizer)

                progress_bar_train.update(1)
                progress_bar_train.set_postfix(loss=f"{loss_value:.4f}", opt_steps=global_step_optimizer, lr=f"{optimizer.param_groups[0]['lr']:.2e}")
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

        if current_accumulation_idx > 0: 
            print(f"Performing final optimizer step for {current_accumulation_idx} remaining accumulated gradients...")
            optimizer.step()
            optimizer.zero_grad()
            global_step_optimizer +=1
            print(f"Final gradients applied. Total optimizer steps: {global_step_optimizer}")

        writer.close()
        print(f"Training finished for mode '{self.mode}'. Final best tracked loss: {current_best_val_loss_tracker:.4f}")

    @staticmethod
    def load_diffusion_unet(model_path: str,
        device: torch.device = torch.device("cuda"),
        verbose: bool = True
    ) -> torch.nn.Module:
        if verbose:
            print(f"Loading UNet model from {model_path}...")
        
        loaded_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if loaded_checkpoint is None:
            raise ValueError(f"torch.load returned None for checkpoint file: {model_path}. File might be empty or corrupted.")

        if 'model_config' not in loaded_checkpoint:
            raise KeyError(f"Checkpoint {model_path} missing 'model_config'. Cannot initialize UNet.")
        model_config = loaded_checkpoint['model_config']
        if model_config is None:
             raise ValueError(f"The 'model_config' key in checkpoint {model_path} is None. Cannot initialize UNet.")


        if 'model_state_dict' not in loaded_checkpoint:
            raise KeyError(f"Checkpoint {model_path} missing 'model_state_dict'. Cannot load UNet weights.")
        model_state_dict = loaded_checkpoint['model_state_dict']
        
        from src.diffusion_modules.unet import Unet 
        # Ensure all necessary params for Unet are present in model_config or have defaults in Unet itself
        # If model_config might have None for keys Unet expects non-None, Unet init must handle it or defaults used carefully.
        unet = Unet(**model_config).to(device) 
        unet.load_state_dict(model_state_dict, strict=True)
        
        if verbose:
            print("UNet model loaded successfully with strict key checking.")
        return unet

    @staticmethod
    def load_model_weights(model, model_path, verbose=False, device='cuda'):
        if not os.path.exists(model_path):
            print(f"Warning: Model weights path not found: {model_path}. Model weights not loaded.") 
            return

        print(f"Loading model weights from: {model_path} onto device: {device}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if checkpoint is None: # Added check for None checkpoint
            print(f"Warning: torch.load returned None for {model_path}. Model weights not loaded.")
            return

        state_dict_to_load = None
        if isinstance(checkpoint, dict): 
            if "model_state_dict" in checkpoint:
                state_dict_to_load = checkpoint["model_state_dict"]
                print("Found 'model_state_dict' in checkpoint.")
            elif "state_dict" in checkpoint and "model" in checkpoint["state_dict"]: 
                state_dict_to_load = checkpoint["state_dict"]["model"]
                print("Found 'state_dict' -> 'model' in checkpoint.")
            else: 
                is_likely_model_state_dict = all(not k.startswith(('optimizer', 'scheduler', 'epoch', 'loss')) for k in checkpoint.keys())
                if is_likely_model_state_dict:
                    state_dict_to_load = checkpoint
                    print("Checkpoint appears to be a raw model state_dict.") 
                else:
                    print(f"Warning: Checkpoint dictionary does not contain a clear model state_dict. Keys: {list(checkpoint.keys())}")
        elif isinstance(checkpoint, torch.nn.Module): 
            state_dict_to_load = checkpoint.state_dict()
            print("Checkpoint was a full model object; extracting state_dict.") 
        else:
            print(f"Warning: Checkpoint format not recognized or is not a dictionary/model. Type: {type(checkpoint)}") 
            return

        if state_dict_to_load:
            model.to(device)
            incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False) 
            if incompatible_keys.missing_keys:
                print(f"Warning: {len(incompatible_keys.missing_keys)} keys in the current model were not found in the checkpoint.") 
                if verbose: print(f"Missing keys: {incompatible_keys.missing_keys}") 
            if incompatible_keys.unexpected_keys:
                print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in the checkpoint were not used by the current model.") 
                if verbose: print(f"Unused (unexpected) keys: {incompatible_keys.unexpected_keys}") 

            if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                print(f"Successfully loaded all parameters into the model on device {device}.")
            else:
                num_loaded_params = sum(1 for k_model in model.state_dict() if k_model in state_dict_to_load and k_model not in incompatible_keys.unexpected_keys)
                print(f"Successfully loaded {num_loaded_params} compatible parameters into the model on device {device}.") 
        else:
            print(f"Warning: Could not extract a loadable state_dict from {model_path}.") 

    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, scheduler, checkpoint_path, verbose_load=False): 
        start_epoch = 0
        loaded_best_loss = float('inf') 

        model.to(device)
        print(f"Attempting to load checkpoint for resume from: {checkpoint_path} onto device: {device}")

        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint file not found at {checkpoint_path}. Training will start from scratch.")
            return start_epoch, loaded_best_loss

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint is None: # Added check for None checkpoint
                print(f"Warning: torch.load returned None for checkpoint {checkpoint_path}. Training will start from scratch.")
                return start_epoch, loaded_best_loss
            
            print(f"Checkpoint dictionary loaded successfully from {checkpoint_path}.")

            # Load model state using the robust load_model_weights method
            if 'model_state_dict' in checkpoint or \
               ('state_dict' in checkpoint and 'model' in checkpoint['state_dict']) or \
               all(not k.startswith(('optimizer', 'scheduler', 'epoch', 'loss')) for k in checkpoint.keys()): # Heuristic for raw state_dict
                DiffusionTrainer.load_model_weights(model, checkpoint_path, verbose=verbose_load, device=device) # Pass path to load_model_weights
                print(f"Model state loaded for resume via load_model_weights.")
            else:
                print("Warning: No clear model state_dict found in checkpoint for resume.")


            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    for state_val in optimizer.state.values(): # Corrected variable name
                        for k, v in state_val.items(): # Corrected variable name
                            if isinstance(v, torch.Tensor):
                                state_val[k] = v.to(device) # Corrected variable name
                    print(f"Optimizer state loaded successfully for resume and moved to {device}.") 
                except Exception as optim_load_err:
                     print(f"Error loading optimizer state for resume: {optim_load_err}. Optimizer will start from scratch.") 
            elif optimizer is None: print("Warning: Optimizer not provided, skipping optimizer state loading for resume.") 
            else: print("Warning: 'optimizer_state_dict' not found in checkpoint. Optimizer starts from scratch for resume.")

            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"Scheduler state loaded successfully for resume.") 
                except Exception as sched_load_err:
                    print(f"Error loading scheduler state for resume: {sched_load_err}. Scheduler may start from scratch or use loaded optimizer LR.")
            elif scheduler is None: print("Info: Scheduler not provided, skipping scheduler state loading for resume.")
            else: print("Info: 'scheduler_state_dict' not found in checkpoint. Scheduler may start from scratch.")

            if 'epoch' in checkpoint:
                saved_epoch = checkpoint['epoch']
                start_epoch = saved_epoch + 1 
                print(f"Resuming training from epoch: {start_epoch}")
            else: print("Warning: Epoch number not found in checkpoint. Starting from epoch 0 for resume.") 

            if 'loss' in checkpoint: 
                loaded_best_loss = checkpoint['loss']
                print(f"Loaded 'best_loss' (from 'loss' key) for resume: {loaded_best_loss:.6f}")
            elif 'best_loss' in checkpoint: 
                loaded_best_loss = checkpoint['best_loss']
                print(f"Loaded 'best_loss' (from 'best_loss' key) for resume: {loaded_best_loss:.6f}")
            else: print("Info: Best loss value not found in checkpoint. Using default best_loss (inf) for resume.")

            if 'global_optimizer_steps' in checkpoint:
                print(f"Checkpoint saved with global_optimizer_steps: {checkpoint['global_optimizer_steps']}")

        except Exception as e:
            print(f"Error loading checkpoint for resume: {e}. Training will start from scratch.")
            start_epoch = 0
            loaded_best_loss = float('inf')
        return start_epoch, loaded_best_loss


class ResidualGenerator:
    """
    A class for generating image residuals using a pre-trained diffusion model and a scheduler.
    """
    def __init__(self,
                 img_channels=3,
                 img_size=256,
                 device='cuda',
                 num_train_timesteps=1000,
                 predict_mode='v_prediction'):
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        if predict_mode not in ["v_prediction", "noise"]:
            raise ValueError("Prediction mode must be 'v_prediction' or 'noise'")
        self.predict_mode = predict_mode

        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999)

        self.betas = cosine_beta_schedule(self.num_train_timesteps).to(self.device)
        scheduler_prediction_type = 'v_prediction' if self.predict_mode == 'v_prediction' else 'epsilon'

        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=self.betas.cpu().numpy(), 
            beta_schedule="trained_betas",
            prediction_type=scheduler_prediction_type,
            clip_sample=False, 
            set_alpha_to_one=False, 
            steps_offset=1, 
        )
        print(f"ResidualGenerator initialized with {type(self.scheduler).__name__}, "
              f"configured for model prediction_type='{self.predict_mode}' (scheduler prediction_type='{scheduler_prediction_type}'). "
              f"Image size: {self.img_size}x{self.img_size}, Channels: {self.img_channels}.")

    @torch.no_grad()
    def generate_residuals(self, model, features, num_images=1, num_inference_steps=50, return_intermediate_steps=False):
        model.eval()
        model.to(self.device)

        print(f"Generating {num_images} residuals using {num_inference_steps} steps "
              f"with {type(self.scheduler).__name__} (model expected to predict: '{self.predict_mode}') "
              f"on device {self.device}...")

        self.scheduler.set_timesteps(num_inference_steps, device=self.device) 

        image_latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device 
        )
        image_latents = image_latents * self.scheduler.init_noise_sigma 

        intermediate_latents_list = []
        if return_intermediate_steps:
            intermediate_latents_list.append(image_latents.clone())

        for t_step in tqdm(self.scheduler.timesteps, desc="Generating residuals"): 
            model_input = image_latents 
            t_for_model = t_step.unsqueeze(0).expand(num_images).to(self.device) 
            
            conditioned_features_for_batch = None
            if features is not None:
                if isinstance(features, list):
                    conditioned_features_for_batch = [f.to(self.device) for f in features if f is not None]
                    if not conditioned_features_for_batch : # if list becomes empty
                        conditioned_features_for_batch = None
                elif isinstance(features, torch.Tensor):
                     conditioned_features_for_batch = features.to(self.device)
                # Ensure it's None if no valid features were processed
                if not conditioned_features_for_batch: conditioned_features_for_batch = None


            model_output = model(model_input, t_for_model, condition=conditioned_features_for_batch)
            scheduler_output = self.scheduler.step(model_output, t_step, image_latents)
            image_latents = scheduler_output.prev_sample

            if return_intermediate_steps:
                intermediate_latents_list.append(image_latents.clone())

        generated_residuals = image_latents
        print("Residual generation complete.")

        if return_intermediate_steps:
            return intermediate_latents_list
        else:
            return generated_residuals