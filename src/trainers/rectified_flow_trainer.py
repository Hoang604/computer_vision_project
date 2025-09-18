import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
from torch.utils.data import DataLoader
from itertools import cycle

class ImageGenerator:
    """
    Generates images using a pre-trained Rectified Flow model.
    """
    def __init__(self, img_channels=3, img_size=160, device='cuda'):
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        print(f"Rectified Flow ImageGenerator initialized on device {self.device} for image size {img_size}x{img_size}.")

    @torch.no_grad()
    def generate_images(self, model, features, num_images=1, num_inference_steps=100, initial_noise=None):
        model.eval()
        model.to(self.device)
        if initial_noise is None:
            z = torch.randn((num_images, self.img_channels, self.img_size, self.img_size), device=self.device)
        else:
            z = initial_noise.to(self.device)
        dt = 1.0 / num_inference_steps
        for i in tqdm(range(num_inference_steps), desc="Solving ODE", leave=False):
            t = torch.full((num_images,), i * dt, device=self.device)
            v = model(z, t, condition=features)
            z = z + v * dt
        return z

class RectifiedFlowTrainer:
    """
    Implements a robust, step-based trainer for Rectified Flow models.
    """
    def __init__(self, device='cuda', mode='rectified'):
        self.device = device
        if mode not in ['rectified', 'reflow']:
            raise ValueError("Mode must be 'rectified' or 'reflow'")
        self.mode = mode
        print(f"RectifiedFlowTrainer initialized in '{self.mode}' mode on device '{self.device}'.")

    def get_interpolated_sample(self, x0, x1, t):
        t_reshaped = t.view(-1, 1, 1, 1)
        return t_reshaped * x1 + (1. - t_reshaped) * x0

    def _setup_training_directories_and_writer(self, log_dir_base, checkpoint_dir_base,
                                                log_dir_param, checkpoint_dir_param):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{self.mode.capitalize()}_{timestamp}"

        if log_dir_param and checkpoint_dir_param:
            log_dir, checkpoint_dir = log_dir_param, checkpoint_dir_param
            experiment_name = os.path.basename(checkpoint_dir)
        else:
            log_dir = os.path.join(log_dir_base, experiment_name)
            checkpoint_dir = os.path.join(checkpoint_dir_base, experiment_name)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        best_checkpoint_path = os.path.join(checkpoint_dir, f'model_{experiment_name}_best.pth')
        latest_checkpoint_path = os.path.join(checkpoint_dir, f'model_{experiment_name}_latest.pth')

        return writer, checkpoint_dir, best_checkpoint_path, latest_checkpoint_path

    def _perform_batch_step(self, model, context_extractor, batch_data,
                            accumulation_steps, is_training=True):
        with torch.set_grad_enabled(is_training):
            if not batch_data or batch_data[0].shape[0] == 0:
                return 0.0

            if self.mode == 'rectified':
                low_res_image_batch, _, x1, _ = batch_data
                x1 = x1.to(self.device)
                x0 = torch.randn_like(x1)
            elif self.mode == 'reflow':
                low_res_image_batch, x0, x1 = batch_data
                x0, x1 = x0.to(self.device), x1.to(self.device)
            
            low_res_image_batch = low_res_image_batch.to(self.device)
            
            condition_features_list = None
            if context_extractor is not None:
                with torch.no_grad():
                    _, raw_features_list_gpu = context_extractor(low_res_image_batch, get_fea=True)
                    condition_features_list = [feat.detach() for feat in raw_features_list_gpu]
            
            target_velocity = x1 - x0
            t = torch.rand(x0.shape[0], device=self.device)
            xt = self.get_interpolated_sample(x0, x1, t)
            predicted_velocity = model(xt, t, condition=condition_features_list)
            loss = F.mse_loss(predicted_velocity, target_velocity)
        
        if is_training:
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
        return loss.detach().item()

    def _run_validation(self, model, context_extractor, val_loader, writer, global_step):
        model.eval()
        total_val_loss = 0.0
        print(f"\nRunning validation at step {global_step}...")
        progress_bar_val = tqdm(total=len(val_loader), desc=f"Validation Step {global_step}")
        with torch.no_grad():
            for batch_data in val_loader:
                loss_value = self._perform_batch_step(
                    model, context_extractor, batch_data, accumulation_steps=1, is_training=False
                )
                total_val_loss += loss_value
                progress_bar_val.update(1)
        progress_bar_val.close()
        avg_val_loss = total_val_loss / len(val_loader) if val_loader else float('nan')
        print(f"Step {global_step} Average Validation Loss: {avg_val_loss:.4f}")
        if writer: writer.add_scalar(f'Loss/{self.mode}_validation_avg', avg_val_loss, global_step)
        return avg_val_loss

    def _save_checkpoint(self, global_step, model, optimizer, scheduler, best_validation_loss, current_loss, checkpoint_path):
        checkpoint_data = {
            'global_step': global_step, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss, 'best_validation_loss': best_validation_loss, 'mode': self.mode,
            'model_config': {
                'base_dim': getattr(model, 'base_dim', None), 'dim_mults': tuple(getattr(model, 'dim_mults', [])),
                'use_attention': getattr(model, 'use_attention', False), 'cond_dim': getattr(model, 'cond_dim', None),
            }
        }
        if scheduler: checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path} (Step {global_step}, Best Val Loss: {best_validation_loss:.4f})")

    def _generate_and_log_samples(self, model, sample_batch_data, global_step, writer, context_extractor, batch_idx_for_log):
        print(f"Generating samples for TensorBoard at step {global_step} (from val batch {batch_idx_for_log})...")
        model.eval()
        initial_noise_for_gen = None

        if self.mode == 'rectified':
            low_res_b, up_scale_b, original_b, _ = sample_batch_data
        else:
            low_res_b, x0_b, original_b = sample_batch_data
            initial_noise_for_gen = x0_b
            up_scale_b = torch.zeros_like(original_b)

        num_samples = min(4, low_res_b.shape[0])
        low_res_b, up_scale_b, original_b = [t[:num_samples].to(self.device) for t in [low_res_b, up_scale_b, original_b]]
        if initial_noise_for_gen is not None: initial_noise_for_gen = initial_noise_for_gen[:num_samples].to(self.device)

        with torch.no_grad():
            _, sample_condition_features = context_extractor(low_res_b, get_fea=True)
            generator = ImageGenerator(img_channels=original_b.shape[1], img_size=original_b.shape[2], device=self.device)
            generated_hr_batch = generator.generate_images(
                model, sample_condition_features, num_samples, 50, initial_noise_for_gen
            )
        
        for i in range(num_samples):
            images = [np.clip((img[i].cpu().numpy() + 1.0) / 2.0, 0, 1) for img in [low_res_b, up_scale_b, original_b, generated_hr_batch]]
            tag_prefix = f'ValBatch_{batch_idx_for_log}_Sample_{i}'
            writer.add_image(f'{tag_prefix}/01_LowRes_Input', images[0], global_step, dataformats='CHW')
            if self.mode == 'rectified': writer.add_image(f'{tag_prefix}/02_HR_RRDB_Baseline', images[1], global_step, dataformats='CHW')
            writer.add_image(f'{tag_prefix}/04_Original_HR_GroundTruth', images[2], global_step, dataformats='CHW')
            writer.add_image(f'{tag_prefix}/03_Predicted_HR', images[3], global_step, dataformats='CHW')
        print(f"Logged {num_samples} sample images.")

    def train(self,
                train_dataset: DataLoader, model: torch.nn.Module, optimizer, scheduler=None,
                context_extractor: torch.nn.Module = None, val_dataset: DataLoader = None,
                max_train_steps=300000, accumulation_steps=1, start_step=0, best_loss=float('inf'),
                validate_every_n_steps=1000, save_every_n_steps=1000,
                log_dir_param=None, checkpoint_dir_param=None,
                log_dir_base="./logs_rectified_flow", checkpoint_dir_base="./checkpoints_rectified_flow"):
        
        model.to(self.device)
        if context_extractor: context_extractor.to(self.device).eval()

        writer, _, best_ckpt_path, latest_ckpt_path = self._setup_training_directories_and_writer(
            log_dir_base, checkpoint_dir_base, log_dir_param, checkpoint_dir_param
        )
        global_step, current_best_val_loss = start_step, best_loss
        
        fixed_sample_batches = []
        if val_dataset:
            num_cached_batches = 4
            print(f"Caching {num_cached_batches} batches from validation set...")
            for i, batch in enumerate(val_dataset):
                if i >= num_cached_batches: break
                fixed_sample_batches.append(batch)
        
        sample_batch_idx_counter = 0
        train_data_iterator = cycle(train_dataset)
        progress_bar = tqdm(initial=start_step, total=max_train_steps, desc=f"Training ({self.mode.capitalize()})")

        optimizer.zero_grad()
        while global_step < max_train_steps:
            model.train()
            
            # MODIFICATION: Tích lũy loss để báo cáo chính xác
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                batch_data = next(train_data_iterator)
                loss = self._perform_batch_step(
                    model, context_extractor, batch_data, accumulation_steps, is_training=True
                )
                accumulated_loss += loss
            
            avg_loss = accumulated_loss / accumulation_steps
            # MODIFICATION END
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar(f'Loss/{self.mode}_train_step_avg', avg_loss, global_step) # Báo cáo loss trung bình
            writer.add_scalar('LearningRate/step', optimizer.param_groups[0]['lr'], global_step)
            progress_bar.update(1)
            progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", best_val=f"{current_best_val_loss:.4f}")
            
            if (global_step + 1) % validate_every_n_steps == 0 and val_dataset:
                val_loss = self._run_validation(model, context_extractor, val_dataset, writer, global_step + 1)
                if val_loss < current_best_val_loss:
                    current_best_val_loss = val_loss
                    self._save_checkpoint(global_step + 1, model, optimizer, scheduler, current_best_val_loss, val_loss, best_ckpt_path)
                
                if fixed_sample_batches:
                    batch_idx = sample_batch_idx_counter % len(fixed_sample_batches)
                    self._generate_and_log_samples(model, fixed_sample_batches[batch_idx], global_step + 1, writer, context_extractor, batch_idx)
                    sample_batch_idx_counter += 1

            if (global_step + 1) % save_every_n_steps == 0:
                 self._save_checkpoint(global_step + 1, model, optimizer, scheduler, current_best_val_loss, avg_loss, latest_ckpt_path)

            global_step += 1
            
        writer.close()
        progress_bar.close()
        print(f"Training finished. Total steps: {global_step}. Final best validation loss: {current_best_val_loss:.4f}")

    @staticmethod
    def load_rectified_flow_unet(model_path: str, device: torch.device = torch.device("cuda"), verbose: bool = True) -> torch.nn.Module:
        # ... (Implementation unchanged) ...
        if verbose: print(f"Loading UNet model from {model_path}...")
        from src.diffusion_modules.unet import Unet 
        checkpoint = torch.load(model_path, map_location=device)
        unet = Unet(**checkpoint['model_config']).to(device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        if verbose: print("Model loaded successfully.")
        return unet
    
    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, scheduler, checkpoint_path, verbose_load=False):
        # ... (Implementation unchanged) ...
        start_step, loaded_best_loss = 0, float('inf')
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")
            return start_step, loaded_best_loss
        print(f"Loading checkpoint for resume from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_step = checkpoint.get('global_step', 0)
        loaded_best_loss = checkpoint.get('best_validation_loss', checkpoint.get('loss', float('inf')))
        
        print(f"Resuming from step: {start_step}. Loaded best validation loss: {loaded_best_loss:.6f}")
        return start_step, loaded_best_loss

