import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import datetime
import numpy as np
from src.diffusion_modules.rrdb import RRDBNet


class BasicRRDBNetTrainer:
    def __init__(self, 
                 model_config: dict,      # {'in_nc':3, 'out_nc':3, 'num_feat': 64, 'num_block': 16, 'gc': 32, 'sr_scale': 4}
                 optimizer_config: dict,  # {'lr': 0.0002, 'beta1': 0.9, 'beta2': 0.999}
                 scheduler_config: dict = None, # Example: {'type': 'CosineAnnealingLR', 't_max': 100, 'eta_min': 0}
                 logging_config: dict = None,   # {'exp_name': 'my_exp', 'log_dir_base': 'logs', 'checkpoint_dir_base': 'ckpts'}
                 device: str = 'cuda'):
        
        self.device = device
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config if scheduler_config else {'type': 'none'} # Default to no scheduler
        self.logging_config = logging_config if logging_config else {}

        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)

        self.start_epoch = 0
        self.global_step_optimizer = 0
        self.batch_step_counter = 0
        
        self.best_loss = float('inf') 
        self.best_checkpoint_path = None 

        self.writer = None
        self.checkpoint_dir = None
        self.log_dir = None
        
        print(f"BasicRRDBNetTrainer initialized for device: {self.device}") 
        print(f"Model Config: {self.model_config}") 
        print(f"Optimizer Config: {self.optimizer_config}") 
        print(f"Scheduler Config: {self.scheduler_config}") 

    def _build_model(self):
        model = RRDBNet(
            in_channels=self.model_config.get('in_nc', 3), 
            out_channels=self.model_config.get('out_nc', 3), 
            rrdb_in_channels=self.model_config.get('num_feat', 64), 
            number_of_rrdb_blocks=self.model_config.get('num_block', 17), 
            growth_channels=self.model_config.get('gc', 32),
            sr_scale=self.model_config.get('sr_scale', 4)
        )
        return model

    def _build_optimizer(self, model_to_optimize):
        lr = self.optimizer_config.get('lr', 0.0002)
        beta1 = self.optimizer_config.get('beta1', 0.9)
        beta2 = self.optimizer_config.get('beta2', 0.999)
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        
        optimizer = torch.optim.AdamW(model_to_optimize.parameters(), 
                                      lr=lr, 
                                      betas=(beta1, beta2), 
                                      weight_decay=weight_decay)
        return optimizer

    def _build_scheduler(self, optimizer_to_schedule):
        scheduler_type = self.scheduler_config.get('type', 'none').lower() 
        
        if scheduler_type == 'none':
            print("No learning rate scheduler will be used.") 
            return None
        
        print(f"Building scheduler of type: {scheduler_type}") 
        if scheduler_type == 'steplr':
            step_size = self.scheduler_config.get('step_lr_step_size', 200000) 
            gamma = self.scheduler_config.get('step_lr_gamma', 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_to_schedule,
                                                        step_size=step_size, 
                                                        gamma=gamma)
            print(f"  StepLR configured with step_size={step_size}, gamma={gamma}") 
        elif scheduler_type == 'cosineannealinglr':
            t_max = self.scheduler_config.get('cosine_t_max', 100) 
            eta_min = self.scheduler_config.get('cosine_eta_min', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule, 
                                                                    T_max=t_max, 
                                                                    eta_min=eta_min)
            print(f"  CosineAnnealingLR configured with T_max={t_max}, eta_min={eta_min}") 
        elif scheduler_type == 'cosineannealingwarmrestarts':
            t_0 = self.scheduler_config.get('cosine_warm_t_0', 10) 
            t_mult = self.scheduler_config.get('cosine_warm_t_mult', 1) 
            eta_min = self.scheduler_config.get('cosine_warm_eta_min', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_to_schedule,
                                                                             T_0=t_0,
                                                                             T_mult=t_mult,
                                                                             eta_min=eta_min)
            print(f"  CosineAnnealingWarmRestarts configured with T_0={t_0}, T_mult={t_mult}, eta_min={eta_min}") 
        elif scheduler_type == 'reducelronplateau':
            mode = self.scheduler_config.get('plateau_mode', 'min')
            factor = self.scheduler_config.get('plateau_factor', 0.1)
            patience = self.scheduler_config.get('plateau_patience', 10) 
            verbose = self.scheduler_config.get('plateau_verbose', True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_to_schedule,
                                                                   mode=mode,
                                                                   factor=factor,
                                                                   patience=patience,
                                                                   verbose=verbose)
            print(f"  ReduceLROnPlateau configured with mode={mode}, factor={factor}, patience={patience}") 
        else:
            print(f"Warning: Scheduler type '{scheduler_type}' not recognized. No scheduler will be used.") 
            return None
        return scheduler

    def _setup_logging_and_checkpointing(self, log_dir_param, checkpoint_dir_param):
        """
        Sets up logging and checkpointing directories.
        Args:
            log_dir_param (str): Directory for logging.
            checkpoint_dir_param (str): Directory for saving checkpoints.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.logging_config.get('exp_name') is not None:
            exp_name = self.logging_config['exp_name']
        else:
            exp_name = f'basic_rrdb_{timestamp}' 
        
        log_dir_base = self.logging_config.get('log_dir_base', 'logs_rrdb_basic_standalone')
        checkpoint_dir_base = self.logging_config.get('checkpoint_dir_base', 'checkpoints_rrdb_basic_standalone')

        self.log_dir = log_dir_param if log_dir_param else os.path.join(log_dir_base, exp_name)
        self.checkpoint_dir = checkpoint_dir_param if checkpoint_dir_param else os.path.join(checkpoint_dir_base, exp_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, 'rrdb_model_best.pth') 
        
        print(f"Basic RRDBNet (Standalone) - Logging to: {self.log_dir}") 
        print(f"Basic RRDBNet (Standalone) - Saving checkpoints to: {self.checkpoint_dir}") 
        print(f"Basic RRDBNet (Standalone) - Best model will be saved to: {self.best_checkpoint_path}") 

    def _perform_one_batch_step(self, batch_data, predict_residual=False, is_training=True):
        img_lr, _, img_hr, img_res = batch_data 
        img_lr = img_lr.to(self.device)

        with torch.set_grad_enabled(is_training): 
            predicted_output = self.model(img_lr) 
            
            if predict_residual:
                target = img_res.to(self.device)
                loss = F.l1_loss(predicted_output, target)
            else:
                target = img_hr.to(self.device)
                loss = F.l1_loss(predicted_output, target) 
        return loss

    def _run_validation_epoch(self, val_loader: DataLoader, epoch: int, predict_residual: bool):
        """
        Runs a validation epoch.
        Args:
            val_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
            predict_residual (bool): Flag to predict residuals instead of HR images.
        Returns:
            avg_val_loss (float): Average validation loss for the epoch.
        """
        self.model.eval() 
        total_val_loss = 0.0
        num_val_batches = 0
        
        print(f"\nRunning validation for RRDBNet epoch {epoch+1}...") 
        progress_bar_val = tqdm(total=len(val_loader), desc=f"Validation RRDBNet Epoch {epoch+1}")

        with torch.no_grad(): 
            for batch_idx, batch_data in enumerate(val_loader):
                loss = self._perform_one_batch_step(batch_data, predict_residual=predict_residual, is_training=False)
                total_val_loss += loss.item()
                num_val_batches += 1
                progress_bar_val.update(1)
                progress_bar_val.set_postfix(val_loss_batch=f"{loss.item():.4f}")
        
        progress_bar_val.close()
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
        
        print(f"Epoch {epoch+1} Average Validation Loss (RRDBNet): {avg_val_loss:.4f}") 
        if self.writer:
            self.writer.add_scalar('Validation/Loss_epoch_avg', avg_val_loss, epoch + 1)
            
        return avg_val_loss

    def _log_rrdb_sample_images(self, 
                               train_loader: DataLoader, 
                               val_loader: DataLoader = None, # Added val_loader
                               epoch: int = 0, 
                               predict_residual: bool = False, 
                               num_samples=4):
        """
        Logs a grid of sample images (LR, HR/Residual Target, HR/Residual Predicted) to TensorBoard.
        Prioritizes val_loader for samples if available.
        """
        self.model.eval() # Set model to evaluation mode for consistent output
        
        data_loader_for_samples = None
        source_name = ""

        if val_loader:
            try:
                # Check if val_loader has data
                _ = next(iter(val_loader)) # Try to get one item
                data_loader_for_samples = val_loader
                source_name = "Validation"
            except StopIteration:
                print("Warning: val_loader is empty. Falling back to train_loader for image logging.") 
                data_loader_for_samples = train_loader
                source_name = "Training"
            except Exception as e: # Catch other potential errors with val_loader
                print(f"Warning: Error with val_loader ({e}). Falling back to train_loader for image logging.") 
                data_loader_for_samples = train_loader
                source_name = "Training"
        else:
            data_loader_for_samples = train_loader
            source_name = "Training"

        if not data_loader_for_samples:
            print("Warning: No data loader available for image logging.") 
            self.model.train()
            return

        try:
            sample_batch = next(iter(data_loader_for_samples))
        except StopIteration:
            print(f"Warning: Could not get a sample batch from {source_name} loader for image logging.") 
            self.model.train() 
            return

        img_lr_batch, _, img_hr_batch, img_res_batch = sample_batch
        
        img_lr_sample = img_lr_batch[:num_samples].to(self.device)
        
        with torch.no_grad():
            predicted_output_sample = self.model(img_lr_sample)

        if predict_residual:
            target_sample = img_res_batch[:num_samples].to(self.device)
            target_name = "Target_Residual"
            predicted_name = "Predicted_Residual"
        else:
            target_sample = img_hr_batch[:num_samples].to(self.device)
            target_name = "Target_HR"
            predicted_name = "Predicted_HR"

        img_lr_log = (img_lr_sample.cpu() + 1.0) / 2.0
        predicted_log = (predicted_output_sample.cpu() + 1.0) / 2.0
        target_log = (target_sample.cpu() + 1.0) / 2.0
        
        img_lr_log = torch.clamp(img_lr_log, 0.0, 1.0)
        predicted_log = torch.clamp(predicted_log, 0.0, 1.0)
        target_log = torch.clamp(target_log, 0.0, 1.0)

        grid_lr = make_grid(img_lr_log, nrow=num_samples)
        grid_predicted = make_grid(predicted_log, nrow=num_samples)
        grid_target = make_grid(target_log, nrow=num_samples)
        
        if self.writer:
            self.writer.add_image(f'Samples_from_{source_name}/01_Input_LR', grid_lr, epoch + 1)
            self.writer.add_image(f'Samples_from_{source_name}/02_{predicted_name}', grid_predicted, epoch + 1)
            self.writer.add_image(f'Samples_from_{source_name}/03_{target_name}', grid_target, epoch + 1)
            print(f"Logged sample RRDB images from {source_name} set to TensorBoard for epoch {epoch + 1}.") 
        
        self.model.train() 

    def train(self, 
              train_loader: DataLoader, 
              epochs: int, 
              val_loader: DataLoader = None,       
              val_every_n_epochs: int = 1,       
              accumulation_steps: int = 1,
              log_dir_param: str = None, 
              checkpoint_dir_param: str = None, 
              resume_checkpoint_path: str = None,
              save_every_n_epochs: int = 5,
              predict_residual: bool = False
             ):
        """
        Main training loop for the BasicRRDBNet.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train.
            val_loader (DataLoader): DataLoader for validation data (optional).
            val_every_n_epochs (int): Validation frequency in epochs.
            accumulation_steps (int): Number of steps for gradient accumulation.
            log_dir_param (str): Directory for logging.
            checkpoint_dir_param (str): Directory for saving checkpoints.
            resume_checkpoint_path (str): Path to checkpoint for resuming training.
            save_every_n_epochs (int): Frequency of saving checkpoints.
            predict_residual (bool): Flag to predict residuals instead of HR images.
        """

        self._setup_logging_and_checkpointing(log_dir_param, checkpoint_dir_param)
        initial_best_loss_from_resume = float('inf')

        if resume_checkpoint_path:
            self.start_epoch, self.global_step_optimizer, initial_best_loss_from_resume = self.load_checkpoint_for_resume(resume_checkpoint_path)
            self.best_loss = initial_best_loss_from_resume 
            self.batch_step_counter = self.start_epoch * len(train_loader) 
        
        print(f"Starting BasicRRDBNet training from epoch {self.start_epoch + 1}/{epochs} on device: {self.device}") 
        print(f"Accumulation steps: {accumulation_steps}") 
        if val_loader:
            print(f"Validation will be performed every {val_every_n_epochs} epoch(s).") 
            print(f"Initial best tracked loss (from resume or inf): {self.best_loss:.6f}") 
        else:
            print("No validation loader provided. 'best_loss' will track training loss.") 
            print(f"Initial best training loss (from resume or inf): {self.best_loss:.6f}") 

        current_accumulation_idx = 0
        if self.start_epoch == 0: 
            self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, epochs):
            self.model.train() 
            progress_bar = tqdm(total=len(train_loader), desc=f"Train RRDBNet Epoch {epoch+1}/{epochs}")
            epoch_total_train_loss = 0.0
            num_batches_in_epoch = 0

            for batch_idx, batch_data in enumerate(train_loader):
                loss = self._perform_one_batch_step(batch_data, predict_residual=predict_residual, is_training=True)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                current_accumulation_idx += 1
                updated_optimizer_this_step = False
                if current_accumulation_idx >= accumulation_steps:
                    self.optimizer.step()
                    if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    current_accumulation_idx = 0
                    self.global_step_optimizer += 1
                    updated_optimizer_this_step = True
                
                loss_value = loss.detach().item()
                epoch_total_train_loss += loss_value
                num_batches_in_epoch += 1
                
                if self.writer:
                    self.writer.add_scalar('Train/Loss_batch_step', loss_value, self.batch_step_counter)
                    if updated_optimizer_this_step and self.optimizer.param_groups:
                         self.writer.add_scalar('Train/LearningRate_optimizer_step', self.optimizer.param_groups[0]['lr'], self.global_step_optimizer)
                
                self.batch_step_counter += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss_value:.4f}", opt_steps=self.global_step_optimizer, lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
            
            progress_bar.close()
            mean_train_loss_epoch = epoch_total_train_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else float('nan')
            print(f"Epoch {epoch+1} Average Training Loss (RRDBNet): {mean_train_loss_epoch:.4f}") 
            if self.writer:
                self.writer.add_scalar('Train/Loss_epoch_avg', mean_train_loss_epoch, epoch + 1)
                if self.optimizer.param_groups:
                    self.writer.add_scalar('Train/LearningRate_epoch_end', self.optimizer.param_groups[0]['lr'], epoch + 1)

            current_loss_for_best_comparison = mean_train_loss_epoch
            is_current_loss_validation = False

            if val_loader and (epoch + 1) % val_every_n_epochs == 0:
                avg_val_loss_epoch = self._run_validation_epoch(val_loader, epoch, predict_residual)
                current_loss_for_best_comparison = avg_val_loss_epoch
                is_current_loss_validation = True
            
            if current_loss_for_best_comparison < self.best_loss:
                self.best_loss = current_loss_for_best_comparison
                self.save_checkpoint(epoch, self.best_loss, is_best_model=True, is_validation_loss=is_current_loss_validation)
                log_msg = "New best model saved with "
                if is_current_loss_validation:
                    log_msg += f"validation loss: {self.best_loss:.4f}"
                else:
                    log_msg += f"training loss: {self.best_loss:.4f}"
                print(f"Epoch {epoch+1}: {log_msg} to {self.best_checkpoint_path}") 
            
            if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
                if not (current_loss_for_best_comparison == self.best_loss and self.best_checkpoint_path == os.path.join(self.checkpoint_dir, f'rrdb_model_epoch_{epoch+1}.pth')):
                     self.save_checkpoint(epoch, mean_train_loss_epoch, is_best_model=False, is_validation_loss=False)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_plateau = current_loss_for_best_comparison if is_current_loss_validation else mean_train_loss_epoch
                    self.scheduler.step(metric_for_plateau)
                    print(f"ReduceLROnPlateau scheduler stepped with metric {metric_for_plateau:.4f}. New LR: {self.optimizer.param_groups[0]['lr']:.2e}") 
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR): 
                    self.scheduler.step()
                    print(f"Epoch-based scheduler ({type(self.scheduler).__name__}) stepped. New LR: {self.optimizer.param_groups[0]['lr']:.2e}") 
            
            if self.writer:
                self._log_rrdb_sample_images(train_loader, val_loader, epoch, predict_residual)


        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} accumulated gradients...") 
            self.optimizer.step()
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                 self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step_optimizer +=1
            print(f"Final gradients applied. Total optimizer steps: {self.global_step_optimizer}") 

        if self.writer:
            self.writer.close()
        print(f"BasicRRDBNet training finished. Final best tracked loss: {self.best_loss:.4f}") 
        if val_loader:
            print("(This was the best validation loss if validation was performed.)") 
        else:
            print("(This was the best training loss as no validation was performed.)") 

    def save_checkpoint(self, epoch, loss_value, is_best_model=False, is_validation_loss=False): 
        """
        Saves the model checkpoint.
        Args:
            epoch (int): Current epoch number.
            loss_value (float): Loss value to save.
            is_best_model (bool): Flag indicating if this is the best model.
            is_validation_loss (bool): Flag indicating if the loss is from validation.
        """

        if not self.checkpoint_dir:
            print("Warning: Checkpoint directory not set. Skipping save.") 
            return

        if is_best_model:
            save_path = self.best_checkpoint_path
        else:
            save_path = os.path.join(self.checkpoint_dir, f'rrdb_model_epoch_{epoch+1}.pth')
        
        trainer_configs_to_save = {
            'model_config': self.model_config,
            'optimizer_config': self.optimizer_config,
            'scheduler_config': self.scheduler_config
        }
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss_value, 
            'is_validation_loss': is_validation_loss, 
            'global_step_optimizer': self.global_step_optimizer,
            'trainer_configs': trainer_configs_to_save,
            'current_best_tracked_loss': self.best_loss 
        }
        torch.save(checkpoint_data, save_path)
        log_msg = f"Saved checkpoint to {save_path} (Epoch {epoch+1}, Loss in file: {loss_value:.4f}"
        if is_validation_loss:
            log_msg += " (Validation)"
        else:
            log_msg += " (Training)"
        log_msg += ")"
        print(log_msg) 

    def load_checkpoint_for_resume(self, checkpoint_path: str):
        """
        Loads a checkpoint for resuming training.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        Returns:
            start_epoch_res (int): The epoch to resume training from.
            global_step_optimizer_res (int): The global step of the optimizer.
            loaded_loss_metric_from_checkpoint (float): The loss metric from the checkpoint.
        """
        start_epoch_res = 0
        global_step_optimizer_res = 0
        loaded_loss_metric_from_checkpoint = float('inf') 

        if not os.path.isfile(checkpoint_path):
            print(f"Resume checkpoint not found at {checkpoint_path}. Training will start from scratch.") 
            return start_epoch_res, global_step_optimizer_res, loaded_loss_metric_from_checkpoint

        print(f"Loading checkpoint for resume from: {checkpoint_path}") 
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            loaded_trainer_configs = checkpoint.get('trainer_configs')
            if loaded_trainer_configs:
                print(f"Checkpoint was saved with trainer configs: {loaded_trainer_configs}") 
                if self.model_config != loaded_trainer_configs.get('model_config'):
                    print("WARNING: Model configuration in checkpoint differs from current trainer's model configuration.") 

            if 'model_state_dict' in checkpoint:
                self.model.to(self.device)
                incompatible_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if incompatible_keys.missing_keys: print(f"Resume Warning: Missing keys in model state_dict: {incompatible_keys.missing_keys}") 
                if incompatible_keys.unexpected_keys: print(f"Resume Info: Unexpected keys in model state_dict from checkpoint: {incompatible_keys.unexpected_keys}") 
                print("Model state loaded.") 
            
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v_opt in state.items():
                        if isinstance(v_opt, torch.Tensor): state[k] = v_opt.to(self.device)
                print("Optimizer state loaded.") 
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded.") 
                except Exception as e_sched_load:
                    print(f"Warning: Could not load scheduler state: {e_sched_load}. Scheduler may start fresh.") 

            start_epoch_res = checkpoint.get('epoch', -1) + 1 
            global_step_optimizer_res = checkpoint.get('global_step_optimizer', 0)
            
            loaded_loss_metric_from_checkpoint = checkpoint.get('current_best_tracked_loss', checkpoint.get('loss', float('inf')))
            
            print(f"Resuming training from epoch {start_epoch_res}, Optimizer steps: {global_step_optimizer_res}") 
            print(f"Loaded 'current_best_tracked_loss' (or 'loss') from checkpoint: {loaded_loss_metric_from_checkpoint:.4f}") 
            if 'is_validation_loss' in checkpoint:
                print(f"  The 'loss' value in the loaded checkpoint was a {'validation' if checkpoint['is_validation_loss'] else 'training'} loss.") 

        except Exception as e:
            print(f"Error loading checkpoint for resume: {e}. Training will start from scratch.") 
            start_epoch_res = 0
            global_step_optimizer_res = 0
            loaded_loss_metric_from_checkpoint = float('inf')
            
        return start_epoch_res, global_step_optimizer_res, loaded_loss_metric_from_checkpoint

    @staticmethod
    def load_model_for_evaluation(model_path: str, 
                                  device: str = 'cuda'):
        """
        Loads a pre-trained RRDBNet model for evaluation.
        Args:
            model_path (str): Path to the pre-trained model checkpoint.
            model_config (dict): Configuration dictionary for the model.
            device (str): Device to load the model on ('cuda' or 'cpu').
        Returns:
            model (torch.nn.Module): The loaded RRDBNet model in evaluation mode.
        """

        if not os.path.isfile(model_path):
            print(f"Evaluation model checkpoint not found at {model_path}. Returning uninitialized model in eval mode.") 
            return model.eval()

        print(f"Loading model for evaluation from: {model_path}") 
        checkpoint = torch.load(model_path, map_location=device)

        model_config = checkpoint.get('trainer_configs', {}).get('model_config', {})
        in_nc_val = model_config.get('in_nc', 3)
        out_nc_val = model_config.get('out_nc', 3)
        nf = model_config.get('num_feat', 64) 
        nb = model_config.get('num_block', 8) 
        gc_val = model_config.get('gc', 32)
        sr_scale_val = model_config.get('sr_scale', 4)

        if nf is None or nb is None or sr_scale_val is None: 
            raise ValueError("model_config must contain 'num_feat', 'num_block', and 'sr_scale'.") 

        model = RRDBNet(
            in_channels=in_nc_val, 
            out_channels=out_nc_val, 
            rrdb_in_channels=nf,  
            number_of_rrdb_blocks=nb, 
            growth_channels=gc_val,
            sr_scale=sr_scale_val
        )
        model.to(device)
        
        state_dict_to_load = None
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and not any(k.startswith('optimizer') or k == 'epoch' or k == 'trainer_configs' for k in checkpoint.keys()):
            state_dict_to_load = checkpoint 
        
        if state_dict_to_load:
            incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False)
            if incompatible_keys.missing_keys:
                print(f"Eval Load Warning: Missing keys in model: {incompatible_keys.missing_keys}") 
            if incompatible_keys.unexpected_keys:
                print(f"Eval Load Info: Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}") 
            print("Model weights loaded successfully for evaluation.") 
        else:
            print(f"Failed to load model weights for evaluation from {model_path}. Checkpoint format might be incorrect.") 
            
        return model.eval()