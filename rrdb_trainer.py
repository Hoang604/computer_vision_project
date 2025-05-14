import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import datetime
import numpy as np # Import numpy for float('inf') if not already there, though float('inf') is standard

# Import RRDBNet from your modules
from diffusion_modules import RRDBNet # OR models.diffsr_modules import RRDBNet


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
        
        # Variables for best model saving
        self.best_loss = float('inf')
        self.best_checkpoint_path = None # Will be set in _setup_logging_and_checkpointing

        self.writer = None
        self.checkpoint_dir = None
        self.log_dir = None
        
        print(f"BasicRRDBNetTrainer initialized for device: {self.device}")
        print(f"Model Config: {self.model_config}")
        print(f"Optimizer Config: {self.optimizer_config}")
        print(f"Scheduler Config: {self.scheduler_config}")

    def _build_model(self):
        # Assuming RRDBNet constructor takes these specific keys from model_config
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
        optimizer = torch.optim.Adam(model_to_optimize.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        return optimizer

    def _build_scheduler(self, optimizer_to_schedule):
        scheduler_type = self.scheduler_config.get('type', 'none').lower() # Ensure case-insensitivity
        
        if scheduler_type == 'none':
            print("No learning rate scheduler will be used.")
            return None
        
        print(f"Building scheduler of type: {scheduler_type}")
        if scheduler_type == 'steplr':
            # StepLR steps based on optimizer steps.
            step_size = self.scheduler_config.get('step_lr_step_size', 200000) # Number of optimizer steps
            gamma = self.scheduler_config.get('step_lr_gamma', 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_to_schedule,
                                                        step_size=step_size, 
                                                        gamma=gamma)
            print(f"  StepLR configured with step_size={step_size}, gamma={gamma}")
        elif scheduler_type == 'cosineannealinglr':
            # CosineAnnealingLR steps per epoch.
            t_max = self.scheduler_config.get('cosine_t_max', 100) # Typically total epochs
            eta_min = self.scheduler_config.get('cosine_eta_min', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule, 
                                                                    T_max=t_max, 
                                                                    eta_min=eta_min)
            print(f"  CosineAnnealingLR configured with T_max={t_max}, eta_min={eta_min}")
        elif scheduler_type == 'cosineannealingwarmrestarts':
            # CosineAnnealingWarmRestarts steps per epoch.
            t_0 = self.scheduler_config.get('cosine_warm_t_0', 10) # Number of epochs for the first restart
            t_mult = self.scheduler_config.get('cosine_warm_t_mult', 1) # Factor to increase T_i after a restart
            eta_min = self.scheduler_config.get('cosine_warm_eta_min', 0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_to_schedule,
                                                                             T_0=t_0,
                                                                             T_mult=t_mult,
                                                                             eta_min=eta_min)
            print(f"  CosineAnnealingWarmRestarts configured with T_0={t_0}, T_mult={t_mult}, eta_min={eta_min}")
        elif scheduler_type == 'reducelronplateau':
            # ReduceLROnPlateau steps per epoch, requires a metric.
            mode = self.scheduler_config.get('plateau_mode', 'min')
            factor = self.scheduler_config.get('plateau_factor', 0.1)
            patience = self.scheduler_config.get('plateau_patience', 10) # Epochs
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

    def _perform_one_train_step(self, batch_data):
        img_lr, _, img_hr, _ = batch_data 
        img_lr = img_lr.to(self.device)
        img_hr = img_hr.to(self.device)

        predicted_hr = self.model(img_lr) 
        loss = F.l1_loss(predicted_hr, img_hr) 
        return loss

    def train(self, 
              train_loader: DataLoader, 
              epochs: int, 
              accumulation_steps: int = 1,
              log_dir_param: str = None, 
              checkpoint_dir_param: str = None, 
              resume_checkpoint_path: str = None,
              save_every_n_epochs: int = 5
             ):
        
        self._setup_logging_and_checkpointing(log_dir_param, checkpoint_dir_param)
        self.best_loss = float('inf') 

        if resume_checkpoint_path:
            self.start_epoch, self.global_step_optimizer, loaded_loss_from_ckpt = self.load_checkpoint_for_resume(resume_checkpoint_path)
            self.best_loss = loaded_loss_from_ckpt 
            self.batch_step_counter = self.start_epoch * len(train_loader) 
        
        print(f"Starting BasicRRDBNet training from epoch {self.start_epoch + 1}/{epochs} on device: {self.device}")
        print(f"Accumulation steps: {accumulation_steps}")
        print(f"Initial best loss: {self.best_loss}")

        current_accumulation_idx = 0
        if self.start_epoch == 0: 
            self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            progress_bar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}/{epochs}")
            epoch_total_loss = 0.0
            num_batches_in_epoch = 0

            for batch_idx, batch_data in enumerate(train_loader):
                loss = self._perform_one_train_step(batch_data)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                current_accumulation_idx += 1
                updated_optimizer_this_step = False
                if current_accumulation_idx >= accumulation_steps:
                    self.optimizer.step()
                    
                    # Scheduler step for StepLR (if it's based on optimizer steps)
                    if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                        self.scheduler.step()
                        # print(f"StepLR stepped at optimizer_step {self.global_step_optimizer}. New LR: {self.optimizer.param_groups[0]['lr']:.2e}")


                    self.optimizer.zero_grad()
                    current_accumulation_idx = 0
                    self.global_step_optimizer += 1
                    updated_optimizer_this_step = True
                
                loss_value = loss.detach().item()
                epoch_total_loss += loss_value
                num_batches_in_epoch += 1
                
                if self.writer:
                    self.writer.add_scalar(f'Train/Loss_batch_step', loss_value, self.batch_step_counter)
                    if updated_optimizer_this_step and self.optimizer.param_groups:
                         self.writer.add_scalar(f'Train/LearningRate_optimizer_step', self.optimizer.param_groups[0]['lr'], self.global_step_optimizer)
                
                self.batch_step_counter += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss_value:.4f}", opt_steps=self.global_step_optimizer, lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
            
            progress_bar.close()

            mean_epoch_loss = epoch_total_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else float('nan')
            print(f"Epoch {epoch+1} Average Training Loss: {mean_epoch_loss:.4f}")
            if self.writer:
                self.writer.add_scalar(f'Train/Loss_epoch_avg', mean_epoch_loss, epoch + 1)
                if self.optimizer.param_groups:
                    self.writer.add_scalar(f'Train/LearningRate_epoch_end', self.optimizer.param_groups[0]['lr'], epoch + 1)
            
            # Scheduler step for epoch-based schedulers
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(mean_epoch_loss)
                    print(f"ReduceLROnPlateau scheduler stepped with metric {mean_epoch_loss:.4f}. New LR (from optimizer): {self.optimizer.param_groups[0]['lr']:.2e}")
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR): # StepLR is handled per optimizer step
                    self.scheduler.step()
                    print(f"Epoch-based scheduler ({type(self.scheduler).__name__}) stepped. New LR (from optimizer): {self.optimizer.param_groups[0]['lr']:.2e}")


            if mean_epoch_loss < self.best_loss:
                self.best_loss = mean_epoch_loss
                self.save_checkpoint(epoch, self.best_loss, is_best_model=True)
                print(f"Epoch {epoch+1}: New best model saved with loss: {self.best_loss:.4f} to {self.best_checkpoint_path}")

            if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
                if not (mean_epoch_loss == self.best_loss and self.best_checkpoint_path == os.path.join(self.checkpoint_dir, f'rrdb_model_epoch_{epoch+1}.pth')):
                     self.save_checkpoint(epoch, mean_epoch_loss, is_best_model=False)

        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} accumulated gradients...")
            self.optimizer.step()
            # Final StepLR step if applicable
            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
                 self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step_optimizer +=1
            print(f"Final gradients applied. Total optimizer steps: {self.global_step_optimizer}")

        if self.writer:
            self.writer.close()
        print(f"BasicRRDBNet training finished. Final best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, epoch, loss_for_this_checkpoint, is_best_model=False): 
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
            'loss': loss_for_this_checkpoint, 
            'global_step_optimizer': self.global_step_optimizer,
            'trainer_configs': trainer_configs_to_save,
            'best_loss_tracker': self.best_loss 
        }
        torch.save(checkpoint_data, save_path)
        print(f"Saved checkpoint to {save_path} (Epoch {epoch+1}, Loss in file: {loss_for_this_checkpoint:.4f})")

    def load_checkpoint_for_resume(self, checkpoint_path: str):
        start_epoch_res = 0
        global_step_optimizer_res = 0
        loaded_loss_res = float('inf') 

        if not os.path.isfile(checkpoint_path):
            print(f"Resume checkpoint not found at {checkpoint_path}. Training will start from scratch.")
            return start_epoch_res, global_step_optimizer_res, loaded_loss_res

        print(f"Loading checkpoint for resume from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            loaded_trainer_configs = checkpoint.get('trainer_configs')
            if loaded_trainer_configs:
                print(f"Checkpoint was saved with trainer configs: {loaded_trainer_configs}")
                if self.model_config != loaded_trainer_configs.get('model_config'):
                    print("WARNING: Model configuration in checkpoint differs from current trainer's model configuration.")
                    print(f"  Checkpoint model_config: {loaded_trainer_configs.get('model_config')}")
                    print(f"  Current trainer model_config: {self.model_config}")
                    print("  This might lead to issues if model architecture changed. Ensure consistency or rebuild model if necessary.")

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
                # Rebuild scheduler with potentially loaded config BEFORE loading its state
                # This ensures the scheduler instance matches the one whose state was saved.
                # However, self.scheduler is already built in __init__. If scheduler_config changed,
                # it should ideally be handled by re-initializing the trainer or having a more dynamic _build_scheduler.
                # For now, we assume scheduler_config from init is compatible with loaded state.
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded.")
                except Exception as e_sched_load:
                    print(f"Warning: Could not load scheduler state: {e_sched_load}. Scheduler may start from a fresh state or its last_epoch might be incorrect.")

            start_epoch_res = checkpoint.get('epoch', -1) + 1 
            global_step_optimizer_res = checkpoint.get('global_step_optimizer', 0)
            loaded_loss_res = checkpoint.get('best_loss_tracker', checkpoint.get('loss', float('inf')))
            
            print(f"Resuming training from epoch {start_epoch_res}, Optimizer steps: {global_step_optimizer_res}")
            print(f"Loaded loss for best_loss initialization: {loaded_loss_res:.4f}")

        except Exception as e:
            print(f"Error loading checkpoint for resume: {e}. Training will start from scratch.")
            start_epoch_res = 0
            global_step_optimizer_res = 0
            loaded_loss_res = float('inf')
            
        return start_epoch_res, global_step_optimizer_res, loaded_loss_res

    @staticmethod
    def load_model_for_evaluation(model_path: str, 
                                  model_config: dict, 
                                  device: str = 'cuda'):
        in_nc_val = model_config.get('in_nc', 3)
        out_nc_val = model_config.get('out_nc', 3)
        nf = model_config.get('num_feat', 64) 
        nb = model_config.get('num_block', 17) 
        gc_val = model_config.get('gc', 32)
        sr_scale_val = model_config.get('sr_scale', 4)

        if nf is None or nb is None or sr_scale_val is None: 
            raise ValueError("model_config must contain 'num_feat', 'num_block', and 'sr_scale' (or their equivalents for RRDBNet).")

        model = RRDBNet(
            in_channels=in_nc_val, 
            out_channels=out_nc_val, 
            rrdb_in_channels=nf,  
            number_of_rrdb_blocks=nb, 
            growth_channels=gc_val,
            sr_scale=sr_scale_val
        )
        model.to(device)

        if not os.path.isfile(model_path):
            print(f"Evaluation model checkpoint not found at {model_path}. Returning uninitialized model in eval mode.")
            return model.eval()

        print(f"Loading model for evaluation from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
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
            print(f"Failed to load model weights for evaluation from {model_path}. Checkpoint format might be incorrect or 'model_state_dict' missing.")
            
        return model.eval()
