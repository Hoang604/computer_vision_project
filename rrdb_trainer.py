import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import datetime

# Import RRDBNet đã được sửa đổi từ module của bạn
from diffusion_modules import RRDBNet # HOẶC models.diffsr_modules import RRDBNet


class BasicRRDBNetTrainer:
    def __init__(self, 
                 model_config: dict,      # {'in_nc':3, 'out_nc':3, 'num_feat': 64, 'num_block': 16, 'gc': 32, 'sr_scale': 4}
                 optimizer_config: dict,  # {'lr': 0.0002, 'beta1': 0.9, 'beta2': 0.999}
                 scheduler_config: dict = None, # {'use_scheduler': True, 'decay_steps': 200000, 'gamma': 0.5}
                 logging_config: dict = None,   # {'exp_name': 'my_exp', 'log_dir_base': 'logs', 'checkpoint_dir_base': 'ckpts'}
                 device: str = 'cuda'):
        
        self.device = device
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config if scheduler_config else {'use_scheduler': False}
        self.logging_config = logging_config if logging_config else {}

        self.model = self._build_model().to(self.device)
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)

        self.start_epoch = 0
        self.global_step_optimizer = 0
        self.batch_step_counter = 0

        self.writer = None
        self.checkpoint_dir = None
        self.log_dir = None
        
        print(f"BasicRRDBNetTrainer initialized for device: {self.device}")
        print(f"Model Config: {self.model_config}")
        print(f"Optimizer Config: {self.optimizer_config}")
        print(f"Scheduler Config: {self.scheduler_config}")

    def _build_model(self):
        model = RRDBNet(
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
        if not self.scheduler_config.get('use_scheduler', False):
            return None
        # Mặc định dùng StepLR nếu được bật
        scheduler_type = self.scheduler_config.get('type', 'StepLR') 
        
        if scheduler_type == 'StepLR':
            decay_steps = self.scheduler_config.get('decay_steps', 200000)
            gamma = self.scheduler_config.get('gamma', 0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_to_schedule,
                                                        step_size=decay_steps,
                                                        gamma=gamma)
        # Thêm các loại scheduler khác ở đây nếu muốn
        # elif scheduler_type == 'CosineAnnealingLR':
        #     t_max = self.scheduler_config.get('t_max', 100) # Ví dụ: số epochs
        #     eta_min = self.scheduler_config.get('eta_min', 0)
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule, T_max=t_max, eta_min=eta_min)
        else:
            print(f"Warning: Scheduler type '{scheduler_type}' not recognized or 'use_scheduler' is false. No scheduler will be used.")
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
        print(f"Basic RRDBNet (Standalone) - Logging to: {self.log_dir}")
        print(f"Basic RRDBNet (Standalone) - Saving checkpoints to: {self.checkpoint_dir}")

    def _perform_one_train_step(self, batch_data):
        img_lr, _, img_hr, _= batch_data
        img_lr = img_lr.to(self.device)
        img_hr = img_hr.to(self.device)

        predicted_hr = self.model(img_lr)
        loss = F.l1_loss(predicted_hr, img_hr)
        return loss

    def train(self, 
              train_loader: DataLoader, 
              epochs: int, # epochs giờ là bắt buộc
              accumulation_steps: int = 1,
              log_dir_param: str = None, 
              checkpoint_dir_param: str = None, 
              resume_checkpoint_path: str = None,
              save_every_n_epochs: int = 5
             ):
        
        self._setup_logging_and_checkpointing(log_dir_param, checkpoint_dir_param)
        
        if resume_checkpoint_path:
            self.load_checkpoint_for_resume(resume_checkpoint_path)
            self.batch_step_counter = self.start_epoch * len(train_loader)

        print(f"Starting BasicRRDBNet training from epoch {self.start_epoch + 1}/{epochs} on device: {self.device}")
        print(f"Accumulation steps: {accumulation_steps}")

        current_accumulation_idx = 0
        if self.start_epoch == 0: 
            self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            progress_bar = tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}/{epochs}")
            epoch_total_loss = 0.0
            num_batches_in_epoch = 0

            for batch_data in train_loader:
                loss = self._perform_one_train_step(batch_data)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                current_accumulation_idx += 1
                if current_accumulation_idx >= accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    current_accumulation_idx = 0
                    self.global_step_optimizer += 1
                
                loss_value = loss.detach().item()
                epoch_total_loss += loss_value
                num_batches_in_epoch += 1
                
                if self.writer:
                    self.writer.add_scalar(f'Train/Loss_batch', loss_value, self.batch_step_counter)
                    if self.optimizer.param_groups:
                         self.writer.add_scalar(f'Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.batch_step_counter)
                
                self.batch_step_counter += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Train Epoch {epoch+1} Loss: {loss_value:.4f} OptSteps: {self.global_step_optimizer}")

            mean_epoch_loss = epoch_total_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else float('nan')
            if self.writer:
                self.writer.add_scalar(f'Train/Loss_epoch', mean_epoch_loss, epoch + 1)
            print(f"Epoch {epoch+1} Average Training Loss: {mean_epoch_loss:.4f}")
            
            if self.scheduler:
                self.scheduler.step()

            if (epoch + 1) % save_every_n_epochs == 0 or (epoch + 1) == epochs:
                self.save_checkpoint(epoch, mean_epoch_loss)
            progress_bar.close()

        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} accumulated gradients...")
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step_optimizer +=1
            print(f"Final gradients applied. Total optimizer steps: {self.global_step_optimizer}")

        if self.writer:
            self.writer.close()
        print(f"BasicRRDBNet training finished.")

    def save_checkpoint(self, epoch, current_loss):
        if not self.checkpoint_dir:
            print("Warning: Checkpoint directory not set. Skipping save.")
            return

        save_path = os.path.join(self.checkpoint_dir, f'rrdb_model_epoch_{epoch+1}.pth')
        # Lưu các config đã dùng để khởi tạo trainer vào checkpoint
        trainer_configs_to_save = {
            'model_config': self.model_config,
            'optimizer_config': self.optimizer_config,
            'scheduler_config': self.scheduler_config
            # logging_config không cần thiết phải lưu lại ở đây vì nó liên quan đến đường dẫn
        }
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': current_loss, 
            'global_step_optimizer': self.global_step_optimizer,
            'trainer_configs': trainer_configs_to_save 
        }
        torch.save(checkpoint_data, save_path)
        print(f"Saved checkpoint to {save_path} (Epoch {epoch+1})")

    def load_checkpoint_for_resume(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            print(f"Resume checkpoint not found at {checkpoint_path}. Training will start from scratch.")
            self.start_epoch = 0; self.global_step_optimizer = 0
            return

        print(f"Loading checkpoint for resume from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # So sánh config hiện tại với config trong checkpoint (tùy chọn)
        loaded_trainer_configs = checkpoint.get('trainer_configs')
        if loaded_trainer_configs:
            print(f"Checkpoint was saved with configs: {loaded_trainer_configs}")
            # TODO: Bạn có thể thêm logic so sánh/cảnh báo nếu config khác biệt đáng kể
            # Ví dụ: if self.model_config != loaded_trainer_configs.get('model_config'): ...

        # Tải trạng thái model
        # `_build_model` đã được gọi trong `__init__` với config hiện tại.
        # Nếu config trong checkpoint khác, bạn cần quyết định có build lại model không.
        # Để đơn giản, ở đây ta giả định cấu trúc model không đổi khi resume.
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded.")
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v_opt in state.items():
                    if isinstance(v_opt, torch.Tensor): state[k] = v_opt.to(self.device)
            print("Optimizer state loaded.")
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
            
        self.start_epoch = checkpoint.get('epoch', -1) + 1
        self.global_step_optimizer = checkpoint.get('global_optimizer_steps', 0)
        print(f"Resuming training from epoch {self.start_epoch}, Optimizer steps: {self.global_step_optimizer}")

    @staticmethod
    def load_model_for_evaluation(model_path: str, 
                                  model_config: dict, # BẮT BUỘC: config để build model
                                  device: str = 'cuda'):
        nf = model_config.get('num_feat')
        nb = model_config.get('num_block')
        gc_val = model_config.get('gc', 32)
        sr_scale_val = model_config.get('sr_scale')
        in_nc_val = model_config.get('in_nc', 3)
        out_nc_val = model_config.get('out_nc', 3)

        if nf is None or nb is None or sr_scale_val is None:
            raise ValueError("model_config must contain 'num_feat', 'num_block', and 'sr_scale'.")

        # Khởi tạo RRDBNet với các tham số từ model_config
        # Giả sử RRDBNet đã được sửa để nhận sr_scale qua __init__
        model = RRDBNet(in_nc=in_nc_val, out_nc=out_nc_val, nf=nf, nb=nb, gc=gc_val, sr_scale=sr_scale_val)
        model.to(device)

        if not os.path.isfile(model_path):
            print(f"Evaluation model checkpoint not found at {model_path}. Returning uninitialized model in eval mode.")
            return model.eval()

        print(f"Loading model for evaluation from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        state_dict_to_load = None
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and not any(k.startswith('optimizer') or k == 'epoch' for k in checkpoint.keys()):
            # Heuristic: Nếu là dict và không có key điển hình của checkpoint đầy đủ -> có thể là state_dict
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