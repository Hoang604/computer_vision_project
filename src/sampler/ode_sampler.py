import torch
import numpy as np
from scipy.integrate import solve_ivp
from diffusers import DDIMScheduler
from cosine_schedule import cosine_beta_schedule

class ODESampler:
    """
    Generates residuals using a generic ODE solver (from Scipy).
    This class is adapted to handle conditional models.
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
        self.predict_mode = predict_mode
        self.shape = None
        self.features = None # To store features for the ODE solver

        # Initialize the scheduler to get alpha/sigma values
        betas = cosine_beta_schedule(self.num_train_timesteps).to(self.device)
        scheduler_prediction_type = 'v_prediction' if self.predict_mode == 'v_prediction' else 'epsilon'
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=betas.cpu().numpy(),
            beta_schedule="trained_betas",
            prediction_type=scheduler_prediction_type
        )

    def _get_ode_derivative(self, s_val, x_s_numpy):
        """
        The derivative function for the ODE solver. It now uses `self.features`
        for conditioning.
        """
        with torch.no_grad():
            s_tensor = torch.tensor([s_val * self.num_train_timesteps], device=self.device)
            x_s = torch.from_numpy(x_s_numpy).reshape(self.shape).to(self.device).float()

            # Get alpha and sigma from the scheduler
            alpha_t = self.scheduler.alphas_cumprod[int(s_tensor.item())]**0.5
            sigma_t = (1 - alpha_t**2)**0.5

            # Predict the model output (noise or v)
            model_output = self.model(x_s, s_tensor, condition=self.features)

            # Convert to predicted noise (epsilon) if necessary
            if self.scheduler.config.prediction_type == "v_prediction":
                predicted_noise = self.scheduler.get_velocity(x_s, model_output, s_tensor)
            else: # 'epsilon'
                predicted_noise = model_output

            # Simplified ODE derivative (related to DDIM formulation)
            derivative = (x_s - alpha_t * predicted_noise) / sigma_t
            
            return derivative.flatten().cpu().numpy()

    @torch.no_grad()
    def generate_residuals(self, model, features, num_images=1, num_inference_steps=50):
        """
        The main sampling function using Scipy's solve_ivp.
        """
        self.model = model.to(self.device)
        self.shape = (num_images, self.img_channels, self.img_size, self.img_size)
        
        # Store features so the derivative function can access them
        self.features = [f.to(self.device) for f in features] if features is not None else None

        # Initial condition: pure random noise
        initial_noise = torch.randn(self.shape, device=self.device)
        
        # Time steps for evaluation (from 1.0 down to near 0)
        t_span = [1.0, 1e-4]
        t_eval = np.linspace(t_span[0], t_span[1], num_inference_steps + 1)
        
        print(f"Starting ODE sampling with Scipy ({num_inference_steps} steps)...")
        solution = solve_ivp(
            fun=self._get_ode_derivative,
            t_span=t_span,
            y0=initial_noise.flatten().cpu().numpy(),
            t_eval=t_eval,
            method='RK45'
        )
        print("Done.")

        # Get the final sample
        final_sample_flat = solution.y[:, -1]
        generated_residuals = torch.from_numpy(final_sample_flat).reshape(self.shape)
        
        return generated_residuals
