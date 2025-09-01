import torch
from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler
from tqdm.auto import tqdm
from cosine_schedule import cosine_beta_schedule


class FastResidualGeneratorDPM:
    """
    The recommended approach.
    Generates residuals using the specialized DPM-Solver, adapted for your
    conditional model and class structure.
    """
    def __init__(self,
                 img_channels=3,
                 img_size=256,
                 device='cuda',
                 num_train_timesteps=1000,
                 predict_mode='epsilon'):

        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.predict_mode = predict_mode

        # Initialize a base scheduler with your cosine schedule
        betas = cosine_beta_schedule(self.num_train_timesteps).cpu()
        
        self.solver = UniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            trained_betas=betas.numpy(),
            prediction_type=predict_mode,
            solver_order=2,
            # steps_offset=1
        )
        print(f"FastResidualGeneratorDPM initialized with {type(self.solver).__name__}.")

    @torch.no_grad()
    def generate_residuals(self, model, features, num_images=1, num_inference_steps=50):
        """
        The main sampling function using DPM-Solver.
        """
        model.eval()
        model.to(self.device)
        
        # Set the number of inference steps for the solver
        self.solver.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.solver.timesteps

        # Initial condition: pure random noise, scaled correctly by the solver
        image_latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device
        ) * self.solver.init_noise_sigma
        
        if features is not None:
            features = [f.to(self.device) for f in features]

        print(f"Generating {num_images} residuals using {num_inference_steps} steps "
              f"with {type(self.solver).__name__}...")

        # The reverse denoising loop
        for t in tqdm(timesteps, desc="Generating residuals with DPM-Solver"):
            # The model call now correctly includes the 'condition' argument
            t_for_model = t.unsqueeze(0).expand(num_images).to(self.device)

            model_output = model(image_latents, t_for_model, condition=features)
            print(f"Model output min/max at step {t.item()}: "
                  f"{model_output.min().item()} / {model_output.max().item()}")
            
            # Use the solver's step function to compute the previous sample
            scheduler_output = self.solver.step(model_output, t, image_latents)
            print(f"lantent min/max at step {t.item()}: "
                  f"{image_latents.min().item()} / {image_latents.max().item()}")
            image_latents = scheduler_output.prev_sample
            print(f"Updated latent min/max at step {t.item()}: "
                  f"{image_latents.min().item()} / {image_latents.max().item()}")

        generated_residuals = image_latents
        print("Residual generation complete.")
        return generated_residuals