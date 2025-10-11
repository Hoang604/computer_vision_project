from src.rfpp_modules.inverse_operators import *
from src.rfpp_modules.networks_edm import SongUNet, DhariwalUNet, EDMPrecondVel
from src.utils.rfpp_utils import straightness, parse_config, save_traj
import time
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
from torch.nn import DataParallel
from tqdm import tqdm
import argparse
import yaml
import logging
from torchvision.utils import save_image
from torchvision import transforms
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
sys.path.append('.')


logger = logging.getLogger(__name__)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

@torch.no_grad()
def sample_ode_generative(model, z1, N, arg, use_tqdm=True, solver='euler', label=None, inversion=False,
                          time_schedule=None, sampler='default', operator=None, ref_img_path="", output_dir=""):
    """
    Generates an image by solving the probability flow ODE with data consistency.

    This function takes an initial noise tensor and iteratively refines it by following the
    vector field predicted by the model. It incorporates a data consistency step to guide
    the generation process for inverse problems like super-resolution.

    Args:
        model (torch.nn.Module): The pre-trained generative model.
        z1 (torch.Tensor): The initial noise tensor of shape (batch, channels, H, W).
        N (int): The total number of discretization steps for the ODE solver.
        arg (argparse.Namespace): A namespace containing various configuration parameters,
            such as `gradient_scale` and `likebaseline`.
        use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
        solver (str, optional): The ODE solver to use ('euler' or 'heun'). Defaults to 'euler'.
        label (torch.Tensor, optional): Conditional labels for the model. Defaults to None.
        inversion (bool, optional): Flag for inversion process. Defaults to False.
        time_schedule (list, optional): A custom list of time steps to use. Defaults to None.
        sampler (str, optional): The sampler type. Defaults to 'default'.
        operator (object): The inverse problem operator (e.g., for super-resolution),
            which provides degradation and transpose methods.
        ref_img_path (str): The file path to the reference low-resolution image for data consistency.
        output_dir (str): Directory to save intermediate outputs like the degraded image.

    Returns:
        tuple: A tuple containing:
            - traj (list): A list of tensors representing the generation trajectory from z1 to z0.
            - x0hat_list (list): A list of predicted clean images (x0) at each step.
            - max_memory (float): The peak GPU memory allocated during the process in MB.
            - total_time (float): The total time taken for the sampling loop in seconds.
    """
    logger.info(
        "Starting sample_ode_generative | solver=%s | steps=%s | device=%s | ref_img=%s",
        solver,
        N,
        z1.device,
        ref_img_path
    )

    assert solver in ['euler', 'heun']
    assert len(z1.shape) == 4
    assert operator is not None
    assert ref_img_path != ""

    downsample = operator.degradation
    upsample = operator.degradation_transpose

    if inversion:
        assert sampler == 'default'
    tq = tqdm if use_tqdm else lambda x: x

    if solver == 'heun':
        if N % 2 == 0:
            logger.error("Invalid step count for Heun solver: N=%s", N)
            raise ValueError("N must be odd when using Heun's method.")
        N = (N + 1) // 2

    traj = []  # to store the trajectory
    x0hat_list = []
    z1 = z1.detach()
    z = z1.clone()
    batchsize = z.shape[0]

    if time_schedule is not None:
        time_schedule = time_schedule + [0]
        sigma_schedule = [t_ / (1-t_ + 1e-6) for t_ in time_schedule]
        logger.debug("Custom sigma schedule: %s", sigma_schedule)
    else:
        def t_func(i): return i / N
        if inversion:
            time_schedule = [t_func(i) for i in range(0, N)] + [1]
            time_schedule[0] = 1e-3
        else:
            time_schedule = [t_func(i) for i in reversed(range(1, N+1))] + [0]
            time_schedule[0] = 1-1e-5

    # Load and preprocess image
    image = Image.open(ref_img_path).convert("RGB").resize((64, 64))
    # show image
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    plt.show()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    preprocessed = transform(image).unsqueeze(0).to(z1.device)

    # Apply degradation and save
    downsampled = downsample(preprocessed)
    upsampled = upsample(downsampled)
    filename = os.path.splitext(os.path.basename(ref_img_path))[0]
    save_image(upsampled * 0.5 + 0.5,
               os.path.join(output_dir, f"{filename}_degraded.png"))

    config = model.module.config if hasattr(model, 'module') else model.config
    if config["label_dim"] > 0 and label is None:
        label = torch.randint(
            0, config["label_dim"], (batchsize,)).to(z1.device)
        label = F.one_hot(
            label, num_classes=config["label_dim"]).type(torch.float32)

    # Track max GPU memory usage when CUDA is available
    track_gpu_memory = torch.cuda.is_available() and z1.device.type == "cuda"
    if track_gpu_memory:
        logger.debug("Resetting CUDA peak memory stats before sampling loop.")
        torch.cuda.reset_peak_memory_stats()
    else:
        logger.info(
            "Skipping CUDA memory tracking (CUDA available: %s, tensor device: %s)",
            torch.cuda.is_available(), z1.device.type
        )
    start_time = time.time()

    traj.append(z.detach().clone())
    for i in tq(range(len(time_schedule[:-1]))):
        t = torch.ones((batchsize), device=z1.device) * time_schedule[i]
        t_next = torch.ones((batchsize), device=z1.device) * time_schedule[i+1]
        dt = t_next[0] - t[0]

        vt = model(z, t, label)
        x0hat = z - vt * t.view(-1, 1, 1, 1)  # x-prediction

        if solver == 'heun' and i < N - 1:
            z_next = z.detach().clone() + vt * dt
            vt_next = model(z_next, t_next, label)
            vt = (vt + vt_next) / 2
            x0hat = z_next - vt_next * t_next.view(-1, 1, 1, 1)

        x0hat_list.append(x0hat)

        if i < N-1:
            with torch.enable_grad():
                z = z.detach().requires_grad_()
                z_h = z - t[0] * vt

                if arg.likebaseline:
                    z = z.detach() - arg.gradient_scale * (downsample(z_h) - downsampled)
                else:
                    loss = torch.mean(torch.nn.functional.mse_loss(downsample(z_h), downsampled, reduction="none") *
                                      operator.get_mask(shape=downsampled.shape))
                    grads = torch.autograd.grad(loss, z)[0]
                    z = z.detach() - arg.gradient_scale * grads

        z = z.detach().clone() + vt * dt
        traj.append(z.detach().clone())

    end_time = time.time()
    if track_gpu_memory:
        max_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    else:
        max_memory = 0.0
    total_time = end_time - start_time

    logger.info(
        "Completed sample_ode_generative | duration=%.2fs | peak_mem=%.2fMB",
        total_time,
        max_memory
    )

    return traj, x0hat_list, max_memory, total_time

def upscale_image(input_image_path, inference_config_path='configs/config_rfpp_inference.yaml'):
    """Upscale a single image using RFPP super-resolution and log key diagnostics."""
    logger.info("Upscale request received | image=%s | config=%s", input_image_path, inference_config_path)

    # Load inference settings
    with open(inference_config_path, 'r') as f:
        arg_dict = yaml.safe_load(f)
    arg = argparse.Namespace(**arg_dict)

    # Parse model config
    config = parse_config(arg.config)
    arg.res = config['img_resolution']
    arg.input_nc = config['in_channels']
    arg.label_dim = config['label_dim']

    # Initialize model
    model_class = DhariwalUNet if config['unet_type'] == 'adm' else SongUNet
    flow_model = model_class(**config)
    flow_model = EDMPrecondVel(flow_model, use_fp16=config.get('use_fp16', False))

    # Load checkpoint
    if arg.ckpt is None or not os.path.exists(arg.ckpt):
        raise ValueError(f"Model checkpoint not found: {arg.ckpt}")
    flow_model.load_state_dict(torch.load(arg.ckpt, map_location="cpu"))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Upscale image using device: %s", device)
    if device.type == "cpu":
        logger.warning("CUDA not available; some GPU-only features will be skipped.")
    flow_model = flow_model.to(device).eval()

    # Setup inverse problem operator
    sampling_config = SamplingConfig()
    sampling_config.inverse_problem = arg.inverse_problem
    sampling_config.noise_sigma = arg.noise_sigma
    sampling_config.kernel_size = getattr(arg, 'kernel_size', 3)
    sampling_config.blur_sigma = getattr(arg, 'blur_sigma', 1.0)
    sampling_config.mask_size = getattr(arg, 'mask_size', 32)
    sampling_config.scale_factor = arg.scale_factor

    operator = get_inverse_operator(sampling_config, device=device)

    # Generate initial noise
    z = torch.randn(1, arg.input_nc, arg.res, arg.res).to(device)
    z = z * (1-1e-5)

    # Setup label if needed
    if arg.label_dim > 0:
        label_onehot = torch.eye(arg.label_dim, device=device)[
            torch.randint(0, arg.label_dim, (1,), device=device)]
    else:
        label_onehot = None

    # Parse time steps
    t_steps = [float(t) for t in arg.t_steps.split(",")
               ] if hasattr(arg, 't_steps') and arg.t_steps else None
    if t_steps:
        t_steps[0] = 1-1e-5

    # Create temporary output directory
    temp_dir = os.path.join(arg.dir, "tmp_upscale") if hasattr(arg, 'dir') else os.path.join(os.getcwd(), "tmp_upscale")
    os.makedirs(temp_dir, exist_ok=True)
    logger.debug("Temporary output directory: %s", temp_dir)

    # Perform super-resolution
    traj, x0_list, _, _ = sample_ode_generative(
        flow_model, z1=z, N=arg.N, arg=arg, use_tqdm=True,
        solver=arg.solver, label=label_onehot, time_schedule=t_steps,
        sampler=getattr(arg, 'sampler', 'default'), operator=operator,
        ref_img_path=input_image_path, output_dir=temp_dir
    )

    # Return the final upscaled image (normalized to [0, 1])
    upscaled_image = traj[-1][0] * 0.5 + 0.5
    logger.info("Upscale completed | image=%s", input_image_path)
    return upscaled_image

def main(inference_config_path='configs/config_rfpp_inference.yaml'):
    """
    Main function to set up and run the super-resolution inference process.

    This function orchestrates the entire inference pipeline. It loads configuration
    from a YAML file, initializes the model and its weights, sets up the environment
    (directories, device), and then triggers the sampling process.

    Args:
        inference_config_path (str, optional): Path to the inference configuration
            YAML file. Defaults to 'configs/config_rfpp_inference.yaml'.
    """
    # Load inference settings from the YAML file
    with open(inference_config_path, 'r') as f:
        arg_dict = yaml.safe_load(f)
    arg = argparse.Namespace(**arg_dict)

    logger.info("Starting batch inference | config=%s", inference_config_path)

    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config is not None
    config = parse_config(arg.config)
    arg.res = config['img_resolution']
    arg.input_nc = config['in_channels']
    arg.label_dim = config['label_dim']

    # Create directories
    if os.path.exists(os.path.join(arg.dir, "samples")):
        print(
            f"Directory {os.path.join(arg.dir, 'samples')} already exists. Exiting...")
        return

    os.makedirs(os.path.join(arg.dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "zs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "trajs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, 'tmp'), exist_ok=True)
    os.environ['TMPDIR'] = os.path.join(arg.dir, 'tmp')

    logger.info("Loaded model configuration: %s", arg.config)

    # Initialize model
    model_class = DhariwalUNet if config['unet_type'] == 'adm' else SongUNet
    flow_model = model_class(**config)

    # Setup device
    device_ids = str(arg.gpu).split(',')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using %s GPU(s): %s", len(device_ids), arg.gpu)
    else:
        logger.warning("CUDA not available; falling back to CPU for inference.")

    pytorch_total_params = sum(p.numel()
                               for p in flow_model.parameters()) / 1000000
    logger.info("Model parameter count: %.2fM", pytorch_total_params)

    flow_model = EDMPrecondVel(
        flow_model, use_fp16=config.get('use_fp16', False))

    if arg.ckpt is None or not os.path.exists(arg.ckpt):
        raise ValueError(f"Model checkpoint not found or not provided. Please update 'ckpt' in {inference_config_path}")
    flow_model.load_state_dict(torch.load(arg.ckpt, map_location="cpu"))

    if len(device_ids) > 1 and device.type == "cuda":
        flow_model = DataParallel(flow_model)
    flow_model = flow_model.to(device).eval()

    if arg.compile:
        flow_model = torch.compile(
            flow_model, mode="reduce-overhead", fullgraph=True)
        logger.info("Model compilation via torch.compile enabled.")

    # Save configs
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(vars(arg), f, indent=4)

    sample(arg, flow_model, device)


@torch.no_grad()
def sample(arg, model, device):
    """Run RFPP super-resolution sampling over all inputs in the configured directory.

    The routine constructs the inverse-operator requested in ``arg`` and iterates over every image inside ``arg.input_dir``. For each image it draws an initial latent ``z``, performs probability-flow ODE sampling via :func:`sample_ode_generative`, saves the resulting super-resolved outputs and optional latent trajectories, and aggregates runtime metrics.

    Args:
        arg (argparse.Namespace): Runtime configuration namespace loaded from the inference
            YAML file. Must contain directories, solver hyperparameters, and inverse problem
            settings such as ``scale_factor`` and ``gradient_scale``.
        model (torch.nn.Module): Pretrained RFPP model already placed on ``device`` and set to
            evaluation mode. The model is called inside the sampler to predict velocities.
        device (torch.device): CUDA device used for both latent tensors and inverse operator
            computations.

    Returns:
        None: Results, metrics, and optional trajectories are written to disk under
        ``arg.dir``. The function raises an error if inputs are missing or no images are found.
    """
    logger.info("Starting sampling loop | input_dir=%s | device=%s", arg.input_dir, device)

    output_dir = os.path.join(arg.dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Setup inverse problem
    sampling_config = SamplingConfig()
    sampling_config.inverse_problem = arg.inverse_problem
    sampling_config.noise_sigma = arg.noise_sigma
    # kernel_size, blur_sigma, mask_size are not in the yaml, so we need to handle them.
    sampling_config.kernel_size = arg.kernel_size
    sampling_config.blur_sigma = arg.blur_sigma
    sampling_config.mask_size = arg.mask_size
    sampling_config.scale_factor = arg.scale_factor

    operator = get_inverse_operator(sampling_config, device=device)
    logger.debug("Inverse operator configured: %s", sampling_config)

    if not os.path.exists(arg.input_dir):
        raise ValueError(f"Input directory not found. Please update 'input_dir' in your config.")

    input_images = sorted(
        glob.glob(os.path.join(arg.input_dir, "*.[pj][np][g]")))
    if not input_images:
        raise ValueError(f"No images found in {arg.input_dir}")
    logger.info("Found %d input images for sampling", len(input_images))

    straightness_list = []
    nfes = []
    total_times = []
    max_memories = []

    for i, img_path in enumerate(tqdm(input_images)):
        z = torch.randn(arg.batchsize, arg.input_nc,
                        arg.res, arg.res).to(device)

        if arg.label_dim > 0:
            label_onehot = torch.eye(arg.label_dim, device=device)[
                torch.randint(0, arg.label_dim, (z.shape[0],), device=device)]
        else:
            label_onehot = None

        if arg.solver in ['euler', 'heun']:
            t_steps = [float(t) for t in arg.t_steps.split(",")
                       ] if arg.t_steps else None
            if t_steps:
                t_steps[0] = 1-1e-5
            z = z * (1-1e-5)
            traj_uncond, traj_uncond_x0, max_memory, total_time = sample_ode_generative(
                model, z1=z, N=arg.N, arg=arg, use_tqdm=False, solver=arg.solver,
                label=label_onehot, time_schedule=t_steps, sampler=arg.sampler,
                operator=operator, ref_img_path=img_path, output_dir=output_dir
            )
            x0 = traj_uncond[-1]
            uncond_straightness = straightness(traj_uncond, mean=False)
            straightness_list.append(uncond_straightness)
            total_times.append(total_time)
            max_memories.append(max_memory)
        else:
            raise NotImplementedError(f"Solver {arg.solver} not implemented")

        if arg.save_traj:
            save_traj(traj_uncond, os.path.join(
                arg.dir, "trajs", f"{i:05d}_traj.png"))
            save_traj(traj_uncond_x0, os.path.join(
                arg.dir, "trajs", f"{i:05d}_traj_x0.png"))

        for idx in range(len(x0)):
            input_filename = os.path.basename(img_path)
            path_img = os.path.join(arg.dir, "samples", input_filename)
            path_z = os.path.join(
                arg.dir, "zs", input_filename.replace('.png', '.npy'))
            save_image(x0[idx] * 0.5 + 0.5, path_img)

            if arg.save_z:
                np.save(path_z, z[idx].cpu().numpy())

    # Save metrics
    if straightness_list:
        straightness_list = torch.stack(straightness_list).view(-1).cpu().numpy()
        straightness_mean = np.mean(straightness_list).item()
        straightness_std = np.std(straightness_list).item()
        print(f"straightness.shape: {straightness_list.shape}")
        print(
            f"straightness_mean: {straightness_mean}, straightness_std: {straightness_std}")
    else:
        straightness_mean, straightness_std = 0, 0

    nfes_mean = np.mean(nfes) if len(nfes) > 0 else arg.N
    print(f"nfes_mean: {nfes_mean}")
    avg_time = np.mean(total_times)
    avg_memory = np.mean(max_memories)
    print(f"average_time: {avg_time:.2f}s")
    print(f"average_max_memory: {avg_memory:.2f}GB")

    result_dict = {
        "straightness_mean": straightness_mean,
        "straightness_std": straightness_std,
        "nfes_mean": nfes_mean,
        "average_time": avg_time,
        "average_max_memory": avg_memory
    }
    with open(os.path.join(arg.dir, 'result_sampling.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)

    # Plot straightness distribution
    if straightness_list:
        plt.figure()
        plt.hist(straightness_list, bins=20)
        plt.savefig(os.path.join(arg.dir, "straightness.png"))
        plt.close()


if __name__ == "__main__":
    main()