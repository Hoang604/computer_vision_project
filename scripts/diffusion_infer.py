from src.utils.bicubic import upscale_image
from tqdm import tqdm
import cv2
import torch
import numpy as np
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
import matplotlib.pyplot as plt
from src.data_handling.dataset import ImageDatasetRRDB
from src.trainers.diffusion_trainer import DiffusionTrainer, ResidualGenerator
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_result(imgs):
    """
    Plots the low-resolution, upscaled, diffusion residual, constructed, and high-resolution images.
    """
    lr, up, diff_res, con, hr_img = imgs

    bicubic_upscaled = upscale_image(lr, scale_factor=4)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(lr, interpolation='nearest')
    axs[0].set_title(f'LR Image\n({lr.shape[0]}x{lr.shape[1]})')

    axs[1].imshow(bicubic_upscaled)
    axs[1].set_title(
        f'Bicubic Upscaled\n({diff_res.shape[0]}x{diff_res.shape[1]})')

    axs[2].imshow(up)
    axs[2].set_title(f'RRDB Upscaled\n({up.shape[0]}x{up.shape[1]})')

    axs[3].imshow(con)
    axs[3].set_title(f'RRDB + Diffusion\n({con.shape[0]}x{con.shape[1]})')

    axs[4].imshow(hr_img)
    axs[4].set_title(f'Ground Truth HR\n({hr_img.shape[0]}x{hr_img.shape[1]})')
    for ax in axs:
        ax.set_aspect('auto')
        ax.axis('off')
    plt.tight_layout()
    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'result_comparison.png')
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
    plt.close(fig)


def plot_inference_result(result_dict, output_path='inference_outputs/inference_result.png'):
    """
    Plots inference results in a 2x3 grid.
    Top row: LR, Bicubic, RRDB
    Bottom row: Residual, Final, HR Ground Truth

    All images occupy equal frame sizes, but original resolutions are preserved
    (LR image will appear pixelated without matplotlib interpolation).

    Args:
        result_dict (dict): Dictionary containing:
            - 'lr': LR image (numpy array, uint8, RGB)
            - 'rrdb_upscaled': RRDB upscaled image
            - 'diffusion_residual': Diffusion residual
            - 'final_upscaled': Final constructed image
            - 'hr_ground_truth': HR ground truth (can be None)
        output_path (str): Path to save the output plot
    """
    lr = result_dict['lr']
    rrdb = result_dict['rrdb_upscaled']
    residual = result_dict['diffusion_residual']
    final = result_dict['final_upscaled']
    hr = result_dict['hr_ground_truth']

    # Create bicubic upscaled version from LR
    bicubic = upscale_image(lr, scale_factor=4)

    # Create 2x3 subplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: LR, Bicubic, RRDB
    axs[0, 0].imshow(lr, interpolation='nearest')
    axs[0, 0].set_title(
        f'LR Image\n({lr.shape[1]}x{lr.shape[0]})', fontsize=12, fontweight='bold')

    axs[0, 1].imshow(bicubic, interpolation='nearest')
    axs[0, 1].set_title(
        f'Bicubic Upscaled\n({bicubic.shape[1]}x{bicubic.shape[0]})', fontsize=12, fontweight='bold')

    axs[0, 2].imshow(rrdb, interpolation='nearest')
    axs[0, 2].set_title(
        f'RRDB Upscaled\n({rrdb.shape[1]}x{rrdb.shape[0]})', fontsize=12, fontweight='bold')

    # Bottom row: Residual, Final, HR Ground Truth
    axs[1, 0].imshow(residual, interpolation='nearest')
    axs[1, 0].set_title(
        f'Diffusion Residual\n({residual.shape[1]}x{residual.shape[0]})', fontsize=12, fontweight='bold')

    axs[1, 1].imshow(final, interpolation='nearest')
    axs[1, 1].set_title(
        f'RRDB + Diffusion (Final)\n({final.shape[1]}x{final.shape[0]})', fontsize=12, fontweight='bold')

    if hr is not None:
        axs[1, 2].imshow(hr, interpolation='nearest')
        axs[1, 2].set_title(
            f'Ground Truth HR\n({hr.shape[1]}x{hr.shape[0]})', fontsize=12, fontweight='bold')
    else:
        axs[1, 2].text(0.5, 0.5, 'No Ground Truth',
                       ha='center', va='center', fontsize=14, transform=axs[1, 2].transAxes)
        axs[1, 2].set_title('Ground Truth HR\n(Not Available)',
                            fontsize=12, fontweight='bold')

    # Set equal aspect and remove axes for all subplots
    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Inference result plot saved to {output_path}")
    plt.close(fig)


def create_denoising_video(base_image_chw_tensor, intermediate_residuals_chw_list, output_filename="denoising_process.mp4", fps=10):
    """
    Creates a video of the denoising process.

    Args:
        base_image_chw_tensor (torch.Tensor): The base upscaled image (e.g., from RRDBNet)
                                             as a CHW tensor, range [-1, 1], on CUDA.
        intermediate_residuals_chw_list (list[torch.Tensor]): A list of CHW tensors,
                                                              each representing the predicted residual
                                                              at an intermediate denoising step.
                                                              Tensors are on CUDA, range approx [-1, 1].
        output_filename (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    if not intermediate_residuals_chw_list:
        print("No intermediate residuals to create a video.")
        return
    c, h, w = base_image_chw_tensor.shape
    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writer = cv2.VideoWriter(video_path, fourcc, float(fps), (w, h))
    print(
        f"Creating video with {len(intermediate_residuals_chw_list)} frames, saving to {video_path}...")
    for i, residual_tensor in enumerate(tqdm(intermediate_residuals_chw_list, desc="Generating video frames")):
        residual_tensor = residual_tensor.to(base_image_chw_tensor.device)
        current_image_tensor = base_image_chw_tensor + residual_tensor
        current_image_tensor = torch.clamp(current_image_tensor, -1.0, 1.0)
        current_image_normalized = (current_image_tensor + 1.0) / 2.0
        frame_hwc = current_image_normalized.permute(1, 2, 0)
        frame_np = (frame_hwc.cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()
    print(f"Video saved successfully to {video_path}")


def inference(config, file_path: str):
    """
    Runs diffusion model inference on a single image specified by file_path.
    Args:
        config: Configuration object with necessary parameters.
        file_path (str): Path to the input image file.
        hr_image_path (str, optional): Path to the high-resolution ground truth image.

    ---
    Returns: 
        dict: Dictionary containing:
            - 'lr': LR image as numpy array (uint8, RGB, range [0, 255])
            - 'rrdb_upscaled': RRDB upscaled image as numpy array (uint8, RGB, range [0, 255])
            - 'diffusion_residual': Diffusion residual as numpy array (uint8, RGB, range [0, 255])
            - 'final_upscaled': Final constructed image as numpy array (uint8, RGB, range [0, 255])
            - 'hr_ground_truth': HR ground truth image as numpy array (uint8, RGB, range [0, 255]) if hr_image_path provided, else None
    """
    print(
        f"Running diffusion inference on image: {file_path} with config: {config}")

    # Extract configuration parameters
    rrdb_path = config.rrdb_path
    output_folder = config.output_folder
    num_timesteps = config.num_timesteps
    lr_size = config.lr_size
    unet_path = config.unet_path if lr_size < 60 else config.unet_320_path

    # Set up models and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=rrdb_path,
        device=device
    )
    context_extractor.eval()
    unet = DiffusionTrainer.load_diffusion_unet(
        unet_path, verbose=True, device=device)
    unet.eval()

    generator = ResidualGenerator(
        img_size=lr_size * 4, predict_mode='noise', device=device)

    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess the input image
    image_pil = Image.open(file_path).convert('RGB')
    hr_image_pil = image_pil.copy()
    lr_image_pil = image_pil.resize((lr_size, lr_size), Image.BICUBIC)
    lr_image_np = np.array(lr_image_pil).astype(np.float32) / 255.0
    lr_image_tensor_chw = torch.from_numpy(
        lr_image_np.transpose(2, 0, 1) * 2.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        up_lr_img_cuda, features_cuda = context_extractor(
            lr_image_tensor_chw, get_fea=True)
    intermediate_residuals_list = generator.generate_residuals(
        unet,
        features=features_cuda,
        num_images=1,
        num_inference_steps=num_timesteps,
        return_intermediate_steps=False
    )

    final_predicted_residual_cuda = intermediate_residuals_list[-1]
    constructed_img_cuda = torch.clamp(
        up_lr_img_cuda + final_predicted_residual_cuda, -1.0, 1.0)
    constructed_img_cpu = constructed_img_cuda.squeeze(0).permute(
        1, 2, 0).detach().cpu().numpy()
    constructed_img_uint8 = (
        (constructed_img_cpu + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # Convert LR image to uint8
    lr_image_uint8 = (lr_image_np * 255.0).astype(np.uint8)

    # Convert RRDB upscaled image to uint8
    rrdb_upscaled_cpu = up_lr_img_cuda.squeeze(
        0).permute(1, 2, 0).detach().cpu().numpy()
    rrdb_upscaled_uint8 = ((rrdb_upscaled_cpu + 1.0) /
                           2.0 * 255.0).astype(np.uint8)

    # Convert diffusion residual to uint8
    residual_cpu = final_predicted_residual_cuda.squeeze(
        0).permute(1, 2, 0).detach().cpu().numpy()
    residual_uint8 = ((residual_cpu + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # Load HR ground truth if provided
    hr_ground_truth_uint8 = None
    hr_image_pil = hr_image_pil.resize(
        (lr_size * 4, lr_size * 4), Image.BICUBIC)
    hr_ground_truth_uint8 = np.array(hr_image_pil).astype(np.uint8)

    return {
        'lr': lr_image_uint8,
        'rrdb_upscaled': rrdb_upscaled_uint8,
        'diffusion_residual': residual_uint8,
        'final_upscaled': constructed_img_uint8,
        'hr_ground_truth': hr_ground_truth_uint8
    }


def main():
    """
    Main function to set up and run the diffusion model inference for a single image.
    """
    img_size = 160
    rrdb_path = 'checkpoints/rrdb/rrdb_20250521-141800/rrdb_model_best.pth'
    unet_path = 'checkpoints/diffusion/noise_20250526-070738/diffusion_model_noise_best.pth'
    img_folder = 'preprocessed_data/rrdb_processed_train'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=rrdb_path,
        device=device
    )
    context_extractor.eval()

    dataset = ImageDatasetRRDB(
        preprocessed_folder_path=img_folder,
        img_size=img_size,
        downscale_factor=4,
        apply_hflip=False
    )
    if len(dataset) == 0:
        print(f"No data found in {img_folder}. Exiting.")
        return

    unet = DiffusionTrainer.load_diffusion_unet(
        unet_path, verbose=True, device=device)
    unet.eval()

    generator = ResidualGenerator(
        img_size=img_size, predict_mode='noise', device=device)
    while True:
        item_idx = np.random.randint(0, len(dataset))
        print(f"\nProcessing sample dataset index: {item_idx}")

        lr_img_tensor_chw, _, hr_original_tensor_chw, _ = dataset[item_idx]
        lr_img_batch_cuda = lr_img_tensor_chw.unsqueeze(0).to(device)

        with torch.no_grad():
            up_lr_img_cuda, features_cuda = context_extractor(
                lr_img_batch_cuda, get_fea=True)

        intermediate_residuals_list = generator.generate_residuals(
            unet,
            features=features_cuda,
            num_images=1,
            num_inference_steps=20,
            return_intermediate_steps=False
        )

        final_predicted_residual_cuda = intermediate_residuals_list[-1]

        # base_for_video_cuda_chw = up_lr_img_cuda.squeeze(0)
        # residuals_for_video_chw_list = [res.squeeze(0) for res in intermediate_residuals_list]
        # video_filename = f"denoising_process_item_{item_idx}.mp4"
        # create_denoising_video(
        #     base_image_chw_tensor=base_for_video_cuda_chw,
        #     intermediate_residuals_chw_list=residuals_for_video_chw_list,
        #     output_filename=video_filename,
        #     fps=10
        # )

        lr_plot = (lr_img_tensor_chw.permute(1, 2, 0).cpu().numpy() + 1) / 2
        up_lr_plot = (up_lr_img_cuda.squeeze(0).permute(
            1, 2, 0).detach().cpu().numpy() + 1) / 2
        final_residual_plot = (final_predicted_residual_cuda.squeeze(
            0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        constructed_img_cuda = torch.clamp(
            up_lr_img_cuda + final_predicted_residual_cuda, -1.0, 1.0)
        constructed_plot = (constructed_img_cuda.squeeze(
            0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        hr_original_plot = (hr_original_tensor_chw.permute(
            1, 2, 0).cpu().numpy() + 1) / 2

        imgs_for_plot = [
            np.clip(lr_plot, 0, 1),
            np.clip(up_lr_plot, 0, 1),
            np.clip(final_residual_plot, 0, 1),
            np.clip(constructed_plot, 0, 1),
            np.clip(hr_original_plot, 0, 1)
        ]
        plot_result(imgs_for_plot)
        input()


if __name__ == '__main__':
    main()
