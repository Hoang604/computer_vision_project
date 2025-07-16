from src.trainers.diffusion_trainer import DiffusionTrainer, ResidualGenerator
from src.data_handling.dataset import ImageDatasetRRDB
import matplotlib.pyplot as plt
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
import numpy as np
from src.diffusion_modules.unet import Unet
import torch
import os
import cv2
from tqdm import tqdm

def plot_result(imgs):
    """
    Plots the low-resolution, upscaled, diffusion residual, constructed, and high-resolution images.
    """
    lr, up, diff_res, con, hr_img = imgs
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(lr, interpolation='nearest')
    axs[0].set_title(f'LR Image\n({lr.shape[0]}x{lr.shape[1]})')
    axs[1].imshow(up)
    axs[1].set_title(f'RRDB Upscaled\n({up.shape[0]}x{up.shape[1]})')
    axs[2].imshow(diff_res)
    axs[2].set_title(f'Predicted Residual\n({diff_res.shape[0]}x{diff_res.shape[1]})')
    axs[3].imshow(con)
    axs[3].set_title(f'Refined Image\n({con.shape[0]}x{con.shape[1]})')
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
    print(f"Creating video with {len(intermediate_residuals_chw_list)} frames, saving to {video_path}...")
    for _, residual_tensor in enumerate(tqdm(intermediate_residuals_chw_list, desc="Generating video frames")):
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

    unet = DiffusionTrainer.load_diffusion_unet(unet_path, verbose=True, device=device)
    unet.eval()

    generator = ResidualGenerator(img_size=img_size, predict_mode='noise', device=device)

    item_idx = np.random.randint(0, len(dataset))
    print(f"\nProcessing sample dataset index: {item_idx}")

    lr_img_tensor_chw, _, hr_original_tensor_chw, _ = dataset[item_idx]
    lr_img_batch_cuda = lr_img_tensor_chw.unsqueeze(0).to(device)

    with torch.no_grad():
        up_lr_img_cuda, features_cuda = context_extractor(lr_img_batch_cuda, get_fea=True)

    intermediate_residuals_list = generator.generate_residuals(
        unet,
        features=features_cuda,
        num_images=1,
        num_inference_steps=50,
        return_intermediate_steps=True
    )

    final_predicted_residual_cuda = intermediate_residuals_list[-1]

    base_for_video_cuda_chw = up_lr_img_cuda.squeeze(0)
    residuals_for_video_chw_list = [res.squeeze(0) for res in intermediate_residuals_list]
    video_filename = f"denoising_process_item_{item_idx}.mp4"
    create_denoising_video(
        base_image_chw_tensor=base_for_video_cuda_chw,
        intermediate_residuals_chw_list=residuals_for_video_chw_list,
        output_filename=video_filename,
        fps=10
    )

    lr_plot = (lr_img_tensor_chw.permute(1, 2, 0).cpu().numpy() + 1) / 2
    up_lr_plot = (up_lr_img_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    final_residual_plot = (final_predicted_residual_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    constructed_img_cuda = torch.clamp(up_lr_img_cuda + final_predicted_residual_cuda, -1.0, 1.0)
    constructed_plot = (constructed_img_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    hr_original_plot = (hr_original_tensor_chw.permute(1, 2, 0).cpu().numpy() + 1) / 2

    imgs_for_plot = [
        np.clip(lr_plot,0,1),
        np.clip(up_lr_plot,0,1),
        np.clip(final_residual_plot,0,1),
        np.clip(constructed_plot,0,1),
        np.clip(hr_original_plot,0,1)
    ]
    plot_result(imgs_for_plot)

if __name__ == '__main__':
    main()
