from src.trainers.diffusion_trainer import DiffusionTrainer, ResidualGenerator
import matplotlib.pyplot as plt
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
import numpy as np
from src.diffusion_modules.unet import Unet
import torch
import os
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch

def process_image(image_path, img_size):
    """
    Processes an image:
    1. Resizes it so its smaller edge equals img_size (maintaining aspect ratio).
    2. Center crops to img_size x img_size.
    3. Converts to a PyTorch tensor (CHW, pixel values in [0,1]).
    4. Normalizes the tensor to the range [-1, 1].
    5. Adds a batch dimension.

    Args:
        image_path (str): Path to the input image.
        img_size (int): Target size for the smaller edge and the crop dimensions.

    Returns:
        torch.Tensor: Processed image as a PyTorch tensor in [B, C, H, W] format,
                      normalized to [-1, 1].
    """
    # Open the image using Pillow
    img = Image.open(image_path).convert("RGB") # Ensure image is in RGB format

    # Define the sequence of transformations
    # 1. T.Resize(img_size):
    #    Resizes the image so that its smaller dimension (height or width)
    #    becomes img_size, while maintaining the original aspect ratio.
    #    For example, if img_size=256:
    #    - an 800x600 image (W x H) becomes (256 * 800/600) x 256 ~ 341x256
    #    - a 600x800 image (W x H) becomes 256 x (256 * 800/600) ~ 256x341
    #    Note: T.Resize with an integer input applies to the smaller edge.
    #
    # 2. T.CenterCrop(img_size):
    #    Crops a square of img_size x img_size from the center of the resized image.
    #    - From a 341x256 image, it crops the central 256x256 portion.
    #    - From a 256x341 image, it crops the central 256x256 portion.
    #
    # 3. T.ToTensor():
    #    Converts the PIL Image (HWC, pixel values 0-255) to a PyTorch tensor (CHW, pixel values 0.0-1.0).
    transform = T.Compose([
        T.Resize(img_size),            # Smaller edge will be img_size, aspect ratio preserved
        T.CenterCrop(img_size),        # Crop the center to img_size x img_size
        T.ToTensor()                   # Convert to a PyTorch tensor (CHW format)
    ])

    # Apply the transformations
    tensor_img = transform(img)

    # Normalize to [-1, 1] range
    # T.ToTensor() scales pixels to [0, 1], so (x * 2.0) - 1.0 maps [0, 1] to [-1, 1]
    tensor_img = tensor_img * 2.0 - 1.0

    # Add batch dimension (BCHW)
    # The model likely expects input in the format [batch_size, channels, height, width]
    tensor_img = tensor_img.unsqueeze(0)

    return tensor_img

def plot_result(imgs):
    """
    Plots the low-resolution, upscaled, diffusion residual, constructed, and high-resolution images.
    """
    lr, up, diff_res, con = imgs
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(lr, interpolation='nearest')
    axs[0].set_title(f'LR Image\n({lr.shape[0]}x{lr.shape[1]})')
    axs[1].imshow(up)
    axs[1].set_title(f'RRDB Upscaled\n({up.shape[0]}x{up.shape[1]})')
    axs[2].imshow(diff_res)
    axs[2].set_title(f'Predicted Residual\n({diff_res.shape[0]}x{diff_res.shape[1]})')
    axs[3].imshow(con)
    axs[3].set_title(f'Refined Image\n({con.shape[0]}x{con.shape[1]})')
    for ax in axs:
        ax.set_aspect('auto')
        ax.axis('off')
    plt.tight_layout()
    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'result_quang.png')
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, float(fps), (w, h))
    print(f"Creating video with {len(intermediate_residuals_chw_list)} frames, saving to {video_path}...")
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

def main():
    """
    Main function to set up and run the diffusion model inference for a single image.
    """
    img_size = 80
    # rrdb_path = 'checkpoints/rrdb/rrdb_20250521-141800/rrdb_model_best.pth'
    rrdb_path = 'checkpoints/rrdb/rrdb_320/rrdb_model_best.pth'
    unet_path = 'checkpoints/diffusion/noise_320/diffusion_model_noise_best.pth'

    folder = 'faces_dataset_small'
    files = os.listdir(f'data/{folder}')
    if not files:
        print(f"No images found in 'data/{folder}'. Please check the directory.")
        return
    print(f"Found {len(files)} images in 'data/{folder}'.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=rrdb_path,
        device=device
    )

    unet = DiffusionTrainer.load_diffusion_unet(
        model_path=unet_path,
        device=device
    ).eval()
    generator = ResidualGenerator(img_size=img_size * 4, predict_mode='noise', device=device)

    while True:
        index = np.random.randint(0, len(files) - 1)
        # 5541
        print(f"Selected image index: {index}, filename: {files[index]}")
        img_path = os.path.join(f'data/{folder}', files[index])
        lr_img_batch = process_image(img_path, img_size).to(device)
        print(f"Input image shape: {lr_img_batch.shape}, dtype: {lr_img_batch.dtype}")
        print(f"Input image range: {lr_img_batch.min().item()} to {lr_img_batch.max().item()}")
        with torch.no_grad():
            up_lr_img_cuda, features_cuda = context_extractor(lr_img_batch, get_fea=True)

        intermediate_residuals_list = generator.generate_residuals(
            unet,
            features=features_cuda,
            num_images=1,
            num_inference_steps=50,
            return_intermediate_steps=True
        )

        final_predicted_residual_cuda = intermediate_residuals_list[-1]

        # base_for_video_cuda_chw = up_lr_img_cuda.squeeze(0)
        # residuals_for_video_chw_list = [res.squeeze(0) for res in intermediate_residuals_list]
        # video_filename = f"denoising_process_quang_img.mp4"
        # create_denoising_video(
        #     base_image_chw_tensor=base_for_video_cuda_chw,
        #     intermediate_residuals_chw_list=residuals_for_video_chw_list,
        #     output_filename=video_filename,
        #     fps=10
        # )

        lr_plot = (lr_img_batch.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
        up_lr_plot = (up_lr_img_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        final_residual_plot = (final_predicted_residual_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        constructed_img_cuda = torch.clamp(up_lr_img_cuda + final_predicted_residual_cuda, -1.0, 1.0)
        constructed_plot = (constructed_img_cuda.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2

        imgs_for_plot = [
            np.clip(lr_plot,0,1),
            np.clip(up_lr_plot,0,1),
            np.clip(final_residual_plot,0,1),
            np.clip(constructed_plot,0,1),
        ]
        plot_result(imgs_for_plot)

if __name__ == '__main__':
    main()
