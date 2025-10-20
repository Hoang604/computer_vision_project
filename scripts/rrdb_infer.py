from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
from src.data_handling.dataset import ImageDataset
from src.utils.bicubic import upscale_image
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torch
from PIL import Image
import os


def plot_feature_map(list_of_tensors: List[torch.Tensor]):
    """
    Plot the feature map of the model.
    Args:
        list_of_tensors: List of PyTorch Tensors. Each tensor is expected to be
                         like (1, C, H, W) or (1, H, W), where the first dimension
                         is a batch dimension of size 1 that can be squeezed.
                         After squeezing, tensors should be (C, H, W) or (H, W).
    """
    # Check if the input list is empty
    if not list_of_tensors:
        print("Input list_of_tensors is empty. Nothing to plot.")
        return

    processed_tensors = []
    for i in range(len(list_of_tensors)):
        # .squeeze(0) removes the first dimension if it's 1.
        # This is typically used to remove a batch dimension of size 1.
        # e.g., (1, C, H, W) -> (C, H, W) or (1, H, W) -> (H, W)
        processed_tensors.append(list_of_tensors[i].squeeze(0))

    # Concatenate tensors.
    # If processed_tensors contained elements like (C1, H, W), (C2, H, W), etc.,
    # torch.cat(processed_tensors, dim=0) results in (C1+C2+..., H, W).
    # This should match the example output torch.Size([192, 40, 40]),
    # where 192 is the total number of individual 2D feature maps (channels),
    # and each map is 40x40.
    try:
        concatenated_features = torch.cat(processed_tensors, dim=0)
    except RuntimeError as e:
        print(f"Error during torch.cat: {e}")
        print("This might happen if tensors have incompatible shapes for concatenation after squeezing.")
        print("Shapes of processed tensors before cat:")
        for i, t in enumerate(processed_tensors):
            print(f"  Shape of processed_tensors[{i}]: {t.shape}")
        return

    # Expected: [N, H, W], e.g., [192, 40, 40]
    print(f"Shape of concatenated_features: {concatenated_features.shape}")

    # Ensure concatenated_features is 3D (Number of maps, Height, Width)
    if concatenated_features.dim() != 3:
        print(
            f"Error: concatenated_features is expected to be 3D (N, H, W), but got shape {concatenated_features.shape}")
        return

    # Select up to the first 64 feature maps for plotting.
    # Each slice concatenated_features[i] will be an (H, W) map.
    num_actual_maps = concatenated_features.shape[0]
    maps_to_plot_count = min(num_actual_maps, 64)

    if maps_to_plot_count == 0:
        print("No feature maps to plot after concatenation and selection.")
        return

    # features_to_display will be a tensor of shape (maps_to_plot_count, H, W)
    features_to_display = concatenated_features[:maps_to_plot_count]

    # Determine subplot grid configuration.
    # We'll use a fixed number of columns (e.g., 8) and calculate the necessary rows.
    num_subplot_cols = 8
    # Calculate rows needed using ceiling division
    num_subplot_rows = (maps_to_plot_count +
                        num_subplot_cols - 1) // num_subplot_cols

    # If maps_to_plot_count is 0, num_subplot_rows would be 0.
    # This case is already handled by the return above, but as a safeguard:
    if num_subplot_rows == 0 and maps_to_plot_count > 0:
        num_subplot_rows = 1  # Ensure at least one row if there are maps

    if num_subplot_rows == 0:  # Should not happen if maps_to_plot_count > 0
        print("No rows for subplots, check logic for maps_to_plot_count.")
        return

    # Create the subplots
    fig, axes = plt.subplots(
        num_subplot_rows, num_subplot_cols, figsize=(15, 15))

    # Flatten the `axes` array for easy indexing, regardless of its original shape.
    # plt.subplots() can return:
    # 1. A single AxesSubplot object (if num_rows=1, num_cols=1)
    # 2. A 1D NumPy array of AxesSubplot objects (if num_rows=1 or num_cols=1, but not both 1)
    # 3. A 2D NumPy array of AxesSubplot objects (if num_rows > 1 and num_cols > 1)
    axes_flat = []
    if maps_to_plot_count > 0:  # Proceed only if there are subplots to create
        if num_subplot_rows == 1 and num_subplot_cols == 1:
            axes_flat = [axes]  # axes is a single AxesSubplot object
        elif num_subplot_rows == 1 or num_subplot_cols == 1:
            # axes is a 1D NumPy array or can be treated as such
            axes_flat = axes.ravel()
        else:
            # axes is a 2D NumPy array
            axes_flat = axes.flatten()

    # Plot each feature map
    for i in range(maps_to_plot_count):
        # This is an (H, W) tensor, e.g., (40,40)
        current_map_tensor = features_to_display[i]

        # Convert tensor to NumPy array for imshow:
        # .detach() creates a new tensor that doesn't require gradients.
        # .cpu() moves the tensor to the CPU (if it's on a GPU).
        # .numpy() converts the CPU tensor to a NumPy array.
        numpy_map = current_map_tensor.detach().cpu().numpy()

        ax = axes_flat[i]  # Select the correct subplot
        ax.imshow(numpy_map, cmap='gray')
        ax.axis('off')  # Turn off axis numbers and ticks

    # Hide any unused subplots in the grid
    # This loop runs from maps_to_plot_count up to the total number of subplots in the grid.
    for j in range(maps_to_plot_count, num_subplot_rows * num_subplot_cols):
        if j < len(axes_flat):  # Check index is within bounds of the flattened axes array
            axes_flat[j].axis('off')

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()


def plot_inference_result(result_dict, output_path='inference_outputs/rrdb_inference_result.png'):
    """
    Plots RRDB inference results in a 2x2 grid.
    Top row: LR, Bicubic
    Bottom row: RRDB, HR Ground Truth

    All images occupy equal frame sizes, but original resolutions are preserved
    (LR image will appear pixelated without matplotlib interpolation).

    Args:
        result_dict (dict): Dictionary containing:
            - 'lr': LR image (numpy array, uint8, RGB)
            - 'rrdb_upscaled': RRDB upscaled image
            - 'hr_ground_truth': HR ground truth (can be None)
        output_path (str): Path to save the output plot
    """
    lr = result_dict['lr']
    rrdb = result_dict['rrdb_upscaled']
    hr = result_dict['hr_ground_truth']

    # Create bicubic upscaled version from LR
    bicubic = upscale_image(lr, scale_factor=4)

    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Top row: LR, Bicubic
    axs[0, 0].imshow(lr, interpolation='nearest')
    axs[0, 0].set_title(
        f'LR Image\n({lr.shape[1]}x{lr.shape[0]})', fontsize=12, fontweight='bold')

    axs[0, 1].imshow(bicubic, interpolation='nearest')
    axs[0, 1].set_title(
        f'Bicubic Upscaled\n({bicubic.shape[1]}x{bicubic.shape[0]})', fontsize=12, fontweight='bold')

    # Bottom row: RRDB, HR Ground Truth
    axs[1, 0].imshow(rrdb, interpolation='nearest')
    axs[1, 0].set_title(
        f'RRDB Upscaled\n({rrdb.shape[1]}x{rrdb.shape[0]})', fontsize=12, fontweight='bold')

    if hr is not None:
        axs[1, 1].imshow(hr, interpolation='nearest')
        axs[1, 1].set_title(
            f'Ground Truth HR\n({hr.shape[1]}x{hr.shape[0]})', fontsize=12, fontweight='bold')
    else:
        axs[1, 1].text(0.5, 0.5, 'No Ground Truth',
                       ha='center', va='center', fontsize=14, transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Ground Truth HR\n(Not Available)',
                            fontsize=12, fontweight='bold')

    # Set equal aspect and remove axes for all subplots
    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"RRDB inference result plot saved to {output_path}")
    plt.close(fig)


def inference(config, file_path: str):
    """
    Runs RRDB model inference on a single image specified by file_path.
    Args:
        config: Configuration object with necessary parameters.
        file_path (str): Path to the input image file.

    ---
    Returns: 
        dict: Dictionary containing:
            - 'lr': LR image as numpy array (uint8, RGB, range [0, 255])
            - 'rrdb_upscaled': RRDB upscaled image as numpy array (uint8, RGB, range [0, 255])
            - 'hr_ground_truth': HR ground truth image as numpy array (uint8, RGB, range [0, 255]) if hr_image_path provided, else None
    """
    print(
        f"Running RRDB inference on image: {file_path} with config: {config}")

    # Extract configuration parameters
    output_folder = config.output_folder
    lr_size = config.lr_size
    rrdb_path = config.rrdb_path if lr_size < 60 else config.rrdb_320_path

    # Set up model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=rrdb_path,
        device=device
    )
    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    # Load and preprocess the input image
    image_pil = Image.open(file_path).convert('RGB')
    hr_image_pil = image_pil.copy()
    lr_image_pil = image_pil.resize((lr_size, lr_size), Image.BICUBIC)
    lr_image_np = np.array(lr_image_pil).astype(np.float32) / 255.0
    lr_image_tensor_chw = torch.from_numpy(
        lr_image_np.transpose(2, 0, 1) * 2.0 - 1.0).unsqueeze(0).to(device)

    # Run RRDB inference
    with torch.no_grad():
        rrdb_output_cuda, _ = model(lr_image_tensor_chw, get_fea=True)

    # Convert RRDB output to uint8
    rrdb_output_cpu = rrdb_output_cuda.squeeze(
        0).permute(1, 2, 0).detach().cpu().numpy()
    rrdb_output_uint8 = ((rrdb_output_cpu + 1.0) /
                         2.0 * 255.0).astype(np.uint8)

    # Convert LR image to uint8
    lr_image_uint8 = (lr_image_np * 255.0).astype(np.uint8)

    # Load HR ground truth if provided
    hr_image_pil = hr_image_pil.resize(
        (lr_size * 4, lr_size * 4), Image.BICUBIC)
    hr_ground_truth_uint8 = np.array(hr_image_pil).astype(np.uint8)

    return {
        'lr': lr_image_uint8,
        'rrdb_upscaled': rrdb_output_uint8,
        'hr_ground_truth': hr_ground_truth_uint8
    }


def main():
    predict_residual = False
    img_size = 160
    # config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 8, 'gc': 32, 'sr_scale': 4}
    config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64,
              'num_block': 17, 'gc': 32, 'sr_scale': 4}
    # model_path = 'checkpoints_rrdb/rrdb_model_best.pth'
    model_path = 'checkpoints_rrdb/rrdb_17_05_16/rrdb_model_best.pth'
    model = BasicRRDBNetTrainer.load_model_for_evaluation(
        model_path=model_path, model_config=config)
    dataset = ImageDataset(folder_path='data/',
                           img_size=img_size, downscale_factor=4)

    item = np.random.randint(0, len(dataset))
    lr_img, up_img, hr_img, _ = dataset.__getitem__(item)
    lr_img = lr_img.unsqueeze(0).cuda()
    if predict_residual:
        res, feas = model(lr_img, get_fea=True)
        res = res.squeeze(0)
        img_construct = res + up_img
        img_construct = img_construct.permute(1, 2, 0).detach().cpu().numpy()
    else:
        img_construct, feas = model(lr_img, get_fea=True)
    # plot_feature_map(feas[2::3])
    print(f"img_construct shape: {img_construct.shape}")
    print(f"img_construct min: {img_construct.min()}")
    print(f"img_construct max: {img_construct.max()}")
    print(f"img_construct mean: {img_construct.mean()}")
    img_construct = img_construct.squeeze(
        0).permute(1, 2, 0).detach().cpu().numpy()

    lr_img = lr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_img = hr_img.permute(1, 2, 0).detach().cpu().numpy()
    up_img = up_img.permute(1, 2, 0).detach().cpu().numpy()

    # Normalize the images to [0, 1] range for display
    lr_img = (lr_img + 1) / 2
    print(f"lr_img min: {lr_img.min()}")
    print(f"lr_img max: {lr_img.max()}")
    hr_img = (hr_img + 1) / 2
    print(f"hr_img min: {hr_img.min()}")
    print(f"hr_img max: {hr_img.max()}")
    img_construct = (img_construct + 1) / 2
    print(f"img_construct min: {img_construct.min()}")
    print(f"img_construct max: {img_construct.max()}")
    up_img = (up_img + 1) / 2
    print(f"up_img min: {up_img.min()}")
    print(f"up_img max: {up_img.max()}")

    # plot the images
    plt.subplot(1, 4, 1)
    plt.title('LR Image')
    plt.imshow(lr_img)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title('HR Image')
    plt.imshow(hr_img)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title('Bicubic Image')
    plt.imshow(up_img)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title('Constructed Image')
    plt.imshow(img_construct)
    plt.axis('off')
    # plt.savefig('test2.png')
    plt.show()


if __name__ == '__main__':
    main()
