from rrdb_trainer import BasicRRDBNetTrainer
from utils.dataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torch

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
        
    print(f"Shape of concatenated_features: {concatenated_features.shape}") # Expected: [N, H, W], e.g., [192, 40, 40]

    # Ensure concatenated_features is 3D (Number of maps, Height, Width)
    if concatenated_features.dim() != 3:
        print(f"Error: concatenated_features is expected to be 3D (N, H, W), but got shape {concatenated_features.shape}")
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
    num_subplot_rows = (maps_to_plot_count + num_subplot_cols - 1) // num_subplot_cols
    
    # If maps_to_plot_count is 0, num_subplot_rows would be 0.
    # This case is already handled by the return above, but as a safeguard:
    if num_subplot_rows == 0 and maps_to_plot_count > 0:
        num_subplot_rows = 1 # Ensure at least one row if there are maps

    if num_subplot_rows == 0: # Should not happen if maps_to_plot_count > 0
        print("No rows for subplots, check logic for maps_to_plot_count.")
        return

    # Create the subplots
    fig, axes = plt.subplots(num_subplot_rows, num_subplot_cols, figsize=(15, 15))

    # Flatten the `axes` array for easy indexing, regardless of its original shape.
    # plt.subplots() can return:
    # 1. A single AxesSubplot object (if num_rows=1, num_cols=1)
    # 2. A 1D NumPy array of AxesSubplot objects (if num_rows=1 or num_cols=1, but not both 1)
    # 3. A 2D NumPy array of AxesSubplot objects (if num_rows > 1 and num_cols > 1)
    axes_flat = []
    if maps_to_plot_count > 0 : # Proceed only if there are subplots to create
        if num_subplot_rows == 1 and num_subplot_cols == 1:
            axes_flat = [axes] # axes is a single AxesSubplot object
        elif num_subplot_rows == 1 or num_subplot_cols == 1:
            # axes is a 1D NumPy array or can be treated as such
            axes_flat = axes.ravel() 
        else:
            # axes is a 2D NumPy array
            axes_flat = axes.flatten()

    # Plot each feature map
    for i in range(maps_to_plot_count):
        current_map_tensor = features_to_display[i] # This is an (H, W) tensor, e.g., (40,40)
        
        # Convert tensor to NumPy array for imshow:
        # .detach() creates a new tensor that doesn't require gradients.
        # .cpu() moves the tensor to the CPU (if it's on a GPU).
        # .numpy() converts the CPU tensor to a NumPy array.
        numpy_map = current_map_tensor.detach().cpu().numpy()
        
        ax = axes_flat[i] # Select the correct subplot
        ax.imshow(numpy_map, cmap='gray')
        ax.axis('off') # Turn off axis numbers and ticks

    # Hide any unused subplots in the grid
    # This loop runs from maps_to_plot_count up to the total number of subplots in the grid.
    for j in range(maps_to_plot_count, num_subplot_rows * num_subplot_cols):
        if j < len(axes_flat): # Check index is within bounds of the flattened axes array
            axes_flat[j].axis('off')
            
    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()

def main():
    predict_residual = False
    img_size = 160
    # config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 8, 'gc': 32, 'sr_scale': 4} 
    config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 17, 'gc': 32, 'sr_scale': 4}
    # model_path = 'checkpoints_rrdb/rrdb_model_best.pth'
    model_path = 'checkpoints_rrdb/rrdb_17_05_16/rrdb_model_best.pth'
    model = BasicRRDBNetTrainer.load_model_for_evaluation(model_path=model_path, model_config=config)
    dataset = ImageDataset(folder_path='data/', img_size=img_size, downscale_factor=4)

    item = np.random.randint(0, len(dataset))
    lr_img, up_img, hr_img, _ = dataset.__getitem__(item)
    lr_img = lr_img.unsqueeze(0).cuda()
    if predict_residual:
        res,feas = model(lr_img, get_fea=True)
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
    img_construct = img_construct.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    lr_img = lr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_img = hr_img.permute(1, 2, 0).detach().cpu().numpy()
    up_img = up_img.permute(1, 2, 0).detach().cpu().numpy()

    # Normalize the images to [0, 1] range for display
    lr_img = (lr_img + 1) / 2
    hr_img = (hr_img + 1) / 2
    img_construct = (img_construct + 1) / 2 
    up_img = (up_img + 1) / 2


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