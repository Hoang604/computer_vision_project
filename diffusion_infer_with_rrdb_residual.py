# diffusion_infer.py
import torch
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
import torchvision.transforms.functional as TF

from utils.dataset import ImageDatasetRRDB # For loading a single sample if needed, or use PIL directly
from diffusion_modules import Unet
from diffusion_trainer import DiffusionTrainer, ResidualGenerator # ResidualGenerator is key here
from rrdb_trainer import BasicRRDBNetTrainer # For loading RRDBNet models

def plot_result_v2(imgs_dict, save_path="result_v2.png"):
    """
    Plots the LR, HR_original, HR_RRDB_upscaled, Residual_true (HR_orig - HR_RRDB),
    Residual_predicted_by_diffusion, and Final_HR_constructed.
    """
    lr = imgs_dict.get('lr_img_display')
    hr_orig = imgs_dict.get('hr_orig_display')
    hr_rrdb = imgs_dict.get('hr_rrdb_display')
    res_true = imgs_dict.get('residual_true_display') # (HR_orig - HR_RRDB)
    res_pred = imgs_dict.get('residual_predicted_display') # Predicted by diffusion
    hr_final = imgs_dict.get('hr_final_constructed_display') # (HR_RRDB + res_pred)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12)) # Adjusted size for better viewing
    fig.suptitle("Super-Resolution Inference Results (RRDB + Diffusion Refinement)", fontsize=16)

    axs[0, 0].imshow(lr)
    axs[0, 0].set_title('1. Low-Resolution Input')

    axs[0, 1].imshow(hr_rrdb)
    axs[0, 1].set_title('2. HR by RRDBNet (Base SR)')
    
    axs[0, 2].imshow(hr_orig)
    axs[0, 2].set_title('3. Ground Truth HR')

    axs[1, 0].imshow(res_true)
    axs[1, 0].set_title('4. True Residual (GT_HR - RRDB_HR)')
    
    axs[1, 1].imshow(res_pred)
    axs[1, 1].set_title('5. Predicted Residual (Diffusion)')

    axs[1, 2].imshow(hr_final)
    axs[1, 2].set_title('6. Final Constructed HR (RRDB_HR + Pred_Res)')

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(save_path)
    print(f"Result image saved to {save_path}") 
    plt.show()


def main_infer(args):
    """
    Main function to set up and run diffusion model inference
    with the new pipeline (RRDB base + Diffusion refinement).
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    # 1. Load the RRDBNet model that was used for generating the base HR_RRDB
    # This configuration MUST match the RRDBNet used in preprocess_data_with_rrdb.py
    rrdb_config_for_base_sr = {
        'in_nc': args.img_channels,
        'out_nc': args.img_channels,
        'num_feat': args.rrdb_num_feat_preproc, # Use specific params for this RRDBNet
        'num_block': args.rrdb_num_block_preproc,
        'gc': args.rrdb_gc_preproc,
        'sr_scale': args.downscale_factor
    }
    try:
        if not args.rrdb_weights_path_for_base_sr or not os.path.exists(args.rrdb_weights_path_for_base_sr):
            raise FileNotFoundError(f"RRDBNet weights for base SR not found at: {args.rrdb_weights_path_for_base_sr}")
        rrdb_model_for_base_sr = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path_for_base_sr,
            model_config=rrdb_config_for_base_sr,
            device=device
        )
        rrdb_model_for_base_sr.eval()
        print(f"RRDBNet for base SR loaded from: {args.rrdb_weights_path_for_base_sr}") 
    except Exception as e:
        print(f"Error loading RRDBNet for base SR: {e}") 
        return

    # 2. Load the RRDBNet model used as the Context Extractor for the U-Net
    # This might be the same as rrdb_model_for_base_sr, or a different one.
    rrdb_config_for_context = {
        'in_nc': args.img_channels,
        'out_nc': args.img_channels,
        'num_feat': args.rrdb_num_feat_context, # Use specific params for this RRDBNet
        'num_block': args.rrdb_num_block_context,
        'gc': args.rrdb_gc_context,
        'sr_scale': args.downscale_factor
    }
    try:
        if not args.rrdb_weights_path_context_extractor or not os.path.exists(args.rrdb_weights_path_context_extractor):
            raise FileNotFoundError(f"RRDBNet weights for context extractor not found at: {args.rrdb_weights_path_context_extractor}")
        context_extractor_model = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=args.rrdb_weights_path_context_extractor,
            model_config=rrdb_config_for_context,
            device=device
        )
        context_extractor_model.eval()
        print(f"Context Extractor RRDBNet loaded from: {args.rrdb_weights_path_context_extractor}") 
    except Exception as e:
        print(f"Error loading Context Extractor RRDBNet: {e}") 
        return

    # 3. Load the U-Net model (Diffusion model)
    unet_model = Unet(
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults),
        use_attention=args.use_attention,
        cond_dim=args.rrdb_num_feat_context, # Should match output features of context_extractor_model
        rrdb_num_blocks=args.rrdb_num_block_context # For Unet's internal cond_proj logic
    ).to(device)
    
    try:
        if not args.unet_weights_path or not os.path.exists(args.unet_weights_path):
            raise FileNotFoundError(f"U-Net weights not found at: {args.unet_weights_path}")
        DiffusionTrainer.load_model_weights(device, unet_model, args.unet_weights_path, verbose=True)
        unet_model.eval()
        print(f"U-Net model loaded from: {args.unet_weights_path}") 
    except Exception as e:
        print(f"Error loading U-Net model: {e}") 
        return

    # 4. Prepare input LR image
    # For simplicity, load a single LR image. You can adapt this to loop through a folder.
    # We need the original HR for comparison if available.
    # Option 1: Load from your preprocessed data (if you want to test on a specific preprocessed sample)
    # lr_tensor_path = os.path.join(args.preprocessed_data_folder_for_infer, 'lr', args.sample_basename + ".pt")
    # hr_orig_tensor_path = os.path.join(args.preprocessed_data_folder_for_infer, 'hr_original', args.sample_basename + ".pt")
    # lr_img_tensor = torch.load(lr_tensor_path).unsqueeze(0).to(device) # Add batch dim
    # hr_orig_for_comparison = torch.load(hr_orig_tensor_path).to(device) # No batch dim needed for comparison display

    # Option 2: Load an arbitrary LR image (e.g., from a test folder, or downscale an HR on the fly)
    # For this example, let's assume we load an HR image and downscale it.
    # If you have an actual LR image, load it directly.
    try:
        if not args.input_hr_image_for_infer or not os.path.exists(args.input_hr_image_for_infer):
            raise FileNotFoundError(f"Input HR image for inference not found at: {args.input_hr_image_for_infer}")

        hr_pil = Image.open(args.input_hr_image_for_infer).convert("RGB")
        # Resize to standard HR size if it's not already
        if hr_pil.size != (args.img_size, args.img_size):
            hr_pil = hr_pil.resize((args.img_size, args.img_size), Image.BICUBIC)

        hr_orig_for_comparison_0_1 = TF.to_tensor(hr_pil) # (C, H, W), [0,1]
        hr_orig_for_comparison = (hr_orig_for_comparison_0_1 * 2.0 - 1.0).to(device) # [-1,1]

        # Create LR version for input
        lr_h = args.img_size // args.downscale_factor
        lr_w = args.img_size // args.downscale_factor
        lr_img_tensor_for_input = TF.resize(
            hr_orig_for_comparison.clone(),
            [lr_h, lr_w],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True
        ).unsqueeze(0).to(device) # Add batch dim, ensure on device

        print(f"Loaded and prepared LR input from: {args.input_hr_image_for_infer}, LR shape: {lr_img_tensor_for_input.shape}") 

    except Exception as e:
        print(f"Error preparing input image for inference: {e}") 
        return


    # 5. Generate HR_RRDB (Base Super-Resolution)
    with torch.no_grad():
        hr_rrdb_tensor = rrdb_model_for_base_sr(lr_img_tensor_for_input).squeeze(0) # Remove batch dim for consistency
    print(f"Generated HR_RRDB, shape: {hr_rrdb_tensor.shape}") 

    # 6. Perform Diffusion-based Residual Prediction
    # The ResidualGenerator's img_size should match the HR size (args.img_size)
    # Its predict_mode should match how the U-Net was trained.
    residual_predictor = ResidualGenerator(
        img_channels=args.img_channels,
        img_size=args.img_size,
        device=device,
        num_train_timesteps=args.diffusion_timesteps, # Timesteps U-Net was trained with
        predict_mode=args.diffusion_mode # 'v_prediction' or 'noise'
    )
    
    with torch.no_grad():
        # lr_img_tensor_for_input already has batch dimension
        predicted_residual_tensor = residual_predictor.generate_residuals(
            model=unet_model,
            low_resolution_image=lr_img_tensor_for_input, # This is the LR image
            context_extractor=context_extractor_model,   # This extracts features from LR
            num_images=lr_img_tensor_for_input.shape[0], # Should be 1 in this example
            num_inference_steps=args.diffusion_inference_steps
        ).squeeze(0) # Remove batch dim
    print(f"Predicted residual by diffusion, shape: {predicted_residual_tensor.shape}") 

    # 7. Construct Final HR Image
    final_hr_constructed_tensor = hr_rrdb_tensor + predicted_residual_tensor
    # Clamp to valid range, e.g., [-1, 1]
    final_hr_constructed_tensor = torch.clamp(final_hr_constructed_tensor, -1.0, 1.0)
    print(f"Constructed final HR, shape: {final_hr_constructed_tensor.shape}") 

    # 8. Prepare images for plotting (convert to displayable format: HWC, [0,1] range, numpy)
    def to_display_format(tensor_chw_neg1_1):
        # tensor_chw_neg1_1 is (C, H, W) and in range [-1, 1]
        tensor_chw_0_1 = (tensor_chw_neg1_1.cpu() + 1.0) / 2.0
        tensor_hwc_0_1 = tensor_chw_0_1.permute(1, 2, 0).numpy()
        return tensor_hwc_0_1

    imgs_for_plot = {
        'lr_img_display': to_display_format(lr_img_tensor_for_input.squeeze(0)),
        'hr_orig_display': to_display_format(hr_orig_for_comparison),
        'hr_rrdb_display': to_display_format(hr_rrdb_tensor),
        'residual_true_display': to_display_format(hr_orig_for_comparison - hr_rrdb_tensor), # True residual
        'residual_predicted_display': to_display_format(predicted_residual_tensor),
        'hr_final_constructed_display': to_display_format(final_hr_constructed_tensor)
    }

    plot_result_v2(imgs_for_plot, save_path=args.output_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference with RRDB + Diffusion Refinement Model")
    parser.add_argument('--input_hr_image_for_infer', type=str, required=True,
                        help='Path to a single High-Resolution image to be downscaled and used for inference.')
    parser.add_argument('--output_image_path', type=str, default="result_diffusion_refined.png",
                        help='Path to save the plotted result image.')
    
    parser.add_argument('--img_size', type=int, default=160, help='Standard HR image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Image channels.')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor for creating LR input.')

    # RRDBNet for Base SR (used in preprocessing and here for base SR)
    parser.add_argument('--rrdb_weights_path_for_base_sr', type=str, required=True,
                        help='Path to RRDBNet weights used for generating the base HR_RRDB image.')
    parser.add_argument('--rrdb_num_feat_preproc', type=int, default=64, help='nf for RRDBNet (base SR).')
    parser.add_argument('--rrdb_num_block_preproc', type=int, default=8, help='nb for RRDBNet (base SR).')
    parser.add_argument('--rrdb_gc_preproc', type=int, default=32, help='gc for RRDBNet (base SR).')

    # RRDBNet for Context Extractor (used to condition U-Net)
    parser.add_argument('--rrdb_weights_path_context_extractor', type=str, required=True,
                        help='Path to RRDBNet weights used as context extractor for U-Net.')
    parser.add_argument('--rrdb_num_feat_context', type=int, default=64, help='nf for RRDBNet (context extractor).')
    parser.add_argument('--rrdb_num_block_context', type=int, default=8, help='nb for RRDBNet (context extractor).')
    parser.add_argument('--rrdb_gc_context', type=int, default=32, help='gc for RRDBNet (context extractor).')

    # U-Net (Diffusion Model)
    parser.add_argument('--unet_weights_path', type=str, required=True, help='Path to trained U-Net weights.')
    parser.add_argument('--unet_base_dim', type=int, default=64, help='Base dim for U-Net.')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Dim mults for U-Net.')
    parser.add_argument('--use_attention', action='store_true', help='U-Net uses attention.')
    
    # Diffusion Process
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Total timesteps U-Net was trained for.')
    parser.add_argument('--diffusion_mode', type=str, default='v_prediction', choices=['v_prediction', 'noise'],
                        help='Prediction mode U-Net was trained with.')
    parser.add_argument('--diffusion_inference_steps', type=int, default=50, help='Number of DDIM inference steps.')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference.')

    args = parser.parse_args()
    
    print("--- Inference Configuration ---") 
    for arg_name, arg_val in vars(args).items():
        print(f"  {arg_name}: {arg_val}") 
    print("-----------------------------") 
    
    main_infer(args)
