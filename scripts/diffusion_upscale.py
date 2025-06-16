import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os
import sys

# Add the project root to the Python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainers.diffusion_trainer import DiffusionTrainer, ResidualGenerator
from src.trainers.rrdb_trainer import BasicRRDBNetTrainer
from src.diffusion_modules.unet import Unet 

# --- Configuration (Set these paths and parameters as needed) ---
RRDB_160_PATH = 'checkpoints/rrdb/rrdb_20250521-141800/rrdb_model_best.pth'
UNET_160_PATH = 'checkpoints/diffusion/noise_20250526-070738/diffusion_model_noise_best.pth'

RRDB_320_PATH = 'checkpoints/rrdb/rrdb_320/rrdb_model_best.pth'
UNET_320_PATH = 'checkpoints/diffusion/noise_320/diffusion_model_noise_best.pth'

WEB_APP_GENERATED_FOLDER_NAME = 'generated_outputs'
STATIC_FOLDER_PARENT_FOR_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'web_app', 'static')
OUTPUT_DIR_FOR_SCRIPT_EXECUTION = os.path.join(os.path.dirname(__file__), "inference_outputs")
VIDEO_SUBDIR = ''

os.makedirs(OUTPUT_DIR_FOR_SCRIPT_EXECUTION, exist_ok=True)

# --- Helper Functions ---

def process_input_image(image_path_or_pil, target_lr_edge_size):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    elif isinstance(image_path_or_pil, Image.Image):
        img = image_path_or_pil.convert("RGB")
    else:
        raise ValueError("image_path_or_pil must be a file path or a PIL Image object.")

    transform = T.Compose([
        T.Resize(target_lr_edge_size),
        T.CenterCrop(target_lr_edge_size),
        T.ToTensor()
    ])
    tensor_img = transform(img)
    tensor_img = tensor_img * 2.0 - 1.0
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

def generate_video_for_web_app(base_image_chw_cuda, intermediate_residuals_chw_list_cuda, base_filename="video", fps=10):
    if not intermediate_residuals_chw_list_cuda:
        print("No intermediate residuals for video.")
        return None
    
    _c, h, w = base_image_chw_cuda.shape
    video_filename = f"denoising_video_{base_filename}.mp4"

    save_dir_for_web = os.path.join(STATIC_FOLDER_PARENT_FOR_SCRIPT, WEB_APP_GENERATED_FOLDER_NAME, VIDEO_SUBDIR)

    if __name__ == "__main__":
        save_dir_actual = os.path.join(OUTPUT_DIR_FOR_SCRIPT_EXECUTION, VIDEO_SUBDIR)
    else:
        save_dir_actual = save_dir_for_web

    os.makedirs(save_dir_actual, exist_ok=True)
    full_video_path = os.path.join(save_dir_actual, video_filename)

    # Change fourcc to H.264
    # 'mp4v' for MPEG-4, 'XVID' for XVID, 'MJPG' for Motion-JPEG, 'H264' for H.264
    fourcc = cv2.VideoWriter_fourcc(*'H264') # Sử dụng 'H264' hoặc 'avc1' cho H.264. 'mp4v' thường là codec mặc định cho .mp4 nếu không có H.264.
                                            # Tuy nhiên, 'H264' có thể không được hỗ trợ trên tất cả các hệ thống hoặc cài đặt OpenCV.
                                            # 'avc1' thường hoạt động tốt hơn với định dạng MP4 cho H.264.
    
    video_writer = cv2.VideoWriter(full_video_path, fourcc, float(fps), (w, h))
    
    # Kiểm tra xem VideoWriter có được mở thành công hay không
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path {full_video_path} with FOURCC {fourcc}.")
        print("Ensure you have the necessary codecs installed (e.g., FFMPEG).")
        return None

    print(f"Creating video with {len(intermediate_residuals_chw_list_cuda)} frames, saving to {full_video_path}...")

    for residual_chw_cuda in tqdm(intermediate_residuals_chw_list_cuda, desc="Generating video frames"):
        current_img_chw_cuda = torch.clamp(base_image_chw_cuda + residual_chw_cuda, -1.0, 1.0)
        current_img_chw_0_1 = (current_img_chw_cuda + 1.0) / 2.0
        frame_hwc_0_1 = current_img_chw_0_1.permute(1, 2, 0)
        frame_np_uint8 = (frame_hwc_0_1.cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Video saved successfully to {full_video_path}")

    relative_path_for_web = os.path.join(VIDEO_SUBDIR, video_filename)
    return relative_path_for_web

# --- Main Upscale Function for app.py ---
def upscale(input_lr_image_path: str,
            target_lr_edge_size: int,
            num_inference_steps: int = 50,
            device_str: str = "cuda"
            ):
    if target_lr_edge_size <= 60:
        rrdb_weights_path = RRDB_160_PATH
        unet_weights_path = UNET_160_PATH
    else:
        rrdb_weights_path = RRDB_320_PATH
        unet_weights_path = UNET_320_PATH
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Upscaling on device: {device}")
    plot_relative_path = None 

    print(f"Loading RRDBNet (context extractor) from: {rrdb_weights_path}")
    try:
        context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(
            model_path=rrdb_weights_path,
            device=device 
        )
        context_extractor.eval()
    except Exception as e:
        print(f"Error loading RRDBNet: {e}")
        raise

    print(f"Loading UNet from: {unet_weights_path}")
    try:
        unet_model = DiffusionTrainer.load_diffusion_unet(
            model_path=unet_weights_path,
            device=device 
        ).eval()
        
        unet_checkpoint_path = unet_weights_path 
        try:
            unet_checkpoint = torch.load(unet_checkpoint_path, map_location='cpu', weights_only=False)
            if unet_checkpoint is None:
                print(f"Warning: torch.load returned None for UNet checkpoint {unet_checkpoint_path} when inspecting for mode/timesteps.")
                diffusion_mode = 'noise' 
                unet_model_config = {}   
            else:
                diffusion_mode = unet_checkpoint.get('mode', 'noise')
                unet_model_config = unet_checkpoint.get('model_config', {})
        except Exception as e_inspect:
            print(f"Warning: Could not inspect UNet checkpoint {unet_checkpoint_path} for mode/timesteps due to error: {e_inspect}. Using defaults.")
            diffusion_mode = 'noise' 
            unet_model_config = {}   

        loaded_timesteps = unet_model_config.get('timesteps', 1000)
        print(f"  UNet diffusion mode: {diffusion_mode}")
        print(f"  UNet timesteps for generator: {loaded_timesteps}")

    except Exception as e:
        print(f"Error loading UNet or its config: {e}")
        raise

    hr_image_edge_size = target_lr_edge_size * context_extractor.sr_scale 
    generator = ResidualGenerator(
        img_size=hr_image_edge_size, 
        predict_mode=diffusion_mode, 
        device=device,
        num_train_timesteps=loaded_timesteps 
    )

    lr_image_bchw_cuda = process_input_image(input_lr_image_path, target_lr_edge_size).to(device)
    print(f"Processed LR image tensor shape: {lr_image_bchw_cuda.shape}")

    with torch.no_grad():
        # --- Debugging print statements ---
        print(f"DEBUG: Type of context_extractor: {type(context_extractor)}")
        print(f"DEBUG: Calling context_extractor with get_fea=True")
        temp_return_value = context_extractor(lr_image_bchw_cuda, get_fea=True)
        
        if isinstance(temp_return_value, tuple):
            print(f"DEBUG: Value returned by context_extractor is a tuple of length: {len(temp_return_value)}")
            if len(temp_return_value) > 0:
                print(f"DEBUG: Type of first element: {type(temp_return_value[0])}")
            if len(temp_return_value) > 1:
                print(f"DEBUG: Type of second element: {type(temp_return_value[1])}")
        else:
            print(f"DEBUG: Value returned by context_extractor is NOT a tuple. Type: {type(temp_return_value)}")
        # --- End of debugging print statements ---
        
        try:
            rrdb_output_hr_bchw_cuda, features_cuda = temp_return_value # Original unpacking line
        except ValueError as ve:
            print(f"DEBUG: ValueError during unpacking: {ve}")
            print(f"DEBUG: temp_return_value was: {temp_return_value}")
            raise ve # Re-raise the error after printing debug info

    print(f"RRDBNet output (HR base) tensor shape: {rrdb_output_hr_bchw_cuda.shape}")
    if features_cuda is not None and isinstance(features_cuda, list):
        print(f"Number of feature maps from RRDBNet: {len(features_cuda)}")
    elif features_cuda is None:
        print("Warning: features_cuda from RRDBNet is None.")
    else:
        print(f"Warning: features_cuda from RRDBNet is not a list. Type: {type(features_cuda)}")


    with torch.no_grad():
        intermediate_residuals_list_cuda = generator.generate_residuals(
            model=unet_model,
            features=features_cuda, 
            num_images=1, 
            num_inference_steps=num_inference_steps,
            return_intermediate_steps=True 
        )
    final_predicted_residual_hr_bchw_cuda = intermediate_residuals_list_cuda[-1]
    print(f"Predicted residual tensor shape: {final_predicted_residual_hr_bchw_cuda.shape}")

    final_hr_image_bchw_cuda = torch.clamp(rrdb_output_hr_bchw_cuda + final_predicted_residual_hr_bchw_cuda, -1.0, 1.0)

    final_hr_image_chw_0_1 = (final_hr_image_bchw_cuda.squeeze(0).cpu() + 1.0) / 2.0
    final_hr_image_hwc_0_1_np = final_hr_image_chw_0_1.permute(1, 2, 0).numpy()
    final_hr_image_hwc_0_1_np = np.clip(final_hr_image_hwc_0_1_np, 0.0, 1.0)

    base_input_filename = os.path.splitext(os.path.basename(input_lr_image_path))[0]

    rrdb_output_hr_chw_cuda = rrdb_output_hr_bchw_cuda.squeeze(0)
    intermediate_residuals_chw_list_cuda_squeezed = [res.squeeze(0) for res in intermediate_residuals_list_cuda]

    video_relative_path = generate_video_for_web_app(
        base_image_chw_cuda=rrdb_output_hr_chw_cuda,
        intermediate_residuals_chw_list_cuda=intermediate_residuals_chw_list_cuda_squeezed,
        base_filename=f"{base_input_filename}_lr{target_lr_edge_size}",
        fps=10
    )
    if video_relative_path:
        video_relative_path = video_relative_path.replace(os.sep, '/')
        
    return final_hr_image_hwc_0_1_np, video_relative_path, plot_relative_path