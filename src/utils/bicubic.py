import os
import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional, Any

# Attempt to import torch for type hinting and tensor conversion
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ALLOWED_SCALE_FACTORS: list[int] = [2, 4, 6, 8, 10]
EPSILON = 1e-5 # For float comparisons

def upscale_image(
    image_source: Union[str, np.ndarray, 'torch.Tensor'],
    scale_factor: int,
    save_image: bool = False,
    output_directory: str = "bicubic_output",
    output_filename_prefix: str = "upscaled"
) -> Optional[Union[np.ndarray, 'torch.Tensor']]:
    """
    Upscales an image using Bicubic interpolation with flexible output typing.
    This version is streamlined to primarily process RGB images.
    - Image files (e.g. grayscale, RGBA) will be converted to RGB.
    - NumPy array or Torch Tensor inputs are expected to be RGB (3 channels).

    - Saved images are always uint8 [0,255] RGB.
    - Returned NumPy array or Torch Tensor matches input type and data range:
        - Input float [0,1] -> Output float [0,1]
        - Input uint8 [0,255] -> Output uint8 [0,255]
        - Input float [0,255] -> Output float [0,255] (same float type)
        - Input np.ndarray -> Output np.ndarray
        - Input torch.Tensor -> Output torch.Tensor (HWC format, on original device)

    Args:
        image_source (Union[str, np.ndarray, torch.Tensor]):
            The source image. For arrays/tensors, expects HWC RGB format.
        scale_factor (int):
            The factor by which to upscale. Must be one of ALLOWED_SCALE_FACTORS.
        save_image (bool, optional):
            If True, save the upscaled image (as uint8 [0,255] RGB). Defaults to False.
        output_directory (str, optional):
            Directory for saved images. Defaults to "bicubic_output".
        output_filename_prefix (str, optional):
            Prefix for saved filenames. Defaults to "upscaled".

    Returns:
        Optional[Union[np.ndarray, torch.Tensor]]:
            The upscaled RGB image, matching input object type (NumPy/Torch) and
            original data type/range characteristics, or None on error.
            Torch Tensors are returned in HWC RGB format.

    Raises:
        ValueError: For invalid scale_factor or unsupported input image properties (e.g. non-RGB array/tensor).
        FileNotFoundError: If image_source path not found.
        ImportError: If torch.Tensor input but torch is not installed.
    """
    if scale_factor not in ALLOWED_SCALE_FACTORS:
        raise ValueError(
            f"scale_factor must be one of {ALLOWED_SCALE_FACTORS}. Got {scale_factor}."
        )

    input_is_tensor: bool = False
    original_dtype: Optional[np.dtype] = None
    input_was_float_0_1: bool = False
    original_torch_device: Optional[Any] = None
    img_for_processing: np.ndarray
    original_stem: str = "input_image"
    original_ext: str = ".png"

    try:
        # --- 1. Input Loading and Normalization (Focus on RGB) ---
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            original_stem = os.path.splitext(os.path.basename(image_source))[0]
            original_ext = os.path.splitext(os.path.basename(image_source))[1] or ".png"
            
            img_pil = Image.open(image_source)
            if img_pil.mode != 'RGB':
                # Convert to RGB (handles L, RGBA, P, CMYK etc.)
                img_pil = img_pil.convert('RGB') 
            
            img_np_loaded = np.array(img_pil)
            img_for_processing = img_np_loaded # Should be HWC, C=3 (RGB)
            original_dtype = img_for_processing.dtype # Typically uint8 from Pillow
            # For files converted to RGB, assume 0-255 range, so input_was_float_0_1 is False

        elif TORCH_AVAILABLE and isinstance(image_source, torch.Tensor):
            input_is_tensor = True
            tensor_in = image_source.detach()
            original_torch_device = tensor_in.device
            
            np_from_tensor = tensor_in.cpu().numpy()

            # Validate tensor shape for RGB (expect HWC or NCHW/NHWC with N=1, C=3)
            if np_from_tensor.ndim == 4: # Batch dimension
                if np_from_tensor.shape[0] == 1:
                    np_from_tensor = np_from_tensor.squeeze(0)
                else:
                    raise ValueError("Batch tensor input not supported. Pass a single image tensor.")
            
            # Now expect 3D tensor (CHW or HWC)
            if np_from_tensor.ndim == 3:
                # CHW to HWC conversion if needed (C=3)
                if np_from_tensor.shape[0] == 3 and np_from_tensor.shape[1] > 3 and np_from_tensor.shape[2] > 3: # Heuristic for CHW
                    np_from_tensor = np_from_tensor.transpose(1, 2, 0)
                
                if np_from_tensor.shape[-1] != 3:
                    raise ValueError(f"Input Tensor is 3D but not RGB (HWC with C=3). Shape: {np_from_tensor.shape}")
            else:
                raise ValueError(f"Input Tensor is not a valid RGB image format (expected 3D HWC or 4D NHWC with N=1). Shape: {np_from_tensor.shape}")

            img_for_processing = np_from_tensor # Now HWC, C=3
            original_dtype = img_for_processing.dtype

            if np.issubdtype(original_dtype, np.floating):
                min_val, max_val = img_for_processing.min(), img_for_processing.max()
                if min_val >= (0.0 - EPSILON) and max_val <= (1.0 + EPSILON):
                    input_was_float_0_1 = True
                img_for_processing = img_for_processing.astype(np.float32) # Use float32 for processing
            elif np.issubdtype(original_dtype, np.uint8):
                pass # Already uint8 [0,255]
            else: # Other integer types
                print(f"Warning: Input Tensor dtype {original_dtype} not uint8 or float. Converting to float32 [0,255] range for processing.")
                max_val_dtype = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 255.0
                img_for_processing = (img_for_processing.astype(np.float32) / max_val_dtype) * 255.0
                original_dtype = np.float32 # Effective original type for range matching

        elif isinstance(image_source, np.ndarray):
            img_for_processing = image_source.copy()
            original_dtype = img_for_processing.dtype

            # Validate NumPy array shape for RGB (expect HWC, C=3)
            if img_for_processing.ndim != 3 or img_for_processing.shape[-1] != 3:
                raise ValueError(f"Input NumPy array is not RGB (HWC with C=3). Shape: {img_for_processing.shape}")

            if np.issubdtype(original_dtype, np.floating):
                min_val, max_val = img_for_processing.min(), img_for_processing.max()
                if min_val >= (0.0 - EPSILON) and max_val <= (1.0 + EPSILON):
                    input_was_float_0_1 = True
                img_for_processing = img_for_processing.astype(np.float32) # Use float32
            elif np.issubdtype(original_dtype, np.uint8):
                pass # Already uint8 [0,255]
            else: # Other integer types
                print(f"Warning: Input NumPy dtype {original_dtype} not uint8 or float. Converting to float32 [0,255] range for processing.")
                max_val_dtype = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 255.0
                img_for_processing = (img_for_processing.astype(np.float32) / max_val_dtype) * 255.0
                original_dtype = np.float32 # Effective original type for range matching
        
        elif not TORCH_AVAILABLE and type(image_source).__module__ == 'torch':
            raise ImportError("Input is a torch.Tensor, but 'torch' is not installed.")
        else:
            raise TypeError(f"Unsupported image_source type: {type(image_source)}")

        # Ensure img_for_processing is HWC, C=3 and suitable dtype for cv2.resize
        if img_for_processing.ndim != 3 or img_for_processing.shape[-1] != 3:
             raise ValueError(f"Image for processing is not RGB (HWC with C=3). Shape: {img_for_processing.shape}")

        if not (img_for_processing.dtype == np.uint8 or \
                np.issubdtype(img_for_processing.dtype, np.floating)):
            print(f"Info: Converting img_for_processing (dtype: {img_for_processing.dtype}) to float32 for resizing.")
            img_for_processing = img_for_processing.astype(np.float32) # Default to float32 if not uint8/float
        
        if img_for_processing.size == 0:
            raise ValueError("Image data is empty after initial processing.")

    except Exception as e:
        print(f"Error during input processing: {e}")
        return None

    # --- 2. Perform Upscaling ---
    try:
        # img_for_processing is guaranteed to be HWC (RGB) at this point
        h, w, c = img_for_processing.shape 
        if c != 3: # Should be caught earlier, but as a safeguard
            raise ValueError(f"Image for processing is not 3-channel RGB. Channels: {c}")
            
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        upscaled_intermediate_np = cv2.resize(img_for_processing, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Output of resize will be HWC, C=3 with same dtype as img_for_processing

    except cv2.error as e:
        print(f"OpenCV error during resizing: {e}")
        return None
    except Exception as e:
        print(f"Error during upscaling step: {e}")
        return None

    # --- 3. Prepare Return Value (matching input type/range) ---
    processed_result_np = upscaled_intermediate_np.copy() # HWC, C=3

    if input_was_float_0_1:
        processed_result_np = np.clip(processed_result_np, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.floating):
            processed_result_np = processed_result_np.astype(original_dtype)
        else: 
            processed_result_np = processed_result_np.astype(np.float32)
    else: # Input was in 0-255 range (uint8 or float)
        processed_result_np = np.clip(processed_result_np, 0, 255)
        if np.issubdtype(original_dtype, np.uint8):
            processed_result_np = processed_result_np.astype(np.uint8)
        elif np.issubdtype(original_dtype, np.floating): 
            processed_result_np = processed_result_np.astype(original_dtype)
        else: 
            processed_result_np = processed_result_np.astype(np.uint8)


    result_image: Union[np.ndarray, 'torch.Tensor']
    if input_is_tensor:
        result_image = torch.from_numpy(processed_result_np.copy()).to(original_torch_device)
    else:
        result_image = processed_result_np

    # --- 4. Save Output Image (always as uint8 [0,255] RGB) ---
    if save_image:
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)

            output_filename = f"{output_filename_prefix}_{original_stem}_x{scale_factor}{original_ext}"
            save_path = os.path.join(output_directory, output_filename)

            img_to_save_np: np.ndarray
            temp_save_img = upscaled_intermediate_np.copy() # HWC, C=3

            if np.issubdtype(temp_save_img.dtype, np.floating):
                if input_was_float_0_1: # Intermediate was float [0,1]
                    img_to_save_np = (np.clip(temp_save_img, 0.0, 1.0) * 255).astype(np.uint8)
                else: # Assumed intermediate was float [0,255]
                    img_to_save_np = np.clip(temp_save_img, 0, 255).astype(np.uint8)
            elif np.issubdtype(temp_save_img.dtype, np.uint8):
                img_to_save_np = np.clip(temp_save_img, 0, 255) 
            else: 
                print(f"Warning: Unexpected dtype {temp_save_img.dtype} for saving. Converting to uint8 [0,255].")
                img_to_save_np = np.clip(temp_save_img.astype(np.float32), 0, 255).astype(np.uint8)
            
            if img_to_save_np.ndim != 3 or img_to_save_np.shape[2] != 3:
                 raise ValueError(f"Cannot save non-RGB image. Shape: {img_to_save_np.shape}")

            img_to_save_pil = Image.fromarray(img_to_save_np, mode='RGB')
            img_to_save_pil.save(save_path)
            print(f"Upscaled image saved to: {save_path}")

        except Exception as e:
            print(f"Error saving image: {e}")

    return result_image
