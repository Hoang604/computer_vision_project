import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import torchvision.transforms.functional as TF
from bicubic import upscale_image


class ImageDataset(Dataset):
    def __init__(self, folder_path: str, img_size: int, downscale_factor: int, upscale_function=upscale_image):
        """
        Initializes the dataset to load images and generate multiple versions.
        All output image tensors (low_res, upscaled, original_resized) will be in the range [-1, 1].
        The residual_image will be in the range [-2, 2].

        Args:
            folder_path (str): Path to the folder containing images.
            img_size (int): Target size for the 'original' processed image (height and width).
            downscale_factor (int): Factor by which to downscale the image for the low-resolution version.
                                   This is also the factor for upscaling the low-res image.
            upscale_function (callable): The actual function to use for upscaling images.
                                         It should expect a [0,1] range tensor and return a [0,1] range tensor.
        """
        self.folder_path = folder_path
        # Find image files - consider adding more extensions if needed
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.original_len = len(self.image_files) # Store original number of files
        self.img_size = img_size
        self.downscale_factor = downscale_factor
        self.upscale_image = upscale_function # Use the provided upscale function

        if not isinstance(self.downscale_factor, int) or self.downscale_factor < 1:
            raise ValueError("downscale_factor must be an integer and >= 1.") # write message on console
        if not isinstance(self.img_size, int) or self.img_size <= 0:
            raise ValueError("img_size must be a positive integer.") # write message on console
        if self.img_size % self.downscale_factor != 0:
            print(f"Warning: img_size ({self.img_size}) is not perfectly divisible by downscale_factor ({self.downscale_factor}). "
                  "This might lead to slight dimension mismatches if not handled carefully by the upscale_image function "
                  "or require the safeguard resize.") # write message on console

        print(f"Found {self.original_len} images in {folder_path}. Target original_img_size: {img_size}x{img_size}, downscale_factor: {downscale_factor}. Image range: [-1, 1].") # write message on console

    def __len__(self):
        """ Returns the total size of the dataset (original + flipped). """
        return self.original_len * 2 # Report double the length for data augmentation

    def __getitem__(self, idx):
        """
        Retrieves a tuple of image tensors:
        (low_res_image, upscaled_image, original_image_resized, residual_image)
        All image tensors (low_res, upscaled, original_resized) are in the [-1, 1] value range.
        The residual_image is the direct difference and will be in the [-2, 2] range.

        Args:
            idx (int): Index of the item to retrieve (0 to 2*original_len - 1).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - low_res_image (C, H_low, W_low), range [-1, 1]
                - upscaled_image (C, H_orig, W_orig), range [-1, 1]
                - original_image_resized (C, H_orig, W_orig), range [-1, 1]
                - residual_image (C, H_orig, W_orig), range [-2, 2]
        """
        # Determine if this index corresponds to a flipped image
        should_flip = idx >= self.original_len

        # Calculate the index of the original image file
        original_idx = idx % self.original_len

        # Construct the image path
        img_path = os.path.join(self.folder_path, self.image_files[original_idx])

        try:
            # Load the PIL image (H, W, C), values in [0, 255]
            image_pil = Image.open(img_path).convert("RGB")

            # Apply horizontal flip if needed, *before* other transforms
            if should_flip:
                image_pil = TF.hflip(image_pil)

            # 1. Original Image
            # Convert PIL Image to Tensor (C, H, W), values in [0,1]
            original_image_as_tensor_0_1 = TF.to_tensor(image_pil)
            # Transform to [-1, 1] range
            original_image_as_tensor = original_image_as_tensor_0_1 * 2.0 - 1.0


            # Resize to the target 'img_size' for the "original" reference image
            # Output is Tensor (C, self.img_size, self.img_size), range [-1,1]
            original_image_resized = TF.resize(
                original_image_as_tensor,
                [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            )

            # 2. Low-Resolution Image (range [-1,1])
            # Calculate dimensions for the low-resolution image
            low_res_h = self.img_size // self.downscale_factor
            low_res_w = self.img_size // self.downscale_factor

            if low_res_h == 0 or low_res_w == 0:
                raise ValueError(
                    f"Calculated low_res dimension is zero ({low_res_h}x{low_res_w}) for img_size={self.img_size} "
                    f"and downscale_factor={self.downscale_factor}. Adjust parameters."
                ) # write message on console

            # Create low-resolution image by downscaling the 'original_image_resized'
            # Output is Tensor (C, low_res_h, low_res_w), range [-1,1]
            low_res_image = TF.resize(
                original_image_resized.clone(),
                [low_res_h, low_res_w],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True
            )

            # 3. Upscaled Image (target range [-1,1])
            
            # Convert low_res_image from [-1,1] to [0,1] for the upscale_image function
            low_res_image_0_1 = (low_res_image.clone() + 1.0) / 2.0

            # The 'upscale_image' function (provided by user) takes low_res_image_0_1 (Tensor C,H,W [0,1])
            # and is expected to return a torch.Tensor (H_up, W_up, C) in [0,1] range.
            returned_upscaled_image_hwc_0_1 = self.upscale_image(
                image_source=low_res_image_0_1, # This is in [0,1]
                scale_factor=self.downscale_factor,
                save_image=False # Assuming we don't need to save it here from dataset
            )

            if returned_upscaled_image_hwc_0_1 is None:
                raise RuntimeError(f"The 'upscale_image' function returned None for image: {img_path}") # write message on console

            if not isinstance(returned_upscaled_image_hwc_0_1, torch.Tensor):
                # write message on console
                print(f"Warning: upscale_image was expected to return a Tensor but returned {type(returned_upscaled_image_hwc_0_1)}. Attempting conversion.")
                if isinstance(returned_upscaled_image_hwc_0_1, np.ndarray):
                    # Assuming np.ndarray is in [0,1] range if upscale_image was supposed to return that
                    returned_upscaled_image_hwc_0_1 = torch.from_numpy(returned_upscaled_image_hwc_0_1).float()
                else:
                    raise TypeError(f"upscale_image returned an unexpected type: {type(returned_upscaled_image_hwc_0_1)}")


            # Convert returned HWC Tensor [0,1] back to [-1,1]
            returned_upscaled_image_hwc_neg1_1 = returned_upscaled_image_hwc_0_1 * 2.0 - 1.0
            
            # Permute to CHW Tensor, range [-1,1]
            upscaled_image_tensor = returned_upscaled_image_hwc_neg1_1.permute(2, 0, 1).float()


            # Ensure the upscaled image tensor matches the dimensions of original_image_resized.
            if upscaled_image_tensor.shape[1:] != original_image_resized.shape[1:]:
                # write message on console
                # print(f"Warning: Dimensions of upscaled image ({upscaled_image_tensor.shape[1:]}) "
                #       f"do not match original_image_resized ({original_image_resized.shape[1:]}) for {img_path}. Resizing upscaled image.")
                upscaled_image_tensor = TF.resize(
                    upscaled_image_tensor,
                    [self.img_size, self.img_size],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    antialias=True
                )

            # 4. Residual Image (range [-2,2])
            # Both original_image_resized and upscaled_image_tensor are (C, self.img_size, self.img_size), range [-1,1]
            # Their difference will be in range [-2, 2]
            residual_image = original_image_resized - upscaled_image_tensor

            return low_res_image, upscaled_image_tensor, original_image_resized, residual_image

        except Exception as e:
            # write message on console
            print(f"Error loading or processing image at index {idx} (original file: {self.image_files[original_idx]}): {e}")
            # Fallback: return dummy tensors of expected shapes and ranges.
            dummy_c = 3
            dummy_low_h = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)
            dummy_low_w = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)

            # Tensors are expected in [-1,1] (or [-2,2] for residual). Zeros are fine for [-1,1] and [-2,2].
            _dummy_low_res = torch.zeros((dummy_c, dummy_low_h, dummy_low_w))
            _dummy_upscaled = torch.zeros((dummy_c, self.img_size, self.img_size))
            _dummy_original = torch.zeros((dummy_c, self.img_size, self.img_size))
            _dummy_residual = torch.zeros((dummy_c, self.img_size, self.img_size))

            # raise e # Option: re-raise the exception to halt on error
            return _dummy_low_res, _dummy_upscaled, _dummy_original, _dummy_residual
        
class ImageDatasetRRDB(Dataset):
    def __init__(self,
                 preprocessed_folder_path: str,
                 img_size: int, # Can be used for verification or if needed by other parts
                 downscale_factor: int, # Same as above
                 apply_hflip: bool = False): # Added option for horizontal flipping
        """
        Initializes the dataset to load preprocessed tensors.
        All output image tensors (low_res, upscaled_rrdb, original_hr)
        will be in the range [-1, 1].
        The residual_image will be in the range [-2, 2].

        Args:
            preprocessed_folder_path (str): Path to the folder containing preprocessed tensors
                                          (subfolders: 'hr_original', 'lr', 'hr_rrdb_upscaled').
            img_size (int): Target size for the 'original' HR image (height and width).
                            Used for consistency checks.
            downscale_factor (int): Factor by which HR was downscaled to get LR.
                                    Used for consistency checks.
            apply_hflip (bool): Whether to apply horizontal flipping as data augmentation.
                                Defaults to False.
        """
        self.preprocessed_folder_path = preprocessed_folder_path
        self.path_hr_original = os.path.join(preprocessed_folder_path, 'hr_original')
        self.path_lr = os.path.join(preprocessed_folder_path, 'lr')
        self.path_hr_rrdb_upscaled = os.path.join(preprocessed_folder_path, 'hr_rrdb_upscaled')

        # Ensure all subdirectories exist
        if not all(os.path.isdir(p) for p in [self.path_hr_original, self.path_lr, self.path_hr_rrdb_upscaled]):
            raise FileNotFoundError(
                f"One or more subdirectories ('hr_original', 'lr', 'hr_rrdb_upscaled') "
                f"not found in {preprocessed_folder_path}. "
                "Please run the preprocessing script first."
            )

        # Assume all subdirectories have the same number of files and matching basenames
        # Get the list of tensor file basenames from one of the directories (e.g., 'lr')
        self.tensor_files_basenames = sorted(
            [f for f in os.listdir(self.path_lr) if f.lower().endswith('.pt')]
        )
        
        if not self.tensor_files_basenames:
            raise ValueError(f"No .pt files found in {self.path_lr}. "
                             "Ensure preprocessing was successful and generated .pt files.") # write message on console

        self.num_original_samples = len(self.tensor_files_basenames)
        self.img_size = img_size
        self.downscale_factor = downscale_factor
        self.apply_hflip = apply_hflip

        print(f"Found {self.num_original_samples} preprocessed tensor sets in {preprocessed_folder_path}.") # write message on console
        if self.apply_hflip:
            print("Horizontal flipping augmentation is ENABLED.") # write message on console
        else:
            print("Horizontal flipping augmentation is DISABLED.") # write message on console


    def __len__(self):
        """ Returns the total size of the dataset (original + flipped if augmentation is enabled). """
        return self.num_original_samples * 2 if self.apply_hflip else self.num_original_samples

    def __getitem__(self, idx):
        """
        Retrieves a tuple of image tensors:
        (low_res_image, upscaled_image_rrdb, original_hr_image, residual_image)
        All image tensors (low_res, upscaled_rrdb, original_hr) are in the [-1, 1] value range.
        The residual_image is the direct difference and will be in the [-2, 2] range.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - low_res_image (C, H_low, W_low), range [-1, 1]
                - upscaled_image_rrdb (C, H_orig, W_orig), range [-1, 1]
                - original_hr_image (C, H_orig, W_orig), range [-1, 1]
                - residual_image (C, H_orig, W_orig), range [-2, 2]
        """
        should_flip_this_sample = False
        if self.apply_hflip:
            should_flip_this_sample = idx >= self.num_original_samples
            actual_idx = idx % self.num_original_samples
        else:
            actual_idx = idx

        base_filename = self.tensor_files_basenames[actual_idx] # e.g., "image_001.pt"

        try:
            # Load tensors (they were saved as CPU tensors)
            # .clone() is used to ensure that if any in-place operations were to happen
            # (like flipping), they don't affect the cached tensor if DataLoader uses workers.
            # For simple loading and returning, .clone() might not be strictly necessary
            # but is safer.
            low_res_image = torch.load(os.path.join(self.path_lr, base_filename)).clone()
            original_hr_image = torch.load(os.path.join(self.path_hr_original, base_filename)).clone()
            upscaled_image_rrdb = torch.load(os.path.join(self.path_hr_rrdb_upscaled, base_filename)).clone()

            # --- Verification (optional but good for debugging) ---
            # Verify HR image size
            if original_hr_image.shape[1] != self.img_size or original_hr_image.shape[2] != self.img_size:
                print(f"Warning: Loaded original_hr_image for {base_filename} has shape {original_hr_image.shape} "
                      f"but expected H/W of {self.img_size}. Check preprocessing consistency.") # write message on console
            # Verify LR image size
            expected_lr_h = self.img_size // self.downscale_factor
            expected_lr_w = self.img_size // self.downscale_factor
            if low_res_image.shape[1] != expected_lr_h or low_res_image.shape[2] != expected_lr_w:
                 print(f"Warning: Loaded low_res_image for {base_filename} has shape {low_res_image.shape} "
                       f"but expected H/W of ({expected_lr_h}, {expected_lr_w}). Check preprocessing consistency.") # write message on console
            # Verify upscaled RRDB image size
            if upscaled_image_rrdb.shape[1] != self.img_size or upscaled_image_rrdb.shape[2] != self.img_size:
                print(f"Warning: Loaded upscaled_image_rrdb for {base_filename} has shape {upscaled_image_rrdb.shape} "
                      f"but expected H/W of {self.img_size}. Check preprocessing consistency.") # write message on console
            # --- End Verification ---

            # Apply horizontal flip if needed
            if should_flip_this_sample:
                low_res_image = TF.hflip(low_res_image)
                original_hr_image = TF.hflip(original_hr_image)
                upscaled_image_rrdb = TF.hflip(upscaled_image_rrdb)

            # Calculate the new residual image
            # Both original_hr_image and upscaled_image_rrdb are (C, self.img_size, self.img_size), range [-1,1]
            # Their difference will be in range [-2, 2]
            residual_image = original_hr_image - upscaled_image_rrdb

            return low_res_image, upscaled_image_rrdb, original_hr_image, residual_image

        except FileNotFoundError as fnf_err:
            print(f"Error: Preprocessed file not found for {base_filename} at index {idx}. {fnf_err}") # write message on console
            # Fallback: return dummy tensors
        except Exception as e:
            print(f"Error loading or processing tensor for {base_filename} at index {idx}: {e}") # write message on console
            # Fallback: return dummy tensors

        # Fallback dummy tensor creation (if any error occurred)
        dummy_c = 3 # Assuming 3 channels
        dummy_lr_h = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)
        dummy_lr_w = max(1, self.img_size // self.downscale_factor if self.downscale_factor > 0 else self.img_size)

        _dummy_low_res = torch.zeros((dummy_c, dummy_lr_h, dummy_lr_w))
        _dummy_upscaled_rrdb = torch.zeros((dummy_c, self.img_size, self.img_size))
        _dummy_original_hr = torch.zeros((dummy_c, self.img_size, self.img_size))
        _dummy_residual = torch.zeros((dummy_c, self.img_size, self.img_size)) # Residual can also be zeros

        return _dummy_low_res, _dummy_upscaled_rrdb, _dummy_original_hr, _dummy_residual