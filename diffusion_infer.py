from diffusion_trainer import DiffusionTrainer, ResidualGenerator
from utils.dataset import ImageDataset
import matplotlib.pyplot as plt
from rrdb_trainer import BasicRRDBNetTrainer
import numpy as np
from diffusion_modules import Unet
import torch

def plot_result(imgs):
    lr, up, hr, res, diff_res, con = imgs

    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs[0, 0].imshow(lr)
    axs[0, 0].set_title('LR Image')
    axs[0, 1].imshow(up)
    axs[0, 1].set_title('Upscaled Image')
    axs[0, 2].imshow(hr)
    axs[0, 2].set_title('HR Image')
    axs[1, 0].imshow(res)
    axs[1, 0].set_title('Residual')
    axs[1, 1].imshow(diff_res)
    axs[1, 1].set_title('Diff Residual')
    axs[1, 2].imshow(con)
    axs[1, 2].set_title('Constructed Image')
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('result.png')


def main():
    """
    Main function to set up and run the diffusion model inference.
    """
    # Load the model
    img_size = 160
    config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 8, 'gc': 32, 'sr_scale': 4} 
    rrdb_model_path = 'checkpoints_rrdb/rrdb_model_best.pth'
    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(model_path=rrdb_model_path, model_config=config)
    
    # Load the dataset
    dataset = ImageDataset(folder_path='data/', img_size=img_size, downscale_factor=4)

    # Create a DataLoader
    item = np.random.randint(0, len(dataset))
    lr_img, up_img, hr_img, residual = dataset.__getitem__(item)
    context = lr_img.unsqueeze(0).to('cuda')
    print(lr_img.shape)

    unet = Unet(use_attention=False)
    weight_path = 'diffusion_checkpoints/noise_20250515-151618/diffusion_model_noise_20250515-151618_best.pth'
    DiffusionTrainer.load_model_weights('cuda', unet, weight_path, True)
    unet.eval()

    # Perform inference
    generator = ResidualGenerator(img_size=img_size, predict_mode='noise')
    diff_residual = generator.generate_residuals(unet,
                                                 context,
                                                 context_extractor)
    diff_residual = diff_residual.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    lr_img = lr_img.permute(1, 2, 0).detach().cpu().numpy()
    up_img = up_img.permute(1, 2, 0).detach().cpu().numpy()
    hr_img = hr_img.permute(1, 2, 0).detach().cpu().numpy()
    residual = residual.permute(1, 2, 0).detach().cpu().numpy()

    constructed_img = up_img + diff_residual

    # Normalize the images to [0, 1] range for display
    lr_img = (lr_img + 1) / 2
    hr_img = (hr_img + 1) / 2
    up_img = (up_img + 1) / 2
    diff_residual = (diff_residual + 1) / 2
    residual = (residual + 1) / 2
    constructed_img = (constructed_img + 1) / 2

    imgs = [lr_img, up_img, hr_img, residual, diff_residual, constructed_img]

    plot_result(imgs)
    


if __name__ == '__main__':
    main()

