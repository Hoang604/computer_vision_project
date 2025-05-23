from diffusion_trainer import DiffusionTrainer, ResidualGenerator
from utils.dataset import ImageDatasetRRDB
import matplotlib.pyplot as plt
from rrdb_trainer import BasicRRDBNetTrainer
import numpy as np
from diffusion_modules import Unet

def plot_result(imgs):
    lr, up, diff_res, con = imgs

    fig, axs = plt.subplots(2, 2, figsize=(10, 10)) 

    print(f"lr shape: {lr.shape}")
    axs[0, 0].imshow(lr, interpolation='nearest')
    axs[0, 0].set_title(f'LR Image ({lr.shape[0]}x{lr.shape[1]})')

    print(f"up shape: {up.shape}")
    axs[0, 1].imshow(up)
    axs[0, 1].set_title(f'Upscaled Image ({up.shape[0]}x{up.shape[1]})')

    print(f"diff_res shape: {diff_res.shape}")
    axs[1, 0].imshow(diff_res) 
    axs[1, 0].set_title(f'Diff Residual ({diff_res.shape[0]}x{diff_res.shape[1]})')

    print(f"con shape: {con.shape}")
    axs[1, 1].imshow(con)
    axs[1, 1].set_title(f'Constructed Image ({con.shape[0]}x{con.shape[1]})')

    for ax_row in axs: 
        for ax in ax_row:
            ax.set_aspect('equal') 
            ax.axis('off')
            
    plt.tight_layout() 
    plt.savefig('result.png')


def main():
    """
    Main function to set up and run the diffusion model inference.
    """
    # Load the model
    img_size = 160
    config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 17, 'gc': 32, 'sr_scale': 4} 
    rrdb_path = '/home/hoangdv/cv_project/checkpoints_rrdb/rrdb_20250521-141800/rrdb_model_best.pth'
    context_extractor = BasicRRDBNetTrainer.load_model_for_evaluation(model_path=rrdb_path, model_config=config)
    
    # Load the dataset
    img_folder = '/media/tuannl1/heavy_weight/data/cv_data/celeba160x160/test/rrdb'
    dataset = ImageDatasetRRDB(preprocessed_folder_path=img_folder, img_size=img_size, downscale_factor=4)

    unet_path = '/home/hoangdv/cv_project/checkpoints_diffusion_v3/noise_20250522-230250/diffusion_model_noise_20250522-230250_best.pth'
    unet = Unet(use_attention=True, rrdb_num_blocks=17)
    DiffusionTrainer.load_model_weights(unet, unet_path, True)
    unet.eval()
    # Create a DataLoader
    while True:
        item = np.random.randint(0, len(dataset))
        lr_img, _, _, _ = dataset.__getitem__(item)
        context_hr = lr_img.unsqueeze(0).to('cuda')
        up_lr_img, features = context_extractor(context_hr, get_fea=True)

        # Perform inference
        generator = ResidualGenerator(img_size=img_size, predict_mode='noise')
        diff_residual = generator.generate_residuals(unet,
                                                     features=features)
        
        diff_residual = diff_residual.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        up_lr_img = up_lr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        lr_img = lr_img.permute(1, 2, 0).detach().cpu().numpy()

        constructed_img = up_lr_img + diff_residual

        # Normalize the images to [0, 1] range for display
        lr_img = (lr_img + 1) / 2
        diff_residual = (diff_residual + 1) / 2
        constructed_img = (constructed_img + 1) / 2
        up_lr_img = (up_lr_img + 1) / 2

        imgs = [lr_img, up_lr_img, diff_residual, constructed_img]

        plot_result(imgs)
        input()
        


if __name__ == '__main__':
    main()

