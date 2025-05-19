from diffusion_trainer import DiffusionTrainer, ResidualGenerator
import matplotlib.pyplot as plt
from rrdb_trainer import BasicRRDBNetTrainer
from utils.dataset import ImageDatasetRRDB
from diffusion_modules import Unet
import torch
import torchvision.transforms.functional as TF
import cv2 as cv
from time import sleep

def plot_result(imgs):
    lr, up_rrdb, diff_res, dif_recon = imgs

    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    print(lr.shape)
    axs[0, 0].imshow(lr)
    axs[0, 0].set_title('LR Image')
    print(up_rrdb.shape)
    axs[0, 1].imshow(up_rrdb)
    axs[0, 1].set_title('Upscaled Image')
    print(diff_res.shape)
    axs[1, 0].imshow(diff_res)
    axs[1, 0].set_title('Diff Residual')
    print(dif_recon.shape)
    axs[1, 1].imshow(dif_recon)
    axs[1, 1].set_title('Reconstructed Image')
    # for ax in axs.flat:
    #     ax.axis('off')
    plt.tight_layout()
    plt.savefig('result2.png')
    plt.show()


def main(args):
    """
    Main function to set up and run the diffusion model inference.
    """
    # Load the model
    with torch.no_grad():
        img_size = args.lr_img_size * 4
        config = {'in_nc': 3, 'out_nc': 3, 'num_feat': 64, 'num_block': 17, 'gc': 32, 'sr_scale': 4} 
        context_extractor = BasicRRDBNetTrainer.\
            load_model_for_evaluation(model_path=args.rrdb_weights_path, 
                                      model_config=config)
        
        # load image
        lr_img = cv.imread(args.lr_img_path, cv.IMREAD_COLOR_RGB)
        lr_img = lr_img.astype('float32') / 255.0

        context = TF.to_tensor(lr_img * 2 - 1).unsqueeze(0).to('cuda') # add batch dimension
        up_rrdb_img, feas = context_extractor(context, get_fea=True)
        # remove the batch dimension
        up_rrdb_img = up_rrdb_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # remove context extractor from GPU
        context_extractor.to('cpu')
        del context_extractor
        del context
        import gc
        gc.collect()
        sleep(1)

        unet = Unet(use_attention=True, rrdb_num_blocks=17)
        DiffusionTrainer.load_model_weights(model=unet, 
                                            model_path='diffusion_checkpoints/diffusion_model_noise_20250519-090456_best.pth', 
                                            verbose=True)
        unet.eval()
        # Perform inference
        generator = ResidualGenerator(img_size=img_size, predict_mode='noise')
        diff_residual = generator.\
            generate_residuals(model=unet,
                               features=feas,
                               num_inference_steps=args.num_inference_steps)
        # remove the batch dimension, permute to HWC and detach from GPU
        diff_residual = diff_residual.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        constructed_img = up_rrdb_img + diff_residual

        # Normalize the images to [0, 1] range for display
        up_rrdb_img = (up_rrdb_img + 1) / 2
        diff_residual = (diff_residual + 1) / 2
        constructed_img = (constructed_img + 1) / 2

        imgs = [lr_img, up_rrdb_img, diff_residual, constructed_img]

        plot_result(imgs)

def get_args():
    """
    Parse command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument('--lr_img_size', type=int, default=160,
                        help='Target HR image size (height and width). Input HR images should already be this size.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run RRDBNet on (e.g., cuda:0, cuda:1, cpu).')
    parser.add_argument('--rrdb_weights_path', type=str, default='checkpoints_rrdb/rrdb_17_05_16/rrdb_model_best.pth',
                        help='Path to pre-trained RRDBNet weights (.pth file).')
    parser.add_argument('--unet_weights_path', type=str, default='/home/hoang/python/cv_project/diffusion_checkpoints/diffusion_model_noise_20250519-090456_best.pth ',
                        help='Path to pre-trained Unet weights (.pth file).')
    parser.add_argument('--lr_img_path', type=str, default='data/a_000007.jpg',
                        help='Path to the low-resolution image for inference.')
    parser.add_argument('--output_path', type=str, default='result.png',
                        help='Path to save the output image.')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps for the diffusion model.')
    
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())

