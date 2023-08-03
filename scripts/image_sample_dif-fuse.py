import sys
# put your path here
sys.path.extend(['/disk/scratch2/alessandro/new_code/Dif-fuse'])
import matplotlib.pyplot as plt
import argparse
import cv2

import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.brain_datasets import *
import torchvision
import torch
import torch.nn as nn
from torchvision import utils
from torch.utils.data import DataLoader
import nibabel as nib

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

def load_niftii_file(file_path):
    image = nib.load(file_path)
    niifti_data = image.get_fdata()
    niifti_data = niifti_data.astype(np.float32)
    return niifti_data

def thresholdf(x, percentile, independent):
    if independent ==1:
        a = x * (x.numpy() > np.percentile(x, percentile, axis = (-2,-1), keepdims = True))
    else:
        a = x * (x.numpy() > np.percentile(x, percentile, axis = (-3,-2,-1), keepdims = True))
    return a

def clean(saliency, threshold, independent):

    saliency = thresholdf(saliency,threshold, independent)
    return saliency

def normalise(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
def main():
    args = create_argparser().parse_args()

    print(vars(args))

    dist_util.setup_dist()
    logger.configure(experiment_name=args.experiment_name)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    if args.use_fp16:
        model.convert_to_fp16()
    if args.gpus > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path)
    )

    model.eval()

    train_dataset = BRATSDatasetSaliency(
            dataset_root_folder_filepath = 'data/brats2021_slices/images',
            saliency_root_folder_filepath= 'saliency_maps',
            df_path='data/brats2021_train.csv',
            transform = None,
            only_positive = True,
            only_negative = False,
            only_flair=False)

    val_dataset = BRATSDatasetSaliency(
            dataset_root_folder_filepath = 'data/brats2021_slices/images',
            saliency_root_folder_filepath= 'saliency_maps',
            df_path='data/brats2021_val.csv',
            transform = None,
            only_positive = True,
            only_negative = False,
            only_flair=False)

    test_dataset = BRATSDatasetSaliency(
            dataset_root_folder_filepath = 'data/brats2021_slices/images',
            saliency_root_folder_filepath= 'saliency_maps',
            df_path='data/brats2021_test.csv',
            transform = None,
            only_positive = True,
            only_negative = False,
            only_flair=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



    logger.log("sampling...")
    all_images = []

    noise_level = 500
    threshold = 90
    range_t = -1
    kernel = 5
    independent = 0

    for i, (image,_,_,sal, ids) in enumerate(val_loader):
            sample_fn = (

                diffusion.diffuse_loop_forward_backward
            )
            sal = clean(sal, threshold=threshold, independent = independent)
            mask = sal.to(torch.float).to(device)
            if kernel>0:
                for j in range(mask.shape[0]):
                    for each_slice in range(mask.shape[1]):
                        # print(i, each_slice)
                        mask[j, each_slice, :, :] = torchvision.transforms.functional.gaussian_blur(
                            mask[j, each_slice, :, :].view(1, mask.shape[2], mask.shape[3]), kernel_size=kernel).view(
                            mask.shape[2], mask.shape[3])


            mask[mask > 0] = 1
            mask[mask == 0] = 0
            mask = mask.to(device)

            image = image.to(torch.float).to(device)
            for k, level in enumerate(['flair', 't1', 't2', 't1ce']):
                utils.save_image((image[:, k, :, :]).unsqueeze(1),os.path.join(logger.get_dir(), f'batch{i}_threshold_{threshold}_{level}_image.png'), nrow=4)

            rec, sample, orig= sample_fn(
                model = model,
                mask = mask,
                shape = (args.batch_size, 4, args.image_size, args.image_size),
                img =image.to(device),
                clip_denoised=args.clip_denoised,
                noise_level=noise_level,
                cond_fn=None,
                device=device,
                range_t =range_t
            )

            rec_img = rec.to(torch.float)
            sample_img = sample.to(torch.float)
            orig_img = orig.to(torch.float)

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            out_path = os.path.join(logger.get_dir(), 'images')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            delta_rec = torch.abs(sample_img - rec_img)

            for k, level in enumerate(['flair', 't1', 't2', 't1ce']):
                utils.save_image((orig_img[:, k, :, :]).unsqueeze(1), os.path.join(logger.get_dir(),
                                                                                   f'images/batch{i}_noiselevel_{noise_level}_{level}_orig.png'),
                                 nrow=4)
                utils.save_image((sample_img[:, k, :, :]).unsqueeze(1), os.path.join(logger.get_dir(),
                                                                                     f'images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_kernelsize_{kernel}_ind_{independent}_{level}_final.png'),
                                 nrow=4)

            delta_recflair = ((delta_rec[:, 0, :, :]).unsqueeze(1)).cpu().detach()
            # delta_flair = ((delta[:, 0, :, :]).unsqueeze(1)).cpu().detach()



            # post-processing to get anomaly map

            delta_recflair = (delta_recflair * (
                        1.0 / torch.amax(delta_recflair, dim=(-3, -2, -1), keepdim=True))).cpu().detach()
            delta_recflair = (delta_recflair * 255)
            delta_recflair = np.array(delta_recflair, dtype=np.uint8)

            erode = np.zeros((args.batch_size, 1, 256, 256), np.uint8)
            for j in range(delta_recflair.shape[0]):
                a = cv2.erode(delta_recflair[j, 0, :, :], kernel5)
                # a = delta_recflair[j, 0, :, :]
                a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel5)
                er = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel5)
                erode[j, 0, :, :] = er
            fig = plt.figure(figsize=(11,11))
            for j in range(args.batch_size):
                plt.subplot(4, 4, j + 1)
                plt.grid(b=None)
                plt.axis('off')
                plt.imshow(erode[j,:,:,:].squeeze(0), interpolation='none', cmap="Reds")
            out_path_img_anomaly = os.path.join(logger.get_dir(),
                                                    f"images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_kernelsize_{kernel}_ind_{independent}_anomaly_map.png")
            plt.savefig(out_path_img_anomaly)
            plt.close(fig)


            fig = plt.figure(figsize=(11, 11))
            for j in range(args.batch_size):
                plt.subplot(4, 4, j + 1)
                plt.grid(b=None)
                plt.axis('off')
                plt.imshow((orig_img[j, 0, :, :]).detach().cpu().numpy(), cmap=plt.cm.bone)
                plt.imshow(erode[j, 0, :, :], interpolation='none', alpha=0.5, cmap="Reds")


            out_path_img_anomaly_overlayed = os.path.join(logger.get_dir(),
                                                    f"images/batch{i}_noiselevel_{noise_level}_threshold_{threshold}_ranget_{range_t}_kernelsize_{kernel}_anomaly_overlayed.png")

            plt.savefig(out_path_img_anomaly_overlayed)
            plt.close(fig)



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        experiment_name='dif-fuse_sampling',
        gpus=2,
        num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="",
        classifier_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
