import argparse
import cv2
import math, os
import torch
import torch.nn as nn
import numpy as np
from skimage import img_as_ubyte, img_as_float32
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from natsort import natsorted
def str2bool(x):
    return x.lower() == 'true'

def str2none(x):
    return None if str(x).lower() == 'none' else x

def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    :parame img: uint8 or float ndarray
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                      [128.553, -74.203, -93.786],
                                      [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_im_type)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C, [0,255]
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, .5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.shape[0] <= 3:
        img1 = img1.transpose(1,2,0)
        img2 = img2.transpose(1,2,0)
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_PSNR(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img_as_ubyte(img)
    Iclean = img_as_ubyte(imclean)
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,], Img[i,:,], border)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img_as_ubyte(img)
    Iclean = img_as_ubyte(imclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,], Img[i,:,], border)
    return (SSIM/Img.shape[0])

def compute_video(gt_path, pred_path, ycbcr=True):
    print(gt_path, pred_path)
    gt_files = natsorted([i for i in Path(gt_path).glob("*.jpg")] + [i for i in Path(gt_path).glob("*.png")] + [i for i in Path(gt_path).glob("*.JPG")])
    pred_files = natsorted([i for i in Path(pred_path).glob("*.jpg")] + [i for i in Path(pred_path).glob("*.png")] + [i for i in Path(pred_path).glob("*.JPG")])
    print(len(pred_files), len(gt_files))
    if len(pred_files) + 6 == len(gt_files):
        gt_files = gt_files[3:-3]
    elif len(pred_files) + 7 == len(gt_files):
        gt_files = gt_files[3:-4]
    total_psnr, total_ssim = [], []
    for gt_file, pred_file in zip(gt_files, pred_files):
        print("Reading {}".format(str(pred_file)))
        print("Reading {}".format(str(gt_file)))
        gt_img = img_as_float32(cv2.imread(str(gt_file), flags=cv2.IMREAD_COLOR)[:,:,::-1])
        pred_img = img_as_float32(cv2.imread(str(pred_file), flags=cv2.IMREAD_COLOR)[:,:,::-1])

        gt_seq = torch.from_numpy(gt_img.transpose([2,0,1])[np.newaxis,])
        pred_seq = torch.from_numpy(pred_img.transpose([2,0,1])[np.newaxis,])

        seq_psnr = batch_PSNR(gt_seq, pred_seq, ycbcr=ycbcr)
        print("Computing PSNR", seq_psnr)
        seq_ssim = batch_SSIM(gt_seq, pred_seq, ycbcr=ycbcr)
        print("Computing SSIM", seq_ssim)

        total_psnr.append(seq_psnr)
        total_ssim.append(seq_ssim)
    video_psnr = np.mean(total_psnr)
    video_ssim = np.mean(total_ssim)
    print("Video psnr is", video_psnr)
    print("Video ssim is", video_ssim)
    return video_psnr, video_ssim, total_psnr, total_ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('--path1', type=str,default= "") # modify the experiments name-->modify all save path
    parser.add_argument('--path2', type=str,default= "")
    parser.add_argument('--dataid', type=int,default= 0)
    args = parser.parse_args()

    ycbcr = True
    datasets = ['nturain', 'rainvidss', 'rainsynlight25', 'rainsyncomplex25']
    dataset = datasets[args.dataid]
    if dataset == 'nturain':
        gt_path = args.path2 # "/home1/ssq/proj1/evnet/data/image/Dataset_Testing_Synthetic/"
        pred_path = args.path1
        gt_paths = natsorted([i for i in Path(gt_path).glob('*_GT')])
        pred_paths = natsorted([i for i in Path(pred_path).glob('*_Rain')])
    elif dataset == 'rainvidss':
        gt_path = args.path2 # '/home1/ssq/proj1/evnet/data/image/dataset_RainVIDSS/val/gt/'
        gt_paths = natsorted([i for i in Path(gt_path).glob('seq_0*')])
        pred_paths = natsorted([i for i in Path(pred_path).glob('seq_0*')])
    elif dataset == 'rainsynlight25':
        pred_path = args.path1
        gt_path = args.path2 # '/home1/ssq/data/RainSynLight25/video_rain_light/test/processed/'
        gt_paths = natsorted([i for i in Path(gt_path).glob('*')])
        pred_paths = natsorted([i for i in Path(pred_path).glob('*')])
    elif dataset == 'rainsyncomplex25':
        pred_path = args.path1
        gt_path = args.path2 # '/home1/ssq/data/RainSynComplex25/video_rain_heavy/test/processed/'
        gt_paths = natsorted([i for i in Path(gt_path).glob('*')])
        pred_paths = natsorted([i for i in Path(pred_path).glob('*')])

    
    
    total_psnr, total_ssim, frames_psnrs, frames_ssims = [], [], [], []
    
    print(len(pred_paths), len(gt_paths))
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        video_psnr, video_ssim, frames_psnr, frames_ssim = compute_video(gt_path, pred_path, ycbcr=ycbcr)
        total_psnr.append(video_psnr)
        total_ssim.append(video_ssim)
        frames_psnrs += frames_psnr
        frames_ssims += frames_ssim
        with open(os.path.join(pred_path, '..', 'result.txt',), 'a+') as f_:
            f_.write("{}, {}\n".format(video_psnr, video_ssim))
    print("Total psnr is", np.mean(total_psnr), len(total_psnr))
    print("Total ssim is", np.mean(total_ssim))
    print("Frame psnr is", np.mean(frames_psnrs), len(frames_psnrs))
    print("Frame ssim is", np.mean(frames_ssims))

