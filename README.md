# <p align=center> [CVPR 2025] Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining</p>

[![paper](https://img.shields.io/badge/CVF-paper-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Semi-Supervised_State-Space_Model_with_Dynamic_Stacking_Filter_for_Real-World_Video_CVPR_2025_paper.pdf)
[![arxiv](https://img.shields.io/badge/arxiv-paper-red.svg)](https://arxiv.org/abs/2505.16811)
[![Closed Issues](https://img.shields.io/github/issues-closed/sunshangquan/VDMamba)](https://github.com/sunshangquan/VDMamba/issues?q=is%3Aissue+is%3Aclosed) 
[![Open Issues](https://img.shields.io/github/issues/sunshangquan/VDMamba)](https://github.com/sunshangquan/VDMamba/issues) 

[RVDT Dataset](https://drive.google.com/drive/folders/1o3WZlYRuZAda6bnnak4yhbVzWN5262UN)

[Visual Results](https://github.com/sunshangquan/VDMamba?tab=readme-ov-file#visual-results)

# Requirment

Python 3.10

Numpy

PyTorch 2.1.1

torchvision 0.10.1

# Visual Results

|    Models    |                     NTURain                       |                      RainsynLight25                       |  RainsynComplex25 |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MSCSC | [Google Drive](https://drive.google.com/file/d/1TG1TmY1-1q4ZPuLnBPd8t7F7zpxf0_zH/view?usp=sharing) | | |
| SLDNet | [Google Drive](https://drive.google.com/file/d/1D3OpTigvXii8g4p2fycBmI9P9sUtwz5C/view?usp=sharing) | | |
| S2VD | [Google Drive](https://drive.google.com/file/d/1k2RLW6WGiiM0tR3Xc8MFkDUOJ6SGeC6V/view?usp=sharing) | | |
| MFGAN | [Google Drive](https://drive.google.com/file/d/1sRW2g3KnjlKAd2mXngiATBv1NDzmXbgT/view?usp=sharing) | | |
| MPEVNet | [Google Drive](https://drive.google.com/file/d/17sfbWY3c5Xdjaf34JNuMIIRiB8WXHzxi/view?usp=sharing) | | |
| VDMamva | [Google Drive](https://drive.google.com/file/d/1qnxC9ww7xrqC1MMPLyyJmxbhhpA_ktcW/view?usp=sharing) | | |

# Data preparation

## NTURain

1. Download the NTURain dataset from https://github.com/hotndy/SPAC-SupplementaryMaterials

2. Extract all clips from .rar files

3. The extracted files should have the following structure:

```
├── Dataset_Testing_RealRain

    ├── ra1_Rain

    ...

    ├── rb3_Rain  

├── Dataset_Testing_Synthetic

    ├── a1_Rain

    ├── a1_GT

    ...

    ├── b4_Rain

    ├── b4_GT 

├── Dataset_Training_Synthetic

    ├── t1_Rain_01

    ...

    ├── t8_Rain_03 

```

## RainSynComplex25 & RainSynLight25

1. Download the RainSynComplex25 & RainSynLight25 dataset from https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018

2. Extract all clips from .zip files

3. The extracted files should have the following structure:

```
├── RainSynComplex25 or RainSynLight25

    ├── video_rain_heavy or video_rain_light

        ├── train

            ├── input

                ...

            ├── processed

                ...

        ├── test

            ├── input

                ...

            ├── processed

                ...


```

## RVDT

It is available in [Google Drive](https://drive.google.com/drive/folders/1o3WZlYRuZAda6bnnak4yhbVzWN5262UN).


# Train

1. Modify the configurations in train_video_rain.sh and comment out other unnecessary commands.

2. Since we borrow the re-implementation of lightflownet3 from https://github.com/lhao0301/pytorch-liteflownet3 and https://github.com/NVIDIA/flownet2-pytorch, you should follow their step of installing correlation_package.

3. run the code 

```
bash train_video_rain.sh
```

# Test for NTURain

- run the code 

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py \
    -list_filename lists/nturain_test.txt \
    -epoch 100 \
    -data_dir /home1/ssq/proj1/evnet/data/image/Dataset_Testing_Synthetic/ \
    -checkpoint_dir ./checkpoints/ \
    -model_name nturain5_vdmamba
```

# Test for RainSynLight25

- run the code 

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py 
    -list_filename lists/rainsynlight25_test.txt 
    -epoch 25 
    -data_dir /home1/ssq/data/RainSynLight25/video_rain_light/test/ 
    -checkpoint_dir ./checkpoints/ 
    -model_name rainsynlight25_vdmamba
    -file_suffix .png
```

# Test for RainSynComplex25

- run the code 

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py 
    -list_filename lists/rainsyncomplex25_test.txt 
    -epoch 82 
    -data_dir /home1/ssq/data/RainSynComplex25/video_rain_heavy/test/ 
    -checkpoint_dir ./checkpoints/ 
    -model_name rainsyncomplex25_vdmamba
    -file_suffix .png 
```
# Test for real video

1. Modify the configurations in test_video_rain.sh

2. run the code 

```
bash test_video_rain.sh
```

# Compute PSNR and SSIM

```
python comp_psnr_ssim.py --path1 [restoration_result] --path2 [ground_truth]
```


# BibTex

```
@inproceedings{sun2025semi,
  title={Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining},
  author={Sun, Shangquan and Ren, Wenqi and Zhou, Juxiang and Wang, Shu and Gan, Jianhou and Cao, Xiaochun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26114--26124},
  year={2025}
}
```
