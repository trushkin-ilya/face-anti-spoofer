# Face Anti-Spoofer ![](https://img.shields.io/badge/python-3.7-informational?logo=python&logoColor=ccc) ![](https://img.shields.io/badge/PyTorch-1.4-informational?logo=pytorch&logoColor=ccc)
Face anti-spoofing task solution using CASIA-SURF CeFA dataset, [FeatherNets](https://github.com/trushkin-ilya/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019) and [Face Alignment in Full Pose Range](https://github.com/trushkin-ilya/3DDFA).
<!--ts-->
  * [Requirements](#requirements)
  * [Setup](#setup)
  * [Train](#train)
  * [Test](#test)
     * [CASIA-SURF](#casia-surf)
     * [Demo with Intel® RealSense™ camera](#demo-with-intel-realsense-camera)
        * [Running](#running)
        * [Demo](#demo)
  * [Submit](#submit)
<!-- Added by: itrushkin, at: Чт июл  2 23:34:25 MSK 2020 -->

<!--te-->



| Model |  Params, M | Computational complexity, MFLOPs | RGB | Depth | IR |Loss function | Optimal LR | Minimal ACER (CASIA-SURF val) | Snapshot |
| --- | --- | ---| --- | --- | --- | --- | --- | --- | --- |
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 3e-6| 0.0242 | [Download](https://drive.google.com/file/d/1IUcyz-rc2s-uTAHgTEI8x98WkexkANC8/view?usp=sharing) |
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :heavy_check_mark: | :x: | Cross-entropy | 3e-6| 0.0174 | [Download](https://drive.google.com/file/d/1TnZonwS1dPs7lLtOjrPtb9JCDQF3u8uR/view?usp=sharing)
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-7 |  0.0397 | [Download](https://drive.google.com/file/d/1Fx6umkuQElj-3rAXJfWcDskCsKO2wxdc/view?usp=sharing)
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :x: | :x: | Focal loss | 3e-6 | 0.0066 | [Download](https://drive.google.com/file/d/1O4Aw3J4Y8C8Ozf2nylYHMLULNOHrGPiE/view?usp=sharing)
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 3e-7 | 0.1542 | [Download](https://drive.google.com/file/d/1DU3pCwUJujoYQmg8XJXzQepNJp7SwG1Y/view?usp=sharing)
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :heavy_check_mark: | :x: | Cross-entropy | 3e-6| 0.1019 | [Download](https://drive.google.com/file/d/1HTHohYimOIKSkusrdfscznr2pDuWQePa/view?usp=sharing)
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 3e-6 | 0 | [Download](https://drive.google.com/file/d/14tAwhKCouix4Jgoe6XtU9WsQUJ4nIRVW/view?usp=sharing)
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :x: | :x: | Focal loss | 3e-7| 0.1666 | [Download](https://drive.google.com/file/d/1VBEYg-6Gs5cSiQadlvUbxNqRw7jDQGqk/view?usp=sharing)
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 1e-3 | 0.0049 | [Download](https://drive.google.com/file/d/1JtvbBqa-mvx61l-PXO0gq48KCstw_R1L/view?usp=sharing)
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-3 | 0 | [Download](https://drive.google.com/file/d/1Ap0e8PiIFdE49v9ihzsdv4TxubANeW7G/view?usp=sharing)
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Focal loss| 1e-4 | 0.0021 | [Download](https://drive.google.com/file/d/1KUnfBUggjWP3TzFc7IrsE0UopV-8hYRx/view?usp=sharing)
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 1e-3 | 0.0001 | [Download](https://drive.google.com/file/d/1eSKpZi6EsNnj69CRq4HMSOpwu5PnFkM5/view?usp=sharing)
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-3 | 0 | [Download](https://drive.google.com/file/d/1nbQAqYrToPTkRSB8NIwnJWYlu6gHF2tU/view?usp=sharing)
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Focal loss | 1e-4 | 0.0034 | [Download](https://drive.google.com/file/d/1nsT7aBzzeqseWd6c1eZwKc_t3XkMXiVV/view?usp=sharing)

[TensorBoard](https://tensorboard.dev/experiment/H45WwYpPTmGzPmcdjZWmYg/)


## Requirements
* Python 3.7.6
* PyTorch 1.4.0

## Setup
1. Get CASIA-SURF dataset.
2. Move dataset folder to `./data/CASIA_SURF`:
```
ln -s <your_path_to_CASIA> ./data/CASIA_SURF
```
3. Install requirements:
```pip install -r requirements.txt```


## Train

1. Tensorboard logs will be written to `./runs` folder. To monitor them during training process, run:
```
tensorboard --logdir runs
```

2. Run training process:
```
python train.py --protocol PROTOCOL --config-path CONFIG_PATH --data_dir DATA_DIR
                [--epochs 10] [--checkpoint ''] [--train_batch_size 1]
                [--val_batch_size 1] [--eval_every 1] [--save_path checkpoints]
                [--num_classes NUM_CLASSES] [--save_every 1] [--num_workers 0]
```
Protocol must be either 1, 2 or 3. It determines CASIA-SURF benchmark sub-protocol of Protocol 4.

## Test
### CASIA-SURF
![](https://storage.googleapis.com/groundai-web-prod/media/users/user_299614/project_411398/images/fig/eccv_fig0.png)
1. When you have the model, you can test it by running:
```
python test.py --protocol PROTOCOL --checkpoint CHECKPOINT --config-path CONFIG_PATH 
               [--data-dir DATA_DIR] [--num_classes NUM_CLASSES] [--batch_size BATCH_SIZE]
               [--visualize VISUALIZE] [--num_workers NUM_WORKERS] [--video_path VIDEO_PATH]
```
Protocol must be either 1, 2 or 3. It determines CASIA-SURF benchmark sub-protocol of Protocol 4.

### Demo with Intel® RealSense™ camera
#### Running
```
python realsense_demo.py --video-path VIDEO_PATH --config-path CONFIG_PATH  [--num_classes NUM_CLASSES]
```
**WARNING:** Current evaluation for RealSense cameras was developed only for legacy devices which supported by [pyrealsense](https://github.com/toinsson/pyrealsense) library. Everything works fine for F200.
#### Demo
<p align="center">
  <img src="misc/out.gif">
</p>

## Submit
Submission is made for Face Anti-spoofing Detection Challenge at CVPR2020.
1. Run:
```
python submit.py --model1_path MODEL1_PATH --model2_path MODEL2_PATH --model3_path MODEL3_PATH 
                 [--num_classes 2] [--batch_size 1] [--output submission.txt]
                 [--num_workers 0] 
```
