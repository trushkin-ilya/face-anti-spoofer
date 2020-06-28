# Face Anti-Spoofer
<p align="center">
  <img src="out.gif">
</p>

Face anti-spoofing task solution using CASIA-SURF CeFA dataset, [FeatherNets](https://github.com/trushkin-ilya/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019) and [Face Alignment in Full Pose Range](https://github.com/trushkin-ilya/3DDFA).

| Model |  Params, M | Computational complexity, MFLOPs | RGB | Depth | IR |Loss function | Optimal LR | Minimal ACER (CASIA-SURF val) |
| --- | --- | ---| --- | --- | --- | --- | --- | --- |
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 3e-6| 0.0068 |
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-7 |  0 |
| FeatherNet | 0.35 | 79.99 | :heavy_check_mark: | :x: | :x: | Focal loss | 3e-6 | 0.0117 | 
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 3e-7 | 0.0397 |
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 3e-6 | 0 |
| MobileLiteNet | 0.57 | 270.91 | :heavy_check_mark: | :x: | :x: | Focal loss | 3e-7| 0.0495 |
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 1e-3 | 0.0304 |
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-3 | 0.0004 |
| ResNet18 | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Focal loss| 1e-4 | 0.03717 | 
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Cross-entropy | 1e-3 | 0.1244 |
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Cross-entropy | 1e-3 | 0.0001 |
| ResNet18 with dropout | 13.95 | 883730 | :heavy_check_mark: | :x: | :x: | Focal loss | 1e-4 | 0.0548 | 

[TensorBoard](https://tensorboard.dev/experiment/SoIKSMcbRniYID003q5glw/#scalars)


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
`pip install -r requirements.txt`


## Train
1. Configure training process. In `./config.yaml` specify keys:
    * `model`: class name of used model. Class must be imported from `models` module.
    * `loss_fn`: loss function to use during training process. Must be imported from `./losses.py`.
    * section `optimizer:`
        * ` class`: class name of used optimizer. Must be imported from `./optimizers.py`.
        * `lr`: optimizer learning rate parameter value.
    * `lr_scheduler`: PyTorch learning rate scheduler class name.
        

2. Tensorboard logs will be written to `./runs` folder. To monitor them during training process, run:
```
tensorboard --logdir runs
```

3. Run training process:
```
python train.py --protocol PROTOCOL
                --checkpoint CHECKPOINT 
                [--epochs 10]
                [--train_batch_size 1]
                [--val_batch_size 1] 
                [--eval_every 1]
                [--save_path checkpoints] 
                [--num_classes 2]
                [--save_every 1]
                [--num_workers 0]
```
Protocol must be either 1, 2 or 3. It determines CASIA-SURF benchmark sub-protocol of Protocol 4.

## Test
### CASIA-SURF
![](https://storage.googleapis.com/groundai-web-prod/media/users/user_299614/project_411398/images/fig/eccv_fig0.png)
1. Configure `./config.yaml`:
 * `model`: class name of used model. Class must be imported from `models` module.

2. When you have the model, you can test it by running:
```
python test.py --protocol PROTOCOL
               --checkpoint CHECKPOINT
               [--data-dir ./data/CASIA_SURF]
               [--num_classes 2]
               [--batch_size 1]
               [--visualize False]
               [--num_workers 0]
               [--depth False]
               [--ir False]
```
Protocol must be either 1, 2 or 3. It determines CASIA-SURF benchmark sub-protocol of Protocol 4.

### Intel® RealSense™ camera
1. Configure `./config.yaml`:
 * `model`: class name of used model. Class must be imported from `models` module.
2. When you have the model, you can test it by running:
```
python test.py --checkpoint CHECKPOINT
               --video_path VIDEO_PATH
               [--num_classes 2]
               [--depth False]
               [--ir False]              
```
**WARNING:** Current evaluation for RealSense cameras was developed only for legacy devices which supported by [pyrealsense](https://github.com/toinsson/pyrealsense) library. Everything works fine for F200.
## Submit
Submission is made for Face Anti-spoofing Detection Challenge at CVPR2020.
1. Run:
```
python submit.py --model1_path MODEL1_PATH
                 --model2_path MODEL2_PATH
                 --model3_path MODEL3_PATH
                 [--num_classes 2]
                 [--batch_size 1]
                 [--output ./submission.txt]
                 [--num_workers 0]
                 [--depth False]
                 [--ir False]
```
