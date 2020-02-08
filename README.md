# Deep Learning based Face Anti-Spoofing
Implementation of face anti-spoofing task solution.

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
1. Launch tensorboard:
`tensorboard --logdir runs`

2. Run training process:
```
python train.py --protocol PROTOCOL 
                [--epochs EPOCHS]
                [--checkpoint CHECKPOINT]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--val_batch_size VAL_BATCH_SIZE] 
                [--eval_every EVAL_EVERY]
                --save_path SAVE_PATH 
                [--lr LR]
                [--num_classes NUM_CLASSES]
                [--save_every SAVE_EVERY]
                [--num_workers NUM_WORKERS]

```
Default parameters:
* `--epochs = 10`
* `--checkpoint = None`
* `--train_batch_size = 1`
* `--val_batch_size = 1`
* `--eval_every = 1`
* `--lr = 3e-3`
* `--num_classes = 2`
* `--save_every = 1`
* `--num_workers = 0`


Protocol should be either 1, 2 or 3. It determines CASIA-SURF benchmark protocol.

## Test
1. When you have the model, you can test it by running:
```
python test.py --protocol PROTOCOL
               [--data-dir DATA_DIR]
               --checkpoint CHECKPOINT
               [--num_classes NUM_CLASSES]
               [--batch_size BATCH_SIZE]
```
Default parameters:
* `--data-dir = ./data/CASIA_SURF`
* `--num-classes = 2`
* `--batch-size = 1`

## Submit
1. Run:
```
python submit.py --model1_path MODEL1_PATH
                 --model2_path MODEL2_PATH
                 --model3_path MODEL3_PATH
                 [--num_classes NUM_CLASSES]
```
Default parameters:
* `--num_classes = 2`
Script will create submission.txt in current working directory.
