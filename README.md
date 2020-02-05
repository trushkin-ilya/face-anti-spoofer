# Deep Learning based Face Anti-Spoofing
Implementation of face anti-spoofing task solution.

## Setup
1. Get CASIA-SURF dataset.
2. Move dataset folder to `data/CASIA_SURF`. Make sure the structure looks like:
```
src/
  ...
data/
  CASIA_SURF/
    dev/
      ...
    train/
      ...
    ...
```
3. Install requirements:
`pip install -r requirements.txt`

## Train
1. Launch tensorboard:
`tensorboard --logdir runs`

2. Run training process:
```
python train.py --protocol <protocol> --batch_size <batch_size> --save_path <save_path>
```

Protocol should be either 1, 2 or 3. It determines CASIA-SURF benchmark protocol.

## Submit
1. Run:
```
python submit.py --model1_path <model1_path> --model2_path <model2_path> --model3_path <model3_path> --num_classes <num_classes=2>
```

Script will create submission.txt in current working directory.
