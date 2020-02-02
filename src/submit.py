import argparse
from torchvision import models
from datasets import CasiaSurfDataset
from torch.utils import data
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    '''
    Phase1: 4@1_train.txt，4@2_train.txt and 4@3_train.txt  are used to train the models 4@1，4@2 and 4@3 respectively.  Then,          
                   model4@1 is used to predict the sample scores in 4@1_dev_res.txt; 
                   model4@2 is used to predict the sample scores in 4@2_dev_res.txt; 
                   model4@3 is used to predict the sample scores in 4@3_dev_res.txt.
Therefore, there are three predicted sample score files in phase1. In order to submit results at one time, participants need to combine the 3 predicted files into one file before result submission via codalab system. The merge way is below:
                 (1) merge order：
                  4@1_dev_res.txt，4@2_dev_res.txt，4@3_dev_res.txt.
                 (2) merge way：
                  Continue straight by column. 
The final merged file (for submission) contains a total of 600 lines. Each line in the file contains two parts separated by a space. The first part is the relative path of each video, and the second part is the prediction score given by the model (representing the probability that the sample (video) belongs to the real face). Such as:
                dev/003000 0.15361   #Note:  line 1- the first row of 4@1_dev_res.txt
                                ......
                dev/003199 0.15361   #Note:  line 200- the last row of 4@1_dev_res.txt
                dev/003200 0.40134   #Note:  line 201- the first row of 4@2_dev_res.txt
                                ......
                 dev/003399 0.40134   #Note:  line 400- the last row of 4@2_dev_res.txt
                dev/003400 0.23394    #Note:  line 401- the first row of 4@3_dev_res.txt
                                ......
                dev/003599 0.23394    #Note:  line 600- the last row of 4@3_dev_res.txt
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model1_path', type=str, required=True)
    argparser.add_argument('--model2_path', type=str, required=True)
    argparser.add_argument('--model3_path', type=str, required=True)
    args = argparser.parse_args()
    model = models.mobilenet_v2(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with open('submission.txt', 'w+') as submission:
        for protocol in [1, 2, 3]:
            model.load_state_dict(torch.load(args[f'model{protocol}_path']))
            print(f"Evaluating protocol {protocol}...")
            model.eval()
            dataset = CasiaSurfDataset(protocol, train=False)
            dataloader = data.DataLoader(
                dataset, sampler=torch.utils.data.SubsetRandomSampler(range(0, 100)))
            result = {}
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader)):
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    liveness_prob = F.softmax(outputs, dim=1)[0][1]
                    video_id = dataset.get_video_id(i)
                    if video_id not in result:
                        result[video_id] = []
                    result[video_id].append(liveness_prob)
            for video_id, frame_probs in result.items():
                print(video_id, np.mean(frame_probs))
                submission.write(f'{video_id} {np.mean(frame_probs):.5f}\n')
        submission.close()
