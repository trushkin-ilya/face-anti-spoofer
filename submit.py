import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import models, transforms
from datasets import CasiaSurfDataset
from torch.utils import data
from models import Ensemble


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
                
    Phase2: After phase1, we will release the testing data in phase2, plus the label of development data. A total of 6 score files need to be submitted:
   (1) merge order：

   4@1_dev_res.txt，4@1_test_res.txt, 4@2_dev_res.txt，4@2_test_res.txt, 4@3_dev_res.txt, 4@3_test_res.txt.

   (2) merge method：

   Continue straight by column. 

The final merged file (for submission) contains a total of 7,200 lines. Each line in the file contains two parts separated by a space. Such as: 

                  dev/003000 0.15361   #Note:  line 1- the first row of 4@1_dev_res.txt

                                ......

                  test/000001 0.94860   #Note:  line 201- the first row of 4@1_test_res.txt           

                                ......

                  dev/003200 0.40134   #Note:  line 2401- the first row of  4@2_dev_res.txt     

                                ......   

                  test/001201 0.23847   #Note:  line 2601- the first row of  4@2_test_res.txt

                                ......

                  dev/003400 0.23394   #Note:  line 4801- the first row of  4@3_dev_res.txt    

                                ......

                  test/001201 0.62544   #Note:  line 5001- the first row of  4@3_test_res.txt  

                                 ......
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model1_path', type=str, required=True)
    argparser.add_argument('--model2_path', type=str, required=True)
    argparser.add_argument('--model3_path', type=str, required=True)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--output', type=str, default='submission.txt')
    argparser.add_argument('--num_workers', type=int, default=0)
    args = argparser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Ensemble(num_classes=args.num_classes, device=device)

    for protocol in [1, 2, 3]:
        model.load_state_dict(torch.load(getattr(args, f'model{protocol}_path'), map_location=device))
        model = model.to(device)
        print(f"Evaluating protocol {protocol}...")
        model.eval()
        for mode in ['dev', 'test']:
            dataset = CasiaSurfDataset(protocol, mode=mode, transform=transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
            dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            df = pd.DataFrame(columns=['prob', 'video_id'], index=np.arange(len(dataloader) * args.batch_size))
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(dataloader)):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    liveness_prob = F.softmax(outputs, dim=1)[:, 1]
                    idx = np.arange(i * args.batch_size, (i + 1) * args.batch_size)
                    for j, p in zip(idx, liveness_prob):
                        video_id = dataset.get_video_id(j)
                        df.iloc[j] = {'prob': p.item(), 'video_id': video_id}

                df.dropna(inplace=True)
                df['prob'] = pd.to_numeric(df['prob'])
                df.groupby('video_id', sort=False).mean().to_csv(args.output, sep=' ', header=False,
                                                                 float_format='%.5f', mode='a')
