from argparse import ArgumentParser

import torch
import yaml

import models
from realsense import RealSenseVideoEvaluator
from transforms import ValidationTransform

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--video-path', type=str, required=True)
    argparser.add_argument('--config-path', type=str, required=True)
    argparser.add_argument('--checkpoint', type=str, required=True)
    args = argparser.parse_args()
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, config['model'])(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    transform = ValidationTransform()
    evaluator = RealSenseVideoEvaluator(model, transform)
    with torch.no_grad():
        if config['depth'] and config['ir']:
            evaluator.process_5ch_video(args.video_path, 'result.mp4')
        elif config['depth']:
            evaluator.process_4ch_video(args.video_path, 'result.mp4')
        else:
            evaluator.process_rgb_video(args.video_path, 'result.mp4')
