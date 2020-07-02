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
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--ir', type=bool, default=False)
    argparser.add_argument('--num_classes', type=int, default=2)
    args = argparser.parse_args()
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models, config['model'])(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    transform = ValidationTransform()
    evaluator = RealSenseVideoEvaluator(model, transform)
    with torch.no_grad():
        if args.depth and args.ir:
            evaluator.process_5ch_video(args.video_path, 'result.mp4')
        elif args.depth:
            evaluator.process_4ch_video(args.video_path, 'result.mp4')
        else:
            evaluator.process_rgb_video(args.video_path, 'result.mp4')
