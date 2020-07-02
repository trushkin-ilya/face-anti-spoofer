from torchvision import transforms
from baseline.datasets import NonZeroCrop
from . import RealSenseVideoEvaluator
from argparse import ArgumentParser
import yaml
import models

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--video-path', type=str, required=True)
    argparser.add_argument('--config-path', type=str, required=True)
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--ir', type=bool, default=False)
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--num_classes', type=int, default=2)
    config = yaml.load('FeatherNetA_4ch_FL.yaml',Loader=yaml.FullLoader)
    args = argparser.parse_args()
    model = getattr(models, config['model'])(num_classes=args.num_classes)
    transform = transforms.Compose([
        NonZeroCrop(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    evaluator = RealSenseVideoEvaluator(model, transform)
    if args.depth and args.ir:
        evaluator.process_5ch_video(args.video_path, 'result.mp4')
    elif args.depth:
        evaluator.process_4ch_video(args.video_path, 'result.mp4')
    else:
        evaluator.process_rgb_video(args.video_path, 'result.mp4')