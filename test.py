from baseline.datasets import CasiaSurfDataset, NonZeroCrop
from tqdm import tqdm
from torchvision import transforms
import models
from torch.utils import data
from torch import nn
from sklearn import metrics
from argparse import ArgumentParser
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def evaluate(dataloader: data.DataLoader, model: nn.Module, visualize: bool = False):
    device = next(model.parameters()).device
    model.eval()
    print("Evaluating...")
    tp, tn, fp, fn = 0, 0, 0, 0
    errors = np.array([], dtype=[('img', torch.Tensor),
                                 ('label', torch.Tensor), ('prob', float)])
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images, labels = batch
            outputs = model(images.to(device))
            outputs = outputs.cpu()
            tn_batch, fp_batch, fn_batch, tp_batch = metrics.confusion_matrix(y_true=labels,
                                                                              y_pred=torch.max(
                                                                                  outputs.data, 1)[1],
                                                                              labels=[0, 1]).ravel()
            if visualize:
                errors_idx = np.where(torch.max(outputs.data, 1)[1] != labels)
                print(errors_idx)
                errors_imgs = list(
                    zip(images[errors_idx], labels[errors_idx], ))
                print(errors_imgs)
                errors = np.append(errors, errors_imgs)

            tp += tp_batch
            tn += tn_batch
            fp += fp_batch
            fn += fn_batch
    apcer = fp / (tn + fp) if fp != 0 else 0
    bpcer = fn / (fn + tp) if fn != 0 else 0
    acer = (apcer + bpcer) / 2
    if visualize:
        print(errors)
        errors.sort(order='prob')
        errors = np.flip(errors)
        print(errors)
        utils.plot_classes_preds(model, zip(*errors))

    return apcer, bpcer, acer


def get_liveness(img_ori, face_detector, face_regressor,
                 segmentor_transform, segmentor, transform, model):
    from face_segmentation.utils.inference import crop_img, parse_roi_box_from_landmark, predict_dense
    from face_segmentation.utils.render import cget_depths_image
    import scipy.io as sio
    from PIL import Image
    rects = face_detector(img_ori, 1)
    tri = sio.loadmat(
        './face_segmentation/visualize/tri.mat')['tri']
    for rect in rects:
        pts = face_regressor(img_ori, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)
        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(120, 120),
                         interpolation=cv2.INTER_LINEAR)
        input = segmentor_transform(img).unsqueeze(0)
        param = segmentor(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        vertices = predict_dense(param, roi_box)
        depths_img = cget_depths_image(img_ori, [vertices], tri - 1)
        mask = (depths_img > 0).astype(np.uint8)
        mask = np.stack((mask,) * 3, axis=-1)
        img = cv2.multiply(img_ori, mask)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input = transform(img).unsqueeze(dim=0)
        outputs = model(input)
        yield np.array(roi_box).astype(int), torch.argmax(outputs).item()


def process_video(video_path, model, transform):
    from face_segmentation import mobilenet_v1
    from face_segmentation.utils.ddfa import ToTensorGjz, NormalizeGjz
    import dlib
    checkpoint_fp = './face_segmentation/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(
        checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    # 62 = 12(pose) + 40(shape) +10(expression)
    segmentor = getattr(mobilenet_v1, arch)(num_classes=62)

    segmentor_dict = segmentor.state_dict()
    # because the segmentor is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        segmentor_dict[k.replace('module.', '')] = checkpoint[k]
    segmentor.load_state_dict(segmentor_dict)
    segmentor.eval()
    face_detector = dlib.get_frontal_face_detector()
    face_regressor = dlib.shape_predictor(
        './face_segmentation/models/shape_predictor_68_face_landmarks.dat')
    segmentor_transform = transforms.Compose(
        [ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    video = cv2.VideoCapture(video_path)
    print(video.open(video_path))
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 30, (frame_width, frame_height))

    while(video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if not ret:
            print(ret)
            break
        print('ok')
        # Our operations on the frame come here
        for bbox, liveness in get_liveness(frame, face_detector, face_regressor, segmentor_transform, segmentor, transform, model):
            color = (0, 255, 0) if liveness else (0, 0, 255)
            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 6)
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video.release()
    out.release()
    cv2.destroyAllWindows()


def main(args):
    model = getattr(models, args.model)(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    transform = transforms.Compose([
        NonZeroCrop(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    model.eval()
    with torch.no_grad():
        if args.video_path:
            process_video(args.video_path, model, transform)
            return

        dataset = CasiaSurfDataset(
            args.protocol, mode='dev', dir=args.data_dir, transform=transform, depth=args.depth, ir=args.ir)
        dataloader = data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        apcer, bpcer, acer = evaluate(dataloader, model, args.visualize)
        print(f'APCER: {apcer}, BPCER: {bpcer}, ACER: {acer}')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--protocol', type=int, required=True)
    argparser.add_argument('--data-dir', type=str,
                           default=os.path.join('data', 'CASIA_SURF'))
    argparser.add_argument('--checkpoint', type=str, required=True)
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--visualize', type=bool, default=False)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--image_path', type=str)
    argparser.add_argument('--video_path', type=str)
    argparser.add_argument('--model', type=str, required=True)
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--ir', type=bool, default=False)
    args = argparser.parse_args()

    main(args)
