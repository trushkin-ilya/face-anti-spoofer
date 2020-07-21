import cv2
import dlib
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from face_segmentation import mobilenet_v1
from face_segmentation.utils.ddfa import ToTensorGjz, NormalizeGjz
from face_segmentation.utils.inference import crop_img, parse_roi_box_from_landmark, predict_dense
from face_segmentation.utils.render import cget_depths_image


class RealSenseVideoEvaluator:
    def __init__(self, model, transform):
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
        self.segmentor = segmentor
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_regressor = dlib.shape_predictor('./face_segmentation/models/shape_predictor_68_face_landmarks.dat')
        self.segmentor_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        self.classifier = model
        self.transform = transform
        self.tri = sio.loadmat('./face_segmentation/visualize/tri.mat')['tri']

    def get_liveness(self, rgb_img, depth_img=None, ir_img=None):
        rects = self.face_detector(rgb_img, 1)
        for rect in rects:
            pts = self.face_regressor(rgb_img, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts)
            img = crop_img(rgb_img, roi_box)
            img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
            input = self.segmentor_transform(img).unsqueeze(0)
            param = self.segmentor(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            vertices = predict_dense(param, roi_box)
            depths_img = cget_depths_image(rgb_img, [vertices], self.tri - 1)
            mask = (depths_img > 0).astype(np.uint8)
            imgs = []
            for img in [rgb_img, depth_img, ir_img]:
                if img is not None:
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=2)
                    channeled_mask = np.stack((mask,) * img.shape[-1], axis=-1)
                    img = cv2.multiply(img, channeled_mask)
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) > 2 else img)
                    img = self.transform(img)
                    imgs += [img]
            input = torch.cat(imgs, dim=0)

            outputs = self.classifier(input.unsqueeze(dim=0))
            yield np.array(roi_box).astype(int), torch.argmax(outputs).item(), F.softmax(outputs, dim=1).max().item()

    def process_rgb_video(self, video_path, output_path=None):
        ## create a device from device id and streams of interest
        video = cv2.VideoCapture(video_path)
        width = (int)(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = (int)(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = (int)(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height)) if output_path else None

        while (video.isOpened()):
            # Capture frame-by-frame
            _, img = video.read()
            for bbox, liveness in self.get_liveness(img):
                color = (0, 255, 0) if liveness else (0, 0, 255)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 6)
            cv2.imshow('frame', img)
            if writer:
                writer.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if writer:
            writer.release()

    def process_5ch_video(self, video_prefix, output_path=None):
        rgb_video = cv2.VideoCapture(f'{video_prefix}_rgb.mp4')
        depth_video = cv2.VideoCapture(f'{video_prefix}_depth.mp4')
        ir_video = cv2.VideoCapture(f'{video_prefix}_ir.mp4')
        width = rgb_video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = rgb_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = rgb_video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) if output_path else None
        while rgb_video.isOpened() and depth_video.isOpened() and ir_video.isOpened():
            # Capture frame-by-frame
            _, rgb_img = rgb_video.read()
            _, depth_img = depth_video.read()
            _, ir_img = ir_video.read()
            for bbox, liveness in self.get_liveness(rgb_img, depth_img, ir_img):
                color = (0, 255, 0) if liveness else (0, 0, 255)
                cv2.rectangle(
                    rgb_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 6)
            cv2.imshow('frame', rgb_img)

            if writer:
                writer.write(rgb_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if writer:
            writer.release()

    def process_4ch_video(self, video_prefix, output_path=None):
        rgb_video = cv2.VideoCapture(f'{video_prefix}_rgb.mp4')
        depth_video = cv2.VideoCapture(f'{video_prefix}_depth.mp4')
        width = rgb_video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = rgb_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = rgb_video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (int(width), int(height))) if output_path else None
        while rgb_video.isOpened() and depth_video.isOpened():
            # Capture frame-by-frame
            _, rgb_img = rgb_video.read()
            _, depth_img = depth_video.read()
            depth_img = np.average(depth_img,axis=2).astype(np.uint8)
            for bbox, liveness, prob in self.get_liveness(rgb_img, depth_img):
                color = (0, 255, 0) if liveness else (0, 0, 255)
                label = f'{prob * 100:.2f}% {"live" if liveness else "spoof"}'
                cv2.rectangle(rgb_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 6)
                labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(rgb_img, (bbox[0], bbox[1]), (bbox[0] + labelSize[0][0], bbox[1] - int(labelSize[0][1])),
                              color, cv2.FILLED)
                cv2.putText(rgb_img, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if not writer:
                cv2.imshow('frame', rgb_img)
            else:
                writer.write(rgb_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if writer:
            writer.release()