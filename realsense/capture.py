import pyrealsense as pyrs
from pyrealsense.stream import ColorStream, DACStream, InfraredStream, DepthStream, CADStream
import cv2
import numpy as np
import argparse


def convert_z16_to_bgr(frame):
    '''Performs depth histogram normalization
    This raw Python implementation is slow. See here for a fast implementation using Cython:
    https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/cython_methods/methods.pyx
    '''
    hist = np.histogram(frame, bins=0x10000)[0]
    hist = np.cumsum(hist)
    hist -= hist[0]
    rgb_frame = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)

    zeros = frame == 0
    non_zeros = frame != 0

    f = hist[frame[non_zeros]] * 255 / hist[0xFFFF]
    rgb_frame[non_zeros, 0] = 255 - f
    rgb_frame[non_zeros, 1] = 0
    rgb_frame[non_zeros, 2] = f
    rgb_frame[zeros, 0] = 20
    rgb_frame[zeros, 1] = 5
    rgb_frame[zeros, 2] = 0
    return rgb_frame


def rs_transform_point_to_point(to_point, extrin, from_point):
    to_point[0] = extrin.rotation[0] * from_point[0] + extrin.rotation[3] * from_point[1] + extrin.rotation[6] * \
                  from_point[2] + extrin.translation[0]
    to_point[1] = extrin.rotation[1] * from_point[0] + extrin.rotation[4] * from_point[1] + extrin.rotation[7] * \
                  from_point[2] + extrin.translation[1]
    to_point[2] = extrin.rotation[2] * from_point[0] + extrin.rotation[5] * from_point[1] + extrin.rotation[8] * \
                  from_point[2] + extrin.translation[2]


def save_camera_data_to_files(device_id=0, rgb: bool = True, depth: bool = True, ir: bool = False, frames: int = 150,
                              outfile_pattern: str = "output"):
    ## start the service - also available as context manager
    with pyrs.Service() as serv:
        streams = []
        rgb_writer, depth_writer, ir_writer = None, None, None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if rgb:
            rgb_stream = ColorStream(width=1280, height=720)
            streams.append(rgb_stream)
            rgb_writer = cv2.VideoWriter(f'{outfile_pattern}_rgb.mp4', fourcc, 30,
                                         (rgb_stream.width, rgb_stream.height))
        if depth:
            depth_stream = DepthStream()
            dac_stream = DACStream()
            cad_stream = CADStream()
            streams.append(depth_stream)
            streams.append(dac_stream)
            streams.append(cad_stream)
            depth_writer = cv2.VideoWriter(f'{outfile_pattern}_depth.mp4', fourcc, 30,
                                           (depth_stream.width, depth_stream.height), False)
        if ir:
            ir_stream = InfraredStream()
            streams.append(ir_stream)
            ir_writer = cv2.VideoWriter(f'{outfile_pattern}_ir.mp4', fourcc, 30, (ir_stream.width, ir_stream.height),
                                        False)
        with serv.Device(device_id=device_id, streams=streams) as cam:
            for _ in range(frames):
                cam.wait_for_frames()
                if rgb_writer:
                    frame = cv2.cvtColor(cam.color, cv2.COLOR_RGB2BGR)
                    rgb_writer.write(frame)
                if depth_writer:
                    frame = cam.dac
                    depth_writer.write(frame)
                if ir_writer:
                    ir_writer.write(cv2.GaussianBlur(cam.infrared, (5, 5), 0))
    for writer in [rgb_writer, depth_writer, ir_writer]:
        if writer:
            writer.release()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device-id', type=int, default=0)
    argparser.add_argument('--rgb', type=bool, default=True)
    argparser.add_argument('--depth', type=bool, default=False)
    argparser.add_argument('--ir', type=bool, default=True)
    argparser.add_argument('--frames', type=int, default=150)
    args = argparser.parse_args()

    save_camera_data_to_files(**vars(args))
