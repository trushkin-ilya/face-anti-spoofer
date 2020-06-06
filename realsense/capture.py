import pyrealsense as rs
from pyrealsense.stream import ColorStream, DepthStream, InfraredStream
import pyrealsense.constants as rs_constants
import cv2
import numpy as np
import argparse

def save_camera_data_to_files(device_id=0, rgb:bool=True, depth:bool=True, ir:bool=True, frames:int=150, outfile_pattern:str="output"):
    ## start the service - also available as context manager
    serv = rs.Service()
    streams = []
    rgb_writer, depth_writer, ir_writer = None, None, None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if rgb:
        rgb_stream = ColorStream(fps=30, width=1280, height=720)
        streams.append(rgb_stream)
        rgb_writer = cv2.VideoWriter(f'{outfile_pattern}_rgb.mp4', fourcc, 30, (1280, 720))
    if depth:
        depth_stream = DepthStream()
        streams.append(depth_stream)
        depth_writer = cv2.VideoWriter(f'{outfile_pattern}_depth.mp4', fourcc, 30, (depth_stream.width, depth_stream.height))
    if ir:
        ir_stream = InfraredStream(fps=30)
        streams.append(ir_stream)
        ir_writer = cv2.VideoWriter(f'{outfile_pattern}_ir.mp4', fourcc, 30, (ir_stream.width, ir_stream.height), False)
    cam = serv.Device(device_id=device_id, streams=streams)
    cam.set_device_option(rs_constants.rs_option.RS_OPTION_F200_LASER_POWER, 15.0)
    for _ in range(frames):
        cam.wait_for_frames()
        frame = cv2.cvtColor(cam.color, cv2.COLOR_RGB2BGR)
        depth = cam._get_pointcloud()
        extrinsics  = cam.get_device_extrinsics(rs_constants.rs_stream.RS_STREAM_DEPTH, rs_constants.rs_stream.RS_STREAM_COLOR)
        depth_to_color = np.zeros(
            depth_stream.height * depth_stream.width * 3, np.uint8
        )
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        ir = cam.infrared
        if rgb_writer:
            rgb_writer.write(frame)
        if depth_writer:
            depth_writer.write(depth.astype(np.uint8))
        if ir_writer:
            ir_writer.write(ir)
    ## stop realsense and service
    cam.stop()
    serv.stop()
    for writer in [rgb_writer, depth_writer, ir_writer]:
        if writer:
            writer.release()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device-id', type=int, default=0)
    argparser.add_argument('--rgb', type=bool, default=True)
    argparser.add_argument('--depth', type=bool, default=True)
    argparser.add_argument('--ir', type=bool, default=True)
    argparser.add_argument('--frames', type=int, default=150)
    args=argparser.parse_args()

    save_camera_data_to_files(**vars(args))