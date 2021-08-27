#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_viewer.py

import os
import argparse
from copy import deepcopy
from pathlib import Path
import cv2
import open3d as o3d
import numpy as np
import rospkg
from time import sleep
from tqdm import tqdm

fps = 30


class ViewerWithCallback:
    def __init__(self, config, device, align_depth_to_color):
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color
        self.rgbs = []
        self.depths = []
        self.sensor = o3d.io.AzureKinectSensor(config)
        self.fps = fps
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def run(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        vis_geometry_added = False
        try:
            for i in tqdm(range(self.fps * args.seconds)):
                rgbd = self.sensor.capture_frame(self.align_depth_to_color)
                if rgbd is None:
                    continue

                color_np = np.asarray(rgbd.color)
                depth_np = np.asarray(rgbd.depth)
                self.rgbs.append(deepcopy(color_np))
                self.depths.append(deepcopy(depth_np))

                if not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True

                vis.update_geometry(rgbd)
                vis.poll_events()
                vis.update_renderer()
        finally:
            for i in range(len(self.rgbs)):
                cv2.imwrite(f"kinect_data/depth_{i:03}.png", self.depths[i])
                cv2.imwrite(f"kinect_data/color_{i:03}.png", self.rgbs[i])
            print("Images saved to \"kinect_data\" directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect rgb & depth collector.')
    parser.add_argument('--config', type=str, default="default_config.json", help='input json kinect config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-s',
                        '--seconds',
                        type=int,
                        default=10,
                        help='Record duration')
    parser.add_argument('-d',
                        '--delay',
                        type=int,
                        default=5,
                        help='Delay before recording')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    path = rospkg.RosPack().get_path("cable_observer")
    Path(os.path.join(path, "kinect_data")).mkdir(parents=True, exist_ok=True)

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    for i in tqdm(range(args.delay)):
        sleep(1)
    v = ViewerWithCallback(config, device, args.align_depth_to_color)
    v.run()
