#!/usr/bin/env python3

# Copyright 2023 Perception for Physical Interaction Laboratory at Poznan University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numpy.typing as npt
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from visualization_msgs.msg import Marker

try:
    from cable_observer.cable_observer import CableObserver
except Exception:
    from cable_observer import CableObserver


class CableObserverNode(Node):
    def __init__(self):
        super().__init__('cable_observer_node')
        self._cable_observer = CableObserver()
        self._frame_id = ''
        self._cable_observer.set_parameters(
            debug=self.declare_parameter('debug', False).value,
            hsv_ranges=self.declare_parameter('hsv_ranges', [0, 0, 0, 179, 255, 255]).value,
            depth_ranges=self.declare_parameter('depth_ranges', [0, 10000]).value,
            depth_scale=self.declare_parameter('depth_scale', 1.0).value,
            min_length=self.declare_parameter('min_length', 10).value,
            num_of_knots=self.declare_parameter('num_of_knots', 25).value,
            num_of_pts=self.declare_parameter('num_of_pts', 256).value,
            vector_dir_len=self.declare_parameter('vector_dir_len', 5).value,
            z_vertical_shift=self.declare_parameter('z_vertical_shift', 0).value,
        )

        self._bridge = CvBridge()
        self.create_subscription(CameraInfo, '/rgb/camera_info', self.camera_info_callback, 10)
        self._rgb_sub = Subscriber(self, Image, '/rgb/image_raw')
        self._depth_sub = Subscriber(self, Image, '/depth_to_rgb/image_raw')
        self._tss = ApproximateTimeSynchronizer([self._rgb_sub, self._depth_sub], 30, 0.1)
        self._tss.registerCallback(self.images_callback)
        self._projection_mat = np.zeros(shape=(3, 2), dtype=np.float64)
        self._marker_pub = self.create_publisher(Marker, 'marker', 10)
        self._cloud_pub = self.create_publisher(PointCloud2, 'cloud', 10)

    def camera_info_callback(self, camera_info_msg: CameraInfo) -> None:
        self._projection_mat[0, 0] = camera_info_msg.p[0]  # fx
        self._projection_mat[1, 0] = camera_info_msg.p[5]  # fy
        self._projection_mat[2, 0] = 1.0
        self._projection_mat[0, 1] = camera_info_msg.p[2]  # cx
        self._projection_mat[1, 1] = camera_info_msg.p[6]  # cy

    def images_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        self._frame_id = rgb_msg.header.frame_id
        rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='64FC1')
        spline_coords = self._cable_observer.track(frame=rgb[..., :3], depth=depth)

        points_3d = self.coords_to_points_3d(spline_coords.T)

        # Publish marker
        marker_msg = self.generate_marker_msg(arr=np.array(
            [points_3d.T[0], points_3d.T[1], points_3d.T[2]]))
        self._marker_pub.publish(marker_msg)

        # Publish point cloud
        cloud_msg = create_cloud_xyz32(rgb_msg.header, points_3d)
        self._cloud_pub.publish(cloud_msg)

    def coords_to_points_3d(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = points[:, 2]
        z = np.stack([z, z, np.ones_like(z)], axis=-1)
        points_3d = z * (points - self._projection_mat[..., 1]) / self._projection_mat[..., 0]

        return points_3d

    def generate_marker_msg(self, arr: npt.NDArray[np.float64]) -> Marker:
        marker_msg = Marker()
        marker_msg.header.frame_id = self._frame_id
        marker_msg.type = marker_msg.LINE_STRIP
        marker_msg.action = marker_msg.ADD

        marker_msg.scale.x = 0.01
        marker_msg.scale.y = 0.01
        marker_msg.scale.z = 0.01

        marker_msg.color.a = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 0.0
        marker_msg.color.b = 0.0

        marker_msg.pose.orientation.w = 1.0

        marker_msg.points = [Point(x=point[0], y=point[1], z=point[2]) for point in arr.T]

        return marker_msg


def main(args=None):
    rclpy.init(args=args)
    node = CableObserverNode()
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
