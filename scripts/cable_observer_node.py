#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import yaml
import rospkg
import time
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64
from cv_bridge import CvBridge, CvBridgeError
from ros_numpy import point_cloud2
from cable_observer.utils.tracking import track


class CableObserver:
    def __init__(self):
        rospack = rospkg.RosPack()
        stream = open(rospack.get_path('cable_observer') + "/config/params.yaml", 'r')
        self.params = yaml.load(stream, Loader=yaml.FullLoader)
        self.bridge = CvBridge()
        self.last_spline_coords = None
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], queue_size=1)
        self.ts.registerCallback(self.images_callback)
        self.coords_pub = rospy.Publisher("/points/prediction", Float64MultiArray, queue_size=1)
        self.inference_ms_pub = rospy.Publisher("/points/inference_ms", Float64, queue_size=1)
        self.marker_pub = rospy.Publisher("/points/marker", Marker, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/image_depth", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        self.pc_pub = rospy.Publisher("/camera/depth/points", PointCloud2, queue_size=1)

    def __del__(self, reason="Shutdown"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown(reason=reason)

    @staticmethod
    def generate_2d_array_msg(arr):
        arr_msg = Float64MultiArray()
        arr_msg.data = np.hstack(arr).tolist()
        arr_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

        arr_msg.layout.dim[0].label = "channels"
        arr_msg.layout.dim[0].size = arr.shape[0]  # channels
        arr_msg.layout.dim[0].stride = arr.size  # channels * samples

        arr_msg.layout.dim[1].label = "samples"
        arr_msg.layout.dim[1].size = arr.shape[1]  # samples
        arr_msg.layout.dim[1].stride = arr.shape[1]  # samples

        return arr_msg

    @staticmethod
    def generate_marker_msg(arr):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = "camera_depth_frame"
        marker_msg.type = marker_msg.POINTS
        marker_msg.action = marker_msg.ADD

        marker_msg.scale.x = 1
        marker_msg.scale.y = 1
        marker_msg.scale.z = 1

        marker_msg.color.a = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 1.0
        marker_msg.color.b = 0.0

        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.pose.position.x = 0.0
        marker_msg.pose.position.y = 0.0
        marker_msg.pose.position.z = 0.0

        marker_msg.points = []

        for sample in arr.T:
            point = Point()
            point.x = sample[0]
            point.y = sample[1]
            point.z = sample[2]
            marker_msg.points.append(point)

        return marker_msg

    def generate_depth_msg(self, mask_depth, depth):
        try:
            depth[np.where(mask_depth == 0)] = 0
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            depth_msg.header.stamp = rospy.Time.now()
            depth_msg.header.frame_id = "kinect2_rgb_optical_frame"
            pc_msg = point_cloud2.array_to_pointcloud2(depth.astype([('x', 'f4'), ('y', 'f4'), ('z', 'f4')]))
            pc_msg.header = depth_msg.header
            return depth_msg, pc_msg
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    @staticmethod
    def generate_camera_info_msg(depth_msg):
        info_msg = CameraInfo()
        info_msg.header = depth_msg.header
        return info_msg

    def images_callback(self, frame_msg, depth_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg=frame_msg, desired_encoding="8UC3")
            depth = self.bridge.imgmsg_to_cv2(img_msg=depth_msg, desired_encoding="16UC1")
            self.main(frame=frame, depth=depth)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    def main(self, frame, depth):
        t_start_s = time.time()
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t, mask_depth = \
            track(frame=frame,
                  depth=depth,
                  last_spline_coords=self.last_spline_coords,
                  params=self.params)
        t_inference_ms = (time.time() - t_start_s) * 1000

        # Publish inference time
        inference_ms_msg = Float64()
        inference_ms_msg.data = t_inference_ms
        self.inference_ms_pub.publish(inference_ms_msg)

        # Publish arrays
        coords_msg = self.generate_2d_array_msg(arr=np.array([spline_coords.T[1], spline_coords.T[0], spline_coords.T[2]]))
        self.coords_pub.publish(coords_msg)

        # Publish marker
        marker_msg = self.generate_marker_msg(arr=np.array([spline_coords.T[1], spline_coords.T[0], spline_coords.T[2]]))
        self.marker_pub.publish(marker_msg)

        # Publish depth & camera info
        depth_msg, pc_msg = self.generate_depth_msg(mask_depth=mask_depth, depth=depth)
        info_msg = self.generate_camera_info_msg(depth_msg=depth_msg)
        self.depth_pub.publish(depth_msg)
        self.pc_pub.publish(pc_msg)
        self.camera_info_pub.publish(info_msg)


if __name__ == "__main__":
    rospy.init_node("cable_observer_node")
    co = CableObserver()
    rospy.spin()
