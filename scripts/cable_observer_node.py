#!/usr/bin/env python3
import os
import rospy
import numpy as np
import pandas as pd
import yaml
import rospkg
import time
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64
from cv_bridge import CvBridge, CvBridgeError
from ros_numpy import point_cloud2
from cable_observer.utils.tracking import track
from std_srvs.srv import Empty, EmptyResponse


class CableObserver:
    def __init__(self):
        ns = rospy.get_namespace()
        rospack = rospkg.RosPack()
        stream = open(rospack.get_path('cable_observer') + "/config/params.yaml", 'r')
        self.frame_id = rospy.get_param("frame_id")
        self.visualize = rospy.get_param("visualize")
        self.csv_path = os.path.join(rospack.get_path('cable_observer'), "spline/spline.csv")
        self.params = yaml.load(stream, Loader=yaml.FullLoader)
        self.bridge = CvBridge()
        self.last_spline_coords = None
        self.df = pd.DataFrame()
        self.df_index = 0
        self.srv = rospy.Service(ns + "save_df", Empty, self.handle_save_df)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.images_callback)
        self.coords_pub = rospy.Publisher(ns + "points/prediction", Float64MultiArray, queue_size=1)
        self.inference_ms_pub = rospy.Publisher(ns + "points/inference_ms", Float64, queue_size=1)
        self.marker_pub = rospy.Publisher(ns + "points/marker", Marker, queue_size=1)
        if self.visualize:
            self.depth_pub = rospy.Publisher(ns + "camera/filtered/depth/image_raw", Image, queue_size=1)
            self.pc_pub = rospy.Publisher(ns + "camera/filtered/depth/points", PointCloud2, queue_size=1)

    def __del__(self, reason="Shutdown"):
        rospy.signal_shutdown(reason=reason)

    def handle_save_df(self, req):
        self.df.to_csv(self.csv_path)
        rospy.loginfo("Dataframe saved")
        return EmptyResponse()

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

    def generate_marker_msg(self, arr):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = self.frame_id
        marker_msg.type = marker_msg.LINE_STRIP
        marker_msg.action = marker_msg.ADD

        marker_msg.scale.x = 1
        marker_msg.scale.y = 1
        marker_msg.scale.z = 1

        marker_msg.color.a = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 1.0
        marker_msg.color.b = 0.0

        marker_msg.pose.orientation.w = 1.0

        marker_msg.points = [Point(x=point[0], y=point[1], z=point[2]) for point in arr.T]

        return marker_msg

    def generate_depth_pc2_msgs(self, mask_depth, depth):
        try:
            depth[np.where(mask_depth == 0)] = 0
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            depth_msg.header.stamp = rospy.Time.now()
            depth_msg.header.frame_id = self.frame_id
            pc_arr = np.array([np.where(depth)[1], np.where(depth)[0], depth[np.where(depth)]])
            output_dtype = np.dtype(
                {'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4']})
            new_points = np.core.records.fromarrays(pc_arr, output_dtype)
            pc_msg = point_cloud2.array_to_pointcloud2(new_points,
                                                       stamp=depth_msg.header.stamp,
                                                       frame_id=depth_msg.header.frame_id)
            return depth_msg, pc_msg
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    def update_dataframe(self, spline_params, spline_coords):
        # Generate dataframe sample for current spline
        spline_metadata = pd.DataFrame({"control_points_x": [[el for el in spline_params['coeffs'][1]], ],
                                        "control_points_y": [[el for el in spline_params['coeffs'][0]], ],
                                        "control_points_z": [[el for el in spline_params['coeffs'][2]], ],
                                        "points_on_curve_x": [[el for el in spline_coords[:, 1]], ],
                                        "points_on_curve_y": [[el for el in spline_coords[:, 0]], ],
                                        "points_on_curve_z": [[el for el in spline_coords[:, 2]], ],
                                        }, index=[0])
        spline_metadata.index = [self.df_index]
        self.df = self.df.append(spline_metadata)
        self.df_index += 1

    def images_callback(self, depth_msg):
        try:
            #frame = self.bridge.imgmsg_to_cv2(img_msg=frame_msg, desired_encoding="8UC3")
            depth = self.bridge.imgmsg_to_cv2(img_msg=depth_msg, desired_encoding="16UC1")
            self.main(depth=depth)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    def main(self, depth):
        t_start_s = time.time()
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t, mask_depth = \
            track(frame=None,
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

        # Publish depth image & pointcloud
        if self.visualize:
            depth_msg, pc_msg = self.generate_depth_pc2_msgs(mask_depth=mask_depth, depth=depth)
            self.depth_pub.publish(depth_msg)
            self.pc_pub.publish(pc_msg)

        # Update dataframe
        self.update_dataframe(spline_params=spline_params, spline_coords=spline_coords)


if __name__ == "__main__":
    rospy.init_node("cable_observer_node")
    co = CableObserver()
    rospy.spin()
