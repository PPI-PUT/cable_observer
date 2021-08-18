#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import yaml
import rospkg
import time
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64
from cv_bridge import CvBridge, CvBridgeError
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

    def images_callback(self, frame_msg, depth_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg=frame_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(img_msg=depth_msg, desired_encoding="16UC1")
            self.main(frame=frame, depth=depth)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    def main(self, frame, depth):
        t_start_s = time.time()
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame=frame,
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


if __name__ == "__main__":
    rospy.init_node("cable_observer_node")
    co = CableObserver()
    rospy.spin()
