#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import yaml
import rospkg
import time
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
        self.mask_sub = rospy.Subscriber("/image/mask/ground_truth", Image, self.mask_callback, queue_size=1)
        self.coords_pub = rospy.Publisher("/points/prediction", Float64MultiArray, queue_size=1)
        self.inference_ms_pub = rospy.Publisher("/points/inference_ms", Float64, queue_size=1)

    def __del__(self, reason="Shutdown"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown(reason=reason)

    def img_to_msg(self, img: np.array, publisher: rospy.Publisher):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            publisher.publish(img_msg)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

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

    def mask_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding="mono8")
            self.main(frame=frame)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)

    def main(self, frame):
        t_start_s = time.time()
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame=frame,
                                                                                          last_spline_coords=self.last_spline_coords,
                                                                                          params=self.params)
        t_inference_ms = (time.time() - t_start_s) * 1000

        # Publish inference time
        inference_ms_msg = Float64()
        inference_ms_msg.data = t_inference_ms
        self.inference_ms_pub.publish(inference_ms_msg)

        # Publish arrays
        coords_msg = self.generate_2d_array_msg(arr=spline_coords.T)
        self.coords_pub.publish(coords_msg)


if __name__ == "__main__":
    rospy.init_node("cable_observer_node")
    co = CableObserver()
    rospy.spin()
