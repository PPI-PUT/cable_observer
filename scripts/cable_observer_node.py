#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
from cable_observer.utils.tracking import track
from cable_observer.utils.image_processing import get_spline_image
from cable_observer.utils.debug_frame_processing import DebugFrameProcessing


class CableObserver:
    def __init__(self):
        self.debug = rospy.get_param("/debug")
        self.path = rospy.get_param("/path")
        self.between = rospy.get_param("/between")
        self.camera = rospy.get_param("/camera")
        self.input = rospy.get_param("/input")
        self.bridge = CvBridge()
        if self.camera:
            self.cap = cv2.VideoCapture(self.input)
        else:
            self.cap = cv2.VideoCapture(self.path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.skip_blank_frames()
        self.poc = []
        self.cps = []
        self.last_spline_coords = None
        self.image_raw_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=1)
        self.image_spline_pub = rospy.Publisher("/camera/image_spline", Image, queue_size=1)
        self.coeffs_pub = rospy.Publisher("/spline/coeffs", Float64MultiArray, queue_size=1)
        self.coords_pub = rospy.Publisher("/spline/coords", Float64MultiArray, queue_size=1)
        if self.debug:
            self.debug_image_merged_pub = rospy.Publisher("/debug/image_merged", Image, queue_size=1)
            self.debug_image_spline_pub = rospy.Publisher("/debug/image_spline", Image, queue_size=1)
            self.debug_image_prediction_pub = rospy.Publisher("/debug/image_prediction", Image, queue_size=1)
            self.debug_image_mask_pub = rospy.Publisher("/debug/image_mask", Image, queue_size=1)
            self.debug_image_skeleton_pub = rospy.Publisher("/debug/image_skeleton", Image, queue_size=1)

    def __del__(self, reason="Shutdown"):
        self.cap.release()
        cv2.destroyAllWindows()
        rospy.signal_shutdown(reason=reason)

    def skip_blank_frames(self):
        for i in range(100):
            _, _ = self.cap.read()

    def img_to_msg(self, img: np.array, publisher: rospy.Publisher):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            publisher.publish(img_msg)
        except (CvBridgeError, TypeError) as e:
            rospy.logwarn(e)
            self.__del__(reason="Exception")

    def publish_array(self, arr: np.array, publisher: rospy.Publisher):
        arr_msg = Float64MultiArray()
        arr_msg.data = np.hstack([arr[0] / self.height, arr[1] / self.width]).tolist()

        dim_v = MultiArrayDimension()
        dim_v.label = "v"
        dim_v.size = len(arr[0])
        dim_v.stride = len(arr[0]) * 2

        dim_u = MultiArrayDimension()
        dim_u.label = "u"
        dim_u.size = len(arr[1])
        dim_u.stride = len(arr)

        dims = [dim_v, dim_u]
        arr_msg.layout.dim = dims

        publisher.publish(arr_msg)

    def main(self, event):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        self.img_to_msg(img=frame, publisher=self.image_raw_pub)

        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame, self.last_spline_coords,
                                                                                          between_grippers=self.between)
        spline_img = get_spline_image(spline_coords=spline_coords, shape=frame.shape)
        self.img_to_msg(img=np.uint8(spline_img*255), publisher=self.image_spline_pub)

        self.publish_array(arr=spline_params["coeffs"], publisher=self.coeffs_pub)
        self.publish_array(arr=spline_coords.T, publisher=self.coords_pub)

        if self.debug:
            dfp = DebugFrameProcessing(frame, self.cps, self.poc, self.last_spline_coords,
                                       spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t)
            self.cps, self.poc, self.last_spline_coords = dfp.get_params()
            dfp.print_t()
            self.img_to_msg(img=dfp.img_frame, publisher=self.debug_image_merged_pub)
            self.img_to_msg(img=dfp.img_spline, publisher=self.debug_image_spline_pub)
            self.img_to_msg(img=dfp.img_pred, publisher=self.debug_image_prediction_pub)
            self.img_to_msg(img=dfp.img_mask, publisher=self.debug_image_mask_pub)
            self.img_to_msg(img=dfp.img_skeleton, publisher=self.debug_image_skeleton_pub)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.__del__(reason="Received \"q\" key")


if __name__ == "__main__":
    rospy.init_node("video_node")
    co = CableObserver()
    rospy.Timer(rospy.Duration(1.0 / 30.0), co.main)
    rospy.spin()
