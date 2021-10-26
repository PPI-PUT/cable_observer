import time
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Float64
from cv_bridge import CvBridge, CvBridgeError
from cable_observer.utils.tracking import track

qos_profile = QoSProfile(depth=10)
#qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT


class CableObserver(Node):
    def __init__(self):
        super().__init__('cable_observer_node')
        stream = open(get_package_share_directory('cable_observer') + "/config/params.yaml", 'r')
        self.params = yaml.load(stream, Loader=yaml.FullLoader)
        self.bridge = CvBridge()
        self.last_spline_coords = None
        self.mask_sub = self.create_subscription(Image, "/camera/color/image_raw", self.mask_callback, qos_profile=qos_profile)
        self.coords_pub = self.create_publisher(Float64MultiArray, "/points/prediction", qos_profile=qos_profile)
        self.inference_ms_pub = self.create_publisher(Float64, "/points/inference_ms", qos_profile=qos_profile)
        self.spline_img_pub = self.create_publisher(Image, "/points/image", qos_profile=qos_profile)
        time.sleep(3)

    def img_to_msg(self, img: np.array, publisher: rclpy.publisher.Publisher):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            img_msg.header.frame_id = "camera_link"
            publisher.publish(img_msg)
        except (CvBridgeError, TypeError) as e:
            self.get_logger().warn(e)

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
            frame = self.bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding="bgr8")
            self.main(frame=frame)
        except (CvBridgeError, TypeError) as e:
            self.get_logger().warn(e)

    def main(self, frame):
        t_start_s = time.time()
        spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t, img_spline = track(frame=frame,
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

        # Publish spline image
        self.img_to_msg(img=img_spline, publisher=self.spline_img_pub)


def main(args=None):
    rclpy.init(args=args)
    cable_observer = CableObserver()
    rclpy.spin(cable_observer)
    cable_observer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
