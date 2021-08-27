#!/usr/bin/env python3
import rospy
import rospkg
import cv2
import argparse
from pathlib import Path
import rosbag
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from glob import glob
from tqdm import tqdm
import os

bridge = CvBridge()
rgb_topic = '/camera/color/image_raw'
depth_topic = '/camera/aligned_depth_to_color/image_raw'
info_topic = '/camera/aligned_depth_to_color/camera_info'
rgb_pub = rospy.Publisher(rgb_topic, Image, queue_size=10)
depth_pub = rospy.Publisher(depth_topic, Image, queue_size=10)
info_pub = rospy.Publisher(info_topic, CameraInfo, queue_size=10)
path = rospkg.RosPack().get_path("cable_observer")
path_images = os.path.join(path, "kinect_data")
path_bag = os.path.join(path, "kinect_bags")
Path(path_bag).mkdir(parents=True, exist_ok=True)
rospy.init_node('camera2bag_node')

parser = argparse.ArgumentParser(description='Images to rosbag file.')
parser.add_argument('-i',
                    '--idx',
                    type=int,
                    default=0,
                    help='Bag file index')
args = parser.parse_args()

bag_name = 'kinect_{}.bag'.format(str(args.idx).rjust(3, "0"))
bag = rosbag.Bag(os.path.join(path_bag, bag_name), 'w')

for i, (d_path, rgb_path) in enumerate(tqdm(zip(sorted(glob(os.path.join(path_images, "depth_*"))),
                                           sorted(glob(os.path.join(path_images, "color_*")))))):
    d = cv2.imread(d_path, -1)
    rgb = cv2.imread(rgb_path)
    rgb_message = bridge.cv2_to_imgmsg(rgb, "rgb8")
    depth_message = bridge.cv2_to_imgmsg(d, encoding="passthrough")
    camera_info_msg = CameraInfo()
    camera_info_msg.K = [915.3726806640625, 0, 650.8181762695312, 0, 914.5042724609375, 361.7558288574219, 0, 0, 1]
    t = rospy.get_rostime()
    rgb_message.header.stamp = t
    depth_message.header.stamp = t
    camera_info_msg.header.stamp = t
    rgb_message.header.seq = i
    depth_message.header.seq = i
    camera_info_msg.header.seq = i
    frame_id = "depth_optical_frame"
    rgb_message.header.frame_id = frame_id
    depth_message.header.frame_id = frame_id
    camera_info_msg.header.frame_id = frame_id
    rgb_pub.publish(rgb_message)
    depth_pub.publish(depth_message)
    info_pub.publish(camera_info_msg)
    bag.write(rgb_topic, rgb_message)
    bag.write(depth_topic, depth_message)
    bag.write(info_topic, camera_info_msg)

bag.close()
print("Bag saved to {}".format(bag_name))
