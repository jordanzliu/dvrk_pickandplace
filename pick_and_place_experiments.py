# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

import jupyros as jr
import rospy
import numpy as np
from sensor_msgs import msg
import cv2
import cv_bridge
from copy import deepcopy
import ipywidgets as widgets
from IPython.display import clear_output, Image, display
import PIL.Image
from cStringIO import StringIO
import matplotlib.pyplot as plt
import dvrk
import PyKDL
import tf
from tf_conversions import posemath

rospy.init_node('notebook')

rospy.get_published_topics()

# +
bridge = cv_bridge.CvBridge()
left_image = None
left_image_msg = None
left_camera_info = None

right_image = None
right_image_msg = None
right_camera_info = None

def left_image_callback(im_msg):
    global left_image, left_image_msg
    left_image = bridge.imgmsg_to_cv2(im_msg, desired_encoding='rgb8')
    left_image_msg = im_msg
    
def right_image_callback(im_msg):
    global right_image, right_image_msg
    right_image = bridge.imgmsg_to_cv2(im_msg, desired_encoding='rgb8')
    right_image_msg = im_msg
    
def left_camera_info_callback(camera_info_msg):
    global left_camera_info
    left_camera_info = camera_info_msg
    
def right_camera_info_callback(camera_info_msg):
    global right_camera_info
    right_camera_info = camera_info_msg
    
jr.subscribe('/stereo/left/image_flipped', msg.Image, left_image_callback)
jr.subscribe('/stereo/left/camera_info', msg.CameraInfo, left_camera_info_callback)
jr.subscribe('/stereo/right/image_flipped', msg.Image, right_image_callback)
jr.subscribe('/stereo/right/camera_info', msg.CameraInfo, right_camera_info_callback)
# -

plt.imshow(left_image)

plt.imshow(right_image)

print("LEFT CAM")
print(left_camera_info)
print("RIGHT_CAM")
print(right_camera_info)

# +
psm1 = None 
ecm = None
suj = None
debug_output = widgets.Output(layout={'border': '1px solid black'})

with debug_output:
    global psm1, ecm
    psm1 = dvrk.psm('PSM1')
    ecm = dvrk.ecm('ECM')
    suj = dvrk.suj('ECM')

# -

psm1.get_current_position()

ecm.get_current_position()

suj.get_current_position()

tf_listener = tf.TransformListener()

tf_listener.getFrameStrings()

import image_geometry
import vision_pipeline
BALL_FEAT_PATH = '../autonomous_surgical_camera/auto_cam/config/features/red_ball.csv'
cv2.imwrite('left.png', left_image)
fp = vision_pipeline.feature_processor([BALL_FEAT_PATH], 'left.png')
left_feats, left_frame = fp.Centroids(left_image_msg)
right_feats, right_frame = fp.Centroids(right_image_msg)
print(left_feats)
plt.imshow(left_frame)

print(right_feats)
plt.imshow(right_frame)

# +
left_cam = image_geometry.PinholeCameraModel()
left_cam.fromCameraInfo(left_camera_info)
right_cam = image_geometry.PinholeCameraModel()
right_cam.fromCameraInfo(right_camera_info)
left_frame_rectified = deepcopy(left_frame)

def rectify(cam, ros_pt):
    return tuple(cam.rectifyPoint((ros_pt.x, ros_pt.y)))

def invert_rectify(cam, ros_pt, frame_dims):
    return tuple(cam.rectifyPoint((frame_dims[0] - ros_pt.x, frame_dims[1] - ros_pt.y)))

left_feat_pts = [(pt.x, pt.y) for pt in left_feats.points]
right_feat_pts = [(pt.x, pt.y)for pt in right_feats.points]
print(left_feat_pts)
print(right_feat_pts)
left_cam.rectifyImage(left_frame, left_frame_rectified)
plt.imshow(left_frame)
# -

right_frame_rectified = deepcopy(right_frame)
left_cam.rectifyImage(right_frame, right_frame_rectified)
plt.imshow(right_frame_rectified)

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10000)


def publish_marker(point, frame, marker_id):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.pose.position.x = point.x()
    marker.pose.position.y = point.y()
    marker.pose.position.z = point.z()
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.lifetime = rospy.Time.from_sec(10000)
    marker_pub.publish(marker)


stereocam = image_geometry.StereoCameraModel()
stereocam.fromCameraInfo(left_camera_info, right_camera_info)
disparity = abs(left_feat_pts[0][0] - right_feat_pts[0][0])
print(disparity)
ball_pos_cam = stereocam.projectPixelTo3d(left_feat_pts[0], disparity)
# opencv coordinates are left handed and ROS coordinates are right handed!!!!!!!! WTF
ball_pos_cam = (ball_pos_cam[1], - ball_pos_cam[0], ball_pos_cam[2])
print(ball_pos_cam)


def tfl_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)


ball_pos_cam = PyKDL.Vector(*ball_pos_cam)
print(ball_pos_cam)
publish_marker(PyKDL.Vector(0, 0, 0), 'world', 1)


# +
tf_cam_to_pitch_link = tf_listener.lookupTransform('ecm_pitch_link', 'camera', rospy.Time())
tf_cam_to_pitch_link = tfl_to_pykdl_frame(tf_cam_to_pitch_link)
tf_pitch_link_to_world = tf_listener.lookupTransform('simworld', 'Jp21_ECM', rospy.Time())
tf_pitch_link_to_world = tfl_to_pykdl_frame(tf_pitch_link_to_world)
tf_cam_to_world = tf_pitch_link_to_world * tf_cam_to_pitch_link

# straight up broadcasted the vision sensor frame
tf_cam_to_world = tf_listener.lookupTransform('simworld', 'Vision_sensor_left', rospy.Time())
tf_cam_to_world = tfl_to_pykdl_frame(tf_cam_to_world)

# +
# there's a hardcoded rotation between J1_PSM1 in sim and PSM1_psm_main
j1_to_main_rot = PyKDL.Rotation(
    PyKDL.Vector(-1,  0,  0),
    PyKDL.Vector( 0,  0, -1),
    PyKDL.Vector( 0, -1,  0)
)
j1_to_main_trans = PyKDL.Vector(0, 0, 0)
j1_to_main_frame = PyKDL.Frame(j1_to_main_rot, j1_to_main_trans)

tf_world_to_psm1_j1 = tf_listener.lookupTransform('J1_PSM1', 'simworld', rospy.Time())
tf_world_to_psm1_j1 = tfl_to_pykdl_frame(tf_world_to_psm1_j1)
tf_world_to_psm1_main = j1_to_main_frame * tf_world_to_psm1_j1
tf_world_to_psm1_main
# -

tf_camera_to_psm1 = tf_world_to_psm1_main * tf_cam_to_world
ball_pos_psm1 = tf_camera_to_psm1 * ball_pos_cam
ball_pos_psm1

ball_pos_world = tf_cam_to_world * ball_pos_cam
ball_pos_world
ball_pos_j1_psm1 = tf_world_to_psm1_j1 * ball_pos_world
print(ball_pos_j1_psm1)


ball_pos_psm1_main = tf_world_to_psm1_main * ball_pos_world
print(ball_pos_psm1_main)

# this is the position piped directly from the sim
# slight inaccuracy compared to `ball_pos_psm1` but numbers are very close
real_ball_pos_psm1_j1 = PyKDL.Vector(-0.1254749298, 0.2540073991, 0.01814687252)
real_ball_pos_psm1_main = j1_to_main_frame * real_ball_pos_psm1_j1
psm1.move(real_ball_pos_psm1_main)
real_ball_pos_psm1_main

psm1.move(real_ball_pos_psm1_main)


