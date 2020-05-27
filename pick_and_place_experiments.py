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
    psm2 = dvrk.psm('PSM2')

# -

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
pick_and_place_utils = None
from pick_and_place_utils import get_feat_position_and_img, tf_to_pykdl_frame, PSM_J1_TO_MAIN_TF

stereo_cam = image_geometry.StereoCameraModel()
stereo_cam.fromCameraInfo(left_camera_info, right_camera_info)
ball_pos_cam, left_frame = get_feat_position_and_img(left_image_msg, right_image_msg, stereo_cam)
print(ball_pos_cam)
plt.imshow(left_frame)
# -

ball_pos_cam = PyKDL.Vector(*ball_pos_cam)
tf_cam_to_jp21 = tf_to_pykdl_frame(tf_listener.lookupTransform('ecm_pitch_link_1', 'camera', rospy.Time()))
ball_pos_jp21 = tf_cam_to_jp21 * ball_pos_cam
ball_pos_jp21

# did this to confirm that the /tf transforms provided by dVRK are within margin of error of 
# actual sim coordinates
tf_insertion_to_jp21 = tf_listener.lookupTransform('ecm_pitch_link_1', 'ecm_insertion_link', rospy.Time())
tf_insertion_to_jp21

tf_jp21_to_world = tf_to_pykdl_frame(tf_listener.lookupTransform('world', 'Jp21_ECM', rospy.Time()))
ball_pos_world = tf_jp21_to_world * ball_pos_jp21
# i'm a winner
ball_pos_world

tf_world_to_psm1_j1 = tf_to_pykdl_frame(tf_listener.lookupTransform('J1_PSM1', 'world', rospy.Time()))
ball_pos_psm1_j1 = tf_world_to_psm1_j1 * ball_pos_world
# ok not off by *too* much
ball_pos_psm1_j1

# ok lets try something else 
from math import pi
j1_to_main_rot = PyKDL.Rotation.RPY(pi / 2, - pi, 0)
j1_to_main_frame = PyKDL.Frame(j1_to_main_rot, PyKDL.Vector())
ball_pos_psm1_main = j1_to_main_frame * ball_pos_psm1_j1
ball_pos_psm1_main

# #### 

psm1_rot = psm1.get_current_position().M
psm1_pos = psm1.get_current_position().p
psm1_pos

psm1.move(PyKDL.Frame(PyKDL.Rotation.Identity(), ball_pos_psm1_main))


