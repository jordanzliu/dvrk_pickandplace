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
import time
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
time.sleep(1)
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

HARDCODED_ECM_POS = np.array([0.0, 0.0, 0.035, 0.0])
# -

tf_listener = tf.TransformListener()

tf_listener.getFrameStrings()

time.sleep(5)
ecm.move_joint(HARDCODED_ECM_POS)

# +
pick_and_place_utils = None
from pick_and_place_utils import get_objects_and_img, tf_to_pykdl_frame, PSM_J1_TO_BASE_LINK_TF, World
import image_geometry

tf_cam_to_jp21 = tf_to_pykdl_frame(tf_listener.lookupTransform('ecm_pitch_link_1', 'camera', rospy.Time()))
tf_jp21_to_world = tf_to_pykdl_frame(tf_listener.lookupTransform('world', 'Jp21_ECM', rospy.Time()))
tf_cam_to_world = tf_jp21_to_world * tf_cam_to_jp21
tf_cam_to_world
# -

tf_world_to_psm2_j1 = tf_to_pykdl_frame(tf_listener.lookupTransform('J1_PSM2', 'world', rospy.Time()))
tf_world_to_psm2_base = PSM_J1_TO_BASE_LINK_TF * tf_world_to_psm2_j1

stereo_cam = image_geometry.StereoCameraModel()
stereo_cam.fromCameraInfo(left_camera_info, right_camera_info)
objects, left_frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, 
                                          cam_to_world_tf=tf_cam_to_world)
world = World(objects)
world

# #### 

# +
from pick_and_place_arm_sm import PickAndPlaceStateMachine

objects_to_pick = deepcopy(world.objects)

approach_vec = PyKDL.Vector(0, 0, -0.02)

print(objects_to_pick)

for obj in objects_to_pick:
    objects, _ = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
    world = World(objects)
    sm = PickAndPlaceStateMachine(psm2, world, tf_world_to_psm2_base, obj, approach_vec)
    
    while not sm.is_done():
        objects, _ = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam)
        world = World(objects)
        sm.update_world(world)
        sm.run_once()
        print("AWDHAWHIOUDAWDAHWUIOPDHUAWDUIOAWDHAWDWDAWD {}".format(sm.is_done()))

# -

