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

left_image = cv2.flip(left_image, 0)
plt.imshow(left_image)

right_image = cv2.flip(right_image, 0)
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

left_feat_pts = [rectify(left_cam, pt) for pt in left_feats.points]
right_feat_pts = [rectify(right_cam, pt) for pt in right_feats.points]
print(left_feat_pts)
print(right_feat_pts)
left_cam.rectifyImage(left_frame, left_frame_rectified)
plt.imshow(left_frame)
# -

right_frame_rectified = deepcopy(right_frame)
left_cam.rectifyImage(right_frame, right_frame_rectified)
plt.imshow(right_frame_rectified)

stereocam = image_geometry.StereoCameraModel()
stereocam.fromCameraInfo(left_camera_info, right_camera_info)
disparity = abs(left_feat_pts[0][0] - right_feat_pts[0][0])
print(disparity)
ball_pos_cam = stereocam.projectPixelTo3d(left_feat_pts[0], disparity)
print(ball_pos_cam)


def tfl_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)


ball_pos_cam = PyKDL.Vector(*ball_pos_cam)


# +
# some hardcoded transforms from suj-simulated.json
ECM_WORLD_ORIGIN_TO_SUJ_ROT_MAT = np.asarray([[ 0.0000,  -1.0000,  0.0000],
                                              [1.0000,  0.0000,  0.0000],
                                              [ 0.0000,  0.0000,  1.0000]])
ECM_WORLD_ORIGIN_TO_SUJ_ROT = PyKDL.Rotation(*ECM_WORLD_ORIGIN_TO_SUJ_ROT_MAT.flatten())
ECM_WORLD_ORIGIN_TO_SUJ_TRANS = PyKDL.Vector(0.0000 , 0.0000, 0.4300)
ECM_WORLD_ORIGIN_TO_SUJ_FRAME = PyKDL.Frame(ECM_WORLD_ORIGIN_TO_SUJ_ROT, ECM_WORLD_ORIGIN_TO_SUJ_TRANS)
ECM_SUJ_TO_WORLD_ORIGIN_FRAME = ECM_WORLD_ORIGIN_TO_SUJ_FRAME.Inverse()

ECM_SUJ_TO_BASE_ROT_MAT = np.asarray([[ 0.0000, 1.0000,  0.0000],
                                      [ -1.0000,  0.0000,  0.0000],
                                      [ 0.0000,  0.0000,  1.0000]])
ECM_SUJ_TO_BASE_ROT = PyKDL.Rotation(*ECM_BASE_TO_SUJ_ROT_MAT.flatten())
ECM_SUJ_TO_BASE_TRANS = PyKDL.Vector(0.6126, 0.0000,   0.1016)
ECM_SUJ_TO_BASE_FRAME = PyKDL.Frame(ECM_BASE_TO_SUJ_ROT, ECM_BASE_TO_SUJ_TRANS)
ECM_BASE_TO_SUJ_FRAME = ECM_SUJ_TO_BASE_FRAME.Inverse()

ECM_BASE_TO_WORLD_FRAME = 

PSM1_WORLD_ORIGIN_TO_SUJ_ROT_MAT = np.asarray([[-1.0000,  0.0000,  0.0000],
                                               [ 0.0000, -1.0000,  0.0000],
                                               [ 0.0000,  0.0000,  1.0000]])
PSM1_WORLD_ORIGIN_TO_SUJ_ROT = PyKDL.Rotation(*PSM1_WORLD_ORIGIN_TO_SUJ_ROT_MAT.flatten())
PSM1_WORLD_ORIGIN_TO_SUJ_TRANS = PyKDL.Vector(0.4864, 0.0000, 0.1524)
PSM1_WORLD_ORIGIN_TO_SUJ_FRAME = PyKDL.Frame(PSM1_WORLD_ORIGIN_TO_SUJ_ROT, PSM1_WORLD_ORIGIN_TO_SUJ_TRANS)

PSM1_SUJ_TO_TOOL_ROT_MAT = np.asarray([[ 0.0000, 1.0000,  0.0000],
                                      [ -1.0000,  0.0000,  0.0000],
                                      [ 0.0000,  0.0000,  1.0000]])
PSM1_SUJ_TO_TOOL_ROT = PyKDL.Rotation(*PSM1_SUJ_TO_TOOL_ROT_MAT.flatten())
PSM1_SUJ_TO_TOOL_TRANS = PyKDL.Vector(0.4864, 0.0000, 0.1524)
PSM1_SUJ_TO_TOOL_FRAME = PyKDL.Frame(PSM1_SUJ_TO_TOOL_ROT, PSM1_SUJ_TO_TOOL_TRANS)

PSM1_WORLD_TO_TOOL_FRAME = PSM1_SUJ_TO_TOOL_FRAME * PSM1_WORLD_ORIGIN_TO_SUJ_FRAME
PSM1_TOOL_TO_WORLD_FRAME = PSM1_WORLD_TO_TOOL_FRAME.Inverse()
# -

tfl_cam_ecm_setup = tf_listener.lookupTransform('ecm_setup_link', 'camera', rospy.Time())
tf_cam_ecm_setup = tfl_to_pykdl_frame(tfl_cam_ecm_setup)
tf_cam_world = ECM_SUJ_TO_WORLD_ORIGIN_FRAME * tf_cam_ecm_setup
ball_pos_world = tf_cam_world * ball_pos_cam
ball_pos_ecm_setup_link = tf_cam_ecm_setup_base * ball_pos_cam
ball_pos_ecm_setup_link # this is sane, i hope the SUJ transform for the 


