import rospy
import numpy as np
from sensor_msgs import msg
import cv2
import cv_bridge
from copy import deepcopy
import dvrk
import PyKDL
import tf
import image_geometry
import vision_pipeline
from tf_conversions import posemath

'''
Some useful methods and 
'''

J1_TO_MAIN_ROT = PyKDL.Rotation(
    PyKDL.Vector(-1,  0,  0),
    PyKDL.Vector( 0,  0, -1),
    PyKDL.Vector( 0, -1,  0)
)

J1_TO_MAIN_TRANS = PyKDL.Vector(0, 0, 0)

J1_TO_PSM_MAIN_FRAME = PyKDL.Frame(J1_TO_MAIN_ROT, J1_TO_MAIN_TRANS)

# TODO: make this less hardcoded
RED_BALL_FEAT_PATH = '../autonomous_surgical_camera/auto_cam/config/features/red_ball.csv'

CV_TO_CAM_FRAME_ROT = np.asarray([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

def get_feat_position_and_img(left_image_msg, right_image_msg, stereo_cam_model):
    # this gets the position of the red ball thing in the camera frame
    # and the image with X's on the desired features
    fp = vision_pipeline.feature_processor([RED_BALL_FEAT_PATH], 'left.png')
    left_feats, left_frame = fp.Centroids(left_image_msg)
    right_feats, right_frame = fp.Centroids(right_image_msg)

    if len(left_feats) < 1 or len(right_feats) < 1: 
        return None

    left_feat_pts = [(pt.x, pt.y) for pt in left_feats.points]
    right_feat_pts = [(pt.x, pt.y)for pt in right_feats.points]
    
    disparity = abs(left_feat_pts[0].x - right_feat_pts[0].x)
    return stereo_cam_model.projectPixelTo3d(left_feat_pts[0], disparity), left_frame

def tf_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)