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
from feature_processor import feature_processor
from tf_conversions import posemath
from math import pi
import dvrk
'''
Some useful methods and constants for picking up a ball with dVRK and CoppeliaSim
'''

PSM_J1_TO_BASE_LINK_ROT = PyKDL.Rotation.RPY(pi / 2, - pi, 0)
PSM_J1_TO_BASE_LINK_TF = PyKDL.Frame(PSM_J1_TO_BASE_LINK_ROT, PyKDL.Vector())

# TODO: make this less hardcoded
RED_BALL_FEAT_PATH = './red_ball.csv'
BLUE_BALL_FEAT_PATH = './blue_ball.csv'
GREEN_BALL_FEAT_PATH = './green_ball.csv'

FEAT_PATHS = [RED_BALL_FEAT_PATH, BLUE_BALL_FEAT_PATH, GREEN_BALL_FEAT_PATH]

CV_TO_CAM_FRAME_ROT = np.asarray([
    [-1, 0, 0], 
    [0, -1, 0],
    [0, 0, 1]
])

def clamp_image_coords(pt, im_shape):
    return tuple(np.clip(pt, (0, 0), np.array(im_shape)[:2] - np.array([1, 1])))


def get_feat_position_and_img(left_image_msg, right_image_msg, stereo_cam_model):
    # this gets the position of the red ball thing in the camera frame
    # and the image with X's on the desired features
    fp = feature_processor(FEAT_PATHS)
    left_feats, left_frame = fp.FindImageFeatures(left_image_msg)
    right_feats, right_frame = fp.FindImageFeatures(right_image_msg)

    left_feat_pts = [feat.pos for feat in left_feats]
    right_feat_pts = [feat.pos for feat in right_feats]
    
    feat_positions = []
    for left_feat, right_feat in zip(left_feat_pts, right_feat_pts):
        disparity = abs(left_feat[0] - right_feat[0])
        ball_pos_cv = stereo_cam_model.projectPixelTo3d(left_feat_pts[0], disparity)
        # there's a fixed rotation to convert this to the camera coordinate frame
        ball_pos_cam = np.matmul(CV_TO_CAM_FRAME_ROT, ball_pos_cv)
        feat_positions.append(tuple(ball_pos_cam))
    print(feat_positions)
    return feat_positions, left_frame


def tf_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)

# a little wrapper, this is to allow parallel actions between psms
class PickAndPlaceTask:
    def __init__(psm, obj_pos_psm, obj_dest_psm, approach_vec):
        self.psm = psm
        self.obj_pos = obj_pos
        self.obj_dest = obj_dest_psm
        self.approach_vec = approach_vec

    def _reached_pose(self, pose):
        # check if we've reached a particular pose within a margin of error
        # TODO: check how the psm class does this
        pass

    def update(self):
        # coroutines
        yield True
