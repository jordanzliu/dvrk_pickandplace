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
from feature_processor import feature_processor, FeatureType
from tf_conversions import posemath
from math import pi
import dvrk
from collections import OrderedDict 

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

class Object3d:
    def __init__(self, pos_cam, type):
        self.pos_cam = pos_cam
        self.type = type

def clamp_image_coords(pt, im_shape):
    return tuple(np.clip(pt, (0, 0), np.array(im_shape)[:2] - np.array([1, 1])))


def get_objects_and_img(left_image_msg, right_image_msg, stereo_cam_model):
    # this gets the position of the red ball thing in the camera frame
    # and the image with X's on the desired features
    fp = feature_processor(FEAT_PATHS)
    left_feats, left_frame = fp.FindImageFeatures(left_image_msg)
    right_feats, right_frame = fp.FindImageFeatures(right_image_msg)

    objects = []
    for left_feat, right_feat in zip(left_feats, right_feats):
        disparity = abs(left_feat.pos[0] - right_feat.pos[0])
        print(left_feat.pos)
        pos_cv = stereo_cam_model.projectPixelTo3d(left_feat.pos, float(disparity))
        # there's a fixed rotation to convert this to the camera coordinate frame
        pos_cam = np.matmul(CV_TO_CAM_FRAME_ROT, pos_cv)
        objects.append(Object3d(pos_cam, left_feat.type))
    print(objects)
    return objects, np.hstack((left_frame, right_frame))


def tf_to_pykdl_frame(tfl_frame):
    pos, rot_quat = tfl_frame
    pos2 = PyKDL.Vector(*pos)
    rot = PyKDL.Rotation.Quaternion(*rot_quat)
    return PyKDL.Frame(rot, pos2)


class FeatureTracker:
    def __init__(self):
        self.last_detections = dict()

    def update(self, detections):
        pass
        # TODO: this