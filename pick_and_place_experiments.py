import rospy
import dvrk
import numpy as np
from sensor_msgs import msg
import cv2
import cv_bridge
from copy import deepcopy
import ipywidgets as widgets
from IPython.display import clear_output, Image, display
import PIL.Image
from io import StringIO
import matplotlib.pyplot as plt
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

# TODO change to rospy subscribe with dvrk wrapper
rospy.Subscriber('/stereo/left/image_flipped', msg.Image, left_image_callback)
rospy.Subscriber('/stereo/left/camera_info', msg.CameraInfo, left_camera_info_callback)
rospy.Subscriber('/stereo/right/image_flipped', msg.Image, right_image_callback)
rospy.Subscriber('/stereo/right/camera_info', msg.CameraInfo, right_camera_info_callback)

while left_image is None:
    time.sleep(0.5)
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

psm1 = dvrk.psm('PSM1')
ecm = dvrk.ecm('ECM')
psm2 = dvrk.psm('PSM2')

HARDCODED_ECM_POS = np.array([0.0, 0.0, 0.0, 0.0])
PSM_HOME_POS = np.asarray([0., 0., 0.05, 0., 0., 0.])

# -

tf_listener = tf.TransformListener()

time.sleep(5)
psm1.move_joint(deepcopy(PSM_HOME_POS))
time.sleep(1)
psm2.move_joint(deepcopy(PSM_HOME_POS))
time.sleep(1)
ecm.move_joint(HARDCODED_ECM_POS)

tf_listener.getFrameStrings()

pick_and_place_utils = None
from pick_and_place_utils import get_objects_and_img, tf_to_pykdl_frame, PSM_J1_TO_BASE_LINK_TF, World
import image_geometry

time.sleep(1)
tf_cam_to_jp21 = tf_to_pykdl_frame(tf_listener.lookupTransform('ecm_pitch_link_1', 'camera', rospy.Time()))
tf_jp21_to_world = tf_to_pykdl_frame(tf_listener.lookupTransform('world', 'Jp21_ECM', rospy.Time()))
tf_cam_to_world = tf_jp21_to_world * tf_cam_to_jp21
tf_cam_to_world

# +
tf_world_to_psm2_j1 = tf_to_pykdl_frame(tf_listener.lookupTransform('J1_PSM2', 'world', rospy.Time()))
tf_world_to_psm2_base = PSM_J1_TO_BASE_LINK_TF * tf_world_to_psm2_j1

tf_world_to_psm1_j1 = tf_to_pykdl_frame(tf_listener.lookupTransform('J1_PSM1', 'world', rospy.Time()))
tf_world_to_psm1_base = PSM_J1_TO_BASE_LINK_TF * tf_world_to_psm1_j1
# -

stereo_cam = image_geometry.StereoCameraModel()
stereo_cam.fromCameraInfo(left_camera_info, right_camera_info)
objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam,
                                     cam_to_world_tf=tf_cam_to_world)
world = World(objects)
world

# +
from pick_and_place_arm_sm import PickAndPlaceStateMachine, PickAndPlaceState
from pick_and_place_hsm import PickAndPlaceHSM
from pick_and_place_dual_arm_sm import PickAndPlaceDualArmStateMachine
import IPython
from timeit import default_timer as timer
from pick_and_place_utils import get_objects_for_psms
from enum import Enum

the_image = IPython.display.Image(frame)

objects_to_pick = deepcopy(world.objects)

# this vector is empirically determined
approach_vec = PyKDL.Vector(0.007, 0, -0.015)


class TaskType(Enum):
    OneArmFSM = 0
    TwoArmFSM = 1
    TwoIndependentFSM = 2
    HCFSM = 3


# change this line to change which task is run
task_type = TaskType.OneArmFSM

start_time = time.time()
if task_type == TaskType.OneArmFSM:
    # ========================================================================================================== 
    # Runs 1 FSM
    # ========================================================================================================== 
    sm = PickAndPlaceStateMachine(psm1, world, tf_world_to_psm1_base, None, approach_vec, closed_loop=True)

    while not (sm.is_done() or sm.state == PickAndPlaceState.HOME):
        objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
        world = World(objects)
        sm.update_world(world)
        sm.run_once()

elif task_type == TaskType.TwoArmFSM:
    # ========================================================================================================== 
    # This runs the single FSM that runs both arms sequentially
    # ========================================================================================================== 
    sm = PickAndPlaceDualArmStateMachine([psm1, psm2], [tf_world_to_psm1_base, tf_world_to_psm2_base], world,
                                         approach_vec)

    while not sm.is_done():
        objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
        world = World(objects)
        sm.update_world(world)
        sm.run_once()

elif task_type == TaskType.TwoIndependentFSM:
    # ========================================================================================================== 
    # Runs 2 independent FSMs, one for each arm
    # ========================================================================================================== 

    objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
    world = World(objects)
    original_bowl = world.bowl

    # assign objects to PSM1/PSM2 state machines
    psm_object_dict = get_objects_for_psms(world.objects, [tf_world_to_psm1_base, tf_world_to_psm2_base])

    psm1_sm = PickAndPlaceStateMachine(psm1, world, tf_world_to_psm1_base, None, approach_vec,
                                       closed_loop=True, pick_closest_to_base_frame=True)

    psm2_sm = PickAndPlaceStateMachine(psm2, world, tf_world_to_psm2_base, None, approach_vec,
                                       closed_loop=True, pick_closest_to_base_frame=True)

    # because we exit out of the loop as soon as both arms are in the HOME state 
    # (i.e. they have began to move to the home position), the end time is exactly the first 'tick' after
    # the last arm has opened its gripper past 60 degrees
    while (psm1_sm.state != PickAndPlaceState.HOME) or (psm2_sm.state != PickAndPlaceState.HOME):
        objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
        world = World(objects)
        psm1_sm.update_world(world)
        psm2_sm.update_world(world)
        psm1_sm.run_once()
        psm2_sm.run_once()
elif task_type == TaskType.HCFSM:
    # ========================================================================================================== 
    # This runs the hierarchical concurrent state machine that runs both arms concurrently
    # ========================================================================================================== 
    hsm = PickAndPlaceHSM([psm1, psm2], [tf_world_to_psm1_base, tf_world_to_psm2_base], world, approach_vec,
                          log_verbose=False)

    while not hsm.is_done():
        objects, frame = get_objects_and_img(left_image_msg, right_image_msg, stereo_cam, tf_cam_to_world)
        world = World(objects)
        hsm.update_world(world)
        hsm.run_once()

completion_time = time.time()
print("Task took {} seconds".format(completion_time - start_time))
# -
time.sleep(1)
psm1.move_joint(np.asarray([0., 0., 0.08, 0., 0., 0.]))
psm2.move_joint(np.asarray([0., 0., 0.08, 0., 0., 0.]))

plt.figure(figsize=(10, 5))
plt.imshow(frame)

time.time()
