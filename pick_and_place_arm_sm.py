# state machine for pick and place based on 
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/StateMachine.html
from enum import Enum
import math
from rospy import loginfo, logwarn
import PyKDL
import pprint

# TODO: failed pickup state transition from APPROACH_DEST to APPROACH_OBJECT
# TODO: pass entire estimated world into the run_once function 

class PickAndPlaceState(Enum):
    OPEN_JAW = 1,
    APPROACH_OBJECT = 2,
    GRAB_OBJECT = 3,
    CLOSE_JAW = 4,
    APPROACH_DEST = 5,
    DROP_OBJECT = 6,
    HOME = 7,
    DONE = 8

VECTOR_EPS = 0.005

DOWN_JAW_ORIENTATION = PyKDL.Rotation.RPY(math.pi, 0, - math.pi / 2.0)

PSM_HOME_POS = PyKDL.Vector(0., 0., -0.1)

def vector_eps_eq(lhs, rhs):
    return bool((lhs - rhs).Norm() < 0.005)

class PickAndPlaceStateMachine:
    def jaw_fully_open(self):
        return True if self.psm.get_current_jaw_position() >= math.pi / 3 else False 


    def jaw_fully_closed(self):
        return True if self.psm.get_current_jaw_position() <= 0 else False


    def _open_jaw(self):
        if self.psm.get_desired_jaw_position() < math.pi / 3:
            self.psm.open_jaw(blocking=False)
            
    
    def _open_jaw_next(self):
        # open_jaw() sets jaw to 80 deg, we check if we're open past 60 deg
        if self.psm.get_current_jaw_position() < math.pi / 3:
            return PickAndPlaceState.OPEN_JAW
        else:
            return PickAndPlaceState.APPROACH_OBJECT


    def _approach_object(self):
        self._set_arm_dest(self._obj_pos())


    def _approach_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_pos()):
            return PickAndPlaceState.GRAB_OBJECT
        else:
            return PickAndPlaceState.APPROACH_OBJECT
    

    def _grab_object(self):
        self._set_arm_dest(self._obj_pos() + self._approach_vec())


    def _grab_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_pos() + self._approach_vec()):
            return PickAndPlaceState.CLOSE_JAW
        else:
            return PickAndPlaceState.GRAB_OBJECT

    
    def _close_jaw(self):
        if self.psm.get_desired_jaw_position() > 0:
            self.psm.close_jaw(blocking=False)

    
    def _close_jaw_next(self):
        if self.psm.get_current_jaw_position() > 0:
            return PickAndPlaceState.CLOSE_JAW
        else:
            return PickAndPlaceState.APPROACH_DEST


    def _approach_dest(self):
        self._set_arm_dest(self._obj_dest())


    def _approach_dest_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_dest()):
            return PickAndPlaceState.DROP_OBJECT
        else:
            return PickAndPlaceState.APPROACH_DEST 


    def _drop_object(self):
        if self.psm.get_desired_jaw_position() < math.pi / 3:
            self.psm.open_jaw(blocking=False)


    def _drop_object_next(self):
        # open_jaw() sets jaw to 80 deg, we check if we're open past 60 deg
        if self.psm.get_current_jaw_position() > math.pi / 3:
            # early out if this is being controlled by the parent state machine
            if not self.closed_loop:
                if self.home_when_done:
                    return PickAndPlaceState.HOME
                else:
                    return PickAndPlaceState.DONE

            elif len(self.world.objects) > 0:
                # there are objects left, find one and go to APPROACH_OBJECT
                closest_object = None
                if self.pick_closest_to_base_frame:
                    # closest object to base frame
                    closest_object = min(self.world.objects,
                                        key=lambda obj : (self.world_to_psm_tf * obj.pos).Norm())
                else:
                    # closest object to current position, only if we're running 
                    closest_object = min(self.world.objects,
                                        key=lambda obj : (self.world_to_psm_tf * obj.pos \
                                                        - self.psm.get_current_position().p).Norm())
                self.object = closest_object
                return PickAndPlaceState.APPROACH_OBJECT
            else:
                return PickAndPlaceState.HOME
        else:
            return PickAndPlaceState.DROP_OBJECT

    
    def _home(self):
        self._set_arm_dest(PSM_HOME_POS)

    def _home_next(self):
        # the home state is used for arm state machines that are completely 
        # finished executing as determined by the parent state machine
        return PickAndPlaceState.HOME 

    def _obj_pos(self):
        return self.world_to_psm_tf * self.object.pos

    
    def _approach_vec(self):
        return self.world_to_psm_tf.M * self.approach_vec


    def _obj_dest(self):
        return self.world_to_psm_tf * self.obj_dest

    def _set_arm_dest(self, dest):
        if self.log_verbose:
            loginfo("Setting {} dest to {}".format(self.psm.name(), dest))
        if self.psm.get_desired_position().p != dest:
            self.psm.move(PyKDL.Frame(DOWN_JAW_ORIENTATION, dest), blocking=False)


    def __init__(self, psm, world, world_to_psm_tf, object, approach_vec, 
                 closed_loop=False, use_down_facing_jaw=True, home_when_done=False, pick_closest_to_base_frame=False, 
                 log_verbose=False):
        self.log_verbose = log_verbose
        self.home_when_done = home_when_done
        self.pick_closest_to_base_frame = pick_closest_to_base_frame
        if self.log_verbose:
            loginfo("PickAndPlaceStateMachine:__init__")
            loginfo("psm: {}, world: {}, world_to_psm_tf: {}, object: {}".format(
                    psm.name(), world, world_to_psm_tf, object))
            loginfo("home_when_done: {}".format(self.home_when_done))
        self.state = PickAndPlaceState.OPEN_JAW
        self.psm = psm
        self.world = world
        self.world_to_psm_tf = world_to_psm_tf

        if object is not None:
            self.object = object
        else:
            self.object = min(self.world.objects,
                              key=lambda obj : (self.world_to_psm_tf * obj.pos \
                                - self.psm.get_current_position().p).Norm())

        self.approach_vec = approach_vec
        # if this is False, we don't check if we successfully picked up the object
        # and go straight to the done state
        self.closed_loop = closed_loop
        self.obj_dest = world.bowl.pos + PyKDL.Vector(0, 0, 0.03)
        self.state_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object, 
            PickAndPlaceState.CLOSE_JAW : self._close_jaw,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest,
            PickAndPlaceState.DROP_OBJECT : self._drop_object,
            PickAndPlaceState.HOME : self._home,
        }
        self.next_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw_next,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object_next,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object_next,
            PickAndPlaceState.CLOSE_JAW : self._close_jaw_next,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest_next,
            PickAndPlaceState.DROP_OBJECT : self._drop_object_next,
            PickAndPlaceState.HOME : self._home_next
        }


    def update_world(self, world):
        self.world = world


    def run_once(self):
        if self.log_verbose:
            loginfo("Running state {}".format(self.state))
        if self.is_done():
            return
        # execute the current state
        self.state_functions[self.state]()

        self.state = self.next_functions[self.state]()

    def is_done(self):
        if self.home_when_done:
            return self.state == PickAndPlaceState.HOME and vector_eps_eq(self.psm.get_current_position().p, PSM_HOME_POS) 
        else:
            return self.state == PickAndPlaceState.DONE
    
    def halt(self):
        # this sets the desired joint position to the current joint position
        self.psm.move(self.psm.get_current_position(), blocking=False)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
