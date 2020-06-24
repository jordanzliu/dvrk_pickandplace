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
    DONE = 7

VECTOR_EPS = 0.005

def vector_eps_eq(lhs, rhs):
    return bool((lhs - rhs).Norm() < 0.005)

class PickAndPlaceStateMachine:
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
        self._set_arm_dest(self._obj_pos() + self.approach_vec)


    def _grab_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self._obj_pos() + self.approach_vec):
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
        # open_jaw() sets jaw to 80 deg, we check if we're open past 30 deg
        if self.psm.get_current_jaw_position() > math.pi / 6:
            # jaw is open, state is done, check if we finish or go back to APPROACH_OBJECT
            
            # object closest to original object
            closest_obj = min(self.world.objects, 
                              key=lambda obj: (obj.pos - self.object.pos).Norm())
            loginfo("Closest object to {}: {}".format(self.object.pos, closest_obj))
            if self.closed_loop and closest_obj.color == self.object.color \
                and (closest_obj.pos - self.object.pos).Norm() < 0.01:
                # we didn't pick up the object, go back to APPROACH_OBJECT
                logwarn("Failed to pick up object {}, trying again".format(self.object))
                return PickAndPlaceState.APPROACH_OBJECT
            else:
                loginfo("Done pick and place for object {}".format(self.object))
                return PickAndPlaceState.DONE

        return PickAndPlaceState.DROP_OBJECT


    def _obj_pos(self):
        return self.world_to_psm_tf * self.object.pos

    
    def _approach_vec(self):
        return self.world_to_psm_tf * self.approach_vec


    def _obj_dest(self):
        return self.world_to_psm_tf * self.obj_dest

    def _set_arm_dest(self, dest):
        loginfo("Setting {} dest to {}".format(self.psm.name(), dest))
        if self.psm.get_desired_position().p != dest:
            self.psm.move(dest, blocking=False)


    def __init__(self, psm, world, world_to_psm_tf, object, approach_vec, closed_loop=True):
        loginfo("PickAndPlaceStateMachine:__init__")
        loginfo("psm: {}, world: {}, world_to_psm_tf: {}, object: {}".format(
            psm.name(), world, world_to_psm_tf, object))
        self.state = PickAndPlaceState.OPEN_JAW
        self.psm = psm
        self.object = object
        self.world = world
        self.world_to_psm_tf = world_to_psm_tf
        self.approach_vec = approach_vec
        # if this is False, we don't check if we successfully picked up the object
        # and go straight to the done state
        self.closed_loop = closed_loop
        self.obj_dest = world.bowl.pos + PyKDL.Vector(0, 0, 0.05)
        self.state_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object, 
            PickAndPlaceState.CLOSE_JAW : self._close_jaw,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest,
            PickAndPlaceState.DROP_OBJECT : self._drop_object
        }
        self.next_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw_next,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object_next,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object_next,
            PickAndPlaceState.CLOSE_JAW : self._close_jaw_next,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest_next,
            PickAndPlaceState.DROP_OBJECT : self._drop_object_next
        }


    def update_world(self, world):
        self.world = world


    def run_once(self):
        loginfo("Running state {}".format(self.state))
        if self.is_done():
            return
        # execute the current state
        self.state_functions[self.state]()

        self.state = self.next_functions[self.state]()

    def is_done(self):
        return self.state == PickAndPlaceState.DONE
    

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
