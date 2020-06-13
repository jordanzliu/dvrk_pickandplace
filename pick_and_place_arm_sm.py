# state machine for pick and place based on 
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/StateMachine.html
from enum import Enum
import math
from rospy import loginfo

# TODO: failed pickup state transition from APPROACH_DEST to APPROACH_OBJECT
# TODO: pass entire estimated world into the run_once function 

class PickAndPlaceState(Enum):
    OPEN_JAW = 1,
    APPROACH_OBJECT = 2,
    GRAB_OBJECT = 3,
    CLOSE_JAW = 4,
    APPROACH_DEST = 5,
    DROP_OBJECT = 6

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
        if self.psm.get_desired_position().p != self.obj_pos:
            self.psm.move(self.obj_pos, blocking=False)


    def _approach_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self.obj_pos):
            return PickAndPlaceState.GRAB_OBJECT
        else:
            return PickAndPlaceState.APPROACH_OBJECT
    

    def _grab_object(self):
        if self.psm.get_desired_position().p != self.obj_pos + self.approach_vec:
            self.psm.move(self.obj_pos + self.approach_vec, blocking=False)


    def _grab_object_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self.obj_pos + self.approach_vec):
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
        if self.psm.get_desired_position().p != self.obj_dest:
            self.psm.move(self.obj_dest, blocking=False)


    def _approach_dest_next(self):
        if self.psm._arm__goal_reached and \
            vector_eps_eq(self.psm.get_current_position().p, self.obj_dest):
            return PickAndPlaceState.DROP_OBJECT
        else:
            return PickAndPlaceState.APPROACH_DEST 


    def _drop_object(self):
        if self.psm.get_desired_jaw_position() < math.pi / 3:
            self.psm.open_jaw(blocking=False)


    def _drop_object_next(self):
        # open_jaw() sets jaw to 80 deg, we check if we're open past 30 deg
        if self.psm.get_current_jaw_position() > math.pi / 6:
            self._done = True
        return PickAndPlaceState.DROP_OBJECT


    def __init__(self, psm, obj_pos, obj_dest, approach_vec):
        self.state = PickAndPlaceState.APPROACH_OBJECT
        self.psm = psm
        self.obj_pos = obj_pos
        self.obj_dest = obj_dest
        self.approach_vec = approach_vec
        self._done = False
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
        pass


    def run_once(self):
        loginfo("Running state {}".format(self.state))
        if self._done:
            return
        # execute the current state
        self.state_functions[self.state]()

        self.state = self.next_functions[self.state]()

    def is_done(self):
        return self._done
    