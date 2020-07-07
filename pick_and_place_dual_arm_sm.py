
from enum import Enum
import math
from rospy import loginfo, logwarn
import PyKDL
import pprint

# This entire class is straight up the worst code i've written since high school
# this is what happens when everyone else in research writes spaghetti code so 
# i might as well
class PickAndPlaceState(Enum):
    OPEN_JAW = 1,
    APPROACH_OBJECT = 2,
    GRAB_OBJECT = 3,
    CLOSE_JAW = 4,
    APPROACH_DEST = 5,
    DROP_OBJECT = 6,
    DONE = 7


VECTOR_EPS = 0.005

DOWN_JAW_ORIENTATION = PyKDL.Rotation.RPY(math.pi, 0, - math.pi / 2.0)

PSM_HOME_POS = PyKDL.Vector(0., 0., -0.1)

def vector_eps_eq(lhs, rhs):
    return bool((lhs - rhs).Norm() < 0.005)

class PickAndPlaceDualArmStateMachine:
    def _open_jaw(self):
        if self.psm1.get_desired_jaw_position() < math.pi / 3:
            self.psm1.open_jaw(blocking=False)
        if self.psm2.get_desired_jaw_position() < math.pi / 3:
            self.psm2.open_jaw(blocking=False)
            
    
    def _open_jaw_next(self):
        # open_jaw() sets jaw to 80 deg, we check if we're open past 60 deg
        if self.psm1.get_current_jaw_position() < math.pi / 3 and \
           self.psm2.get_current_jaw_position() < math.pi / 3:
            return PickAndPlaceState.OPEN_JAW
        else:
            return PickAndPlaceState.APPROACH_OBJECT


    def _approach_object(self):
        if self._psm1_object_pos() is not None:
            self._set_arm_dest(self.psm1, self._psm1_object_pos())
        if self._psm2_object_pos() is not None:
            self._set_arm_dest(self.psm2, self._psm2_object_pos())


    def _approach_object_next(self):
        if self._reached_dest(self.psm1, self._psm1_object_pos()) and \
           self._reached_dest(self.psm2, self._psm2_object_pos()):
            return PickAndPlaceState.GRAB_OBJECT
        else:
            return PickAndPlaceState.APPROACH_OBJECT
    

    def _grab_object(self):
        psm1_obj_pos = self._psm1_object_pos()
        psm2_obj_pos = self._psm2_object_pos()

        if psm1_obj_pos is not None:
            self._set_arm_dest(self.psm1, psm1_obj_pos + self.psm1_approach_vec)
        if psm2_obj_pos is not None:
            self._set_arm_dest(self.psm2, psm2_obj_pos + self.psm2_approach_vec)


    def _grab_object_next(self):
        if self._reached_dest(self.psm1, self._psm1_object_pos() + self.psm1_approach_vec) and \
           self._reached_dest(self.psm2, self._psm2_object_pos() + self.psm2_approach_vec):
            return PickAndPlaceState.CLOSE_JAW
        else:
            return PickAndPlaceState.GRAB_OBJECT

    
    def _close_jaw(self):
        if self.psm1.get_desired_jaw_position() > 0:
            self.psm1.close_jaw(blocking=False)
        if self.psm2.get_desired_jaw_position() > 0:
            self.psm2.close_jaw(blocking=False)

    
    def _close_jaw_next(self):
        if self.psm1.get_current_jaw_position() > 0 and \
            self.psm2.get_current_jaw_position() > 0:
            return PickAndPlaceState.CLOSE_JAW
        else:
            return PickAndPlaceState.APPROACH_DEST


    def _approach_dest(self):
        if self._psm1_object_pos() is not None:
            self._set_arm_dest(self.psm1, self._obj_dest(self.psm1))
        if self._psm1_object_pos() is not None:
            self._set_arm_dest(self.psm2, self._obj_dest(self.psm2))


    def _approach_dest_next(self):
        if self._reached_dest(self.psm1, self._obj_dest(self.psm1)) and \
            self._reached_dest(self.psm2, self._obj_dest(self.psm2)):
            return PickAndPlaceState.DROP_OBJECT
        else:
            return PickAndPlaceState.OPEN_JAW          


    def _obj_dest(self, psm):
        if psm == self.psm1:
            return self.world_to_psm1_tf * self.obj_dest
        else:
            return self.world_to_psm2_tf * self.obj_dest


    def _reached_dest(self, psm, point):
        return psm._arm__goal_reached and \
                vector_eps_eq(psm.get_current_position().p, point)


    def _set_arm_dest(self, psm, dest):
        if self.log_verbose:
            loginfo("Setting {} dest to {}".format(psm.name(), dest))
        if psm.get_desired_position().p != dest:
            psm.move(PyKDL.Frame(DOWN_JAW_ORIENTATION, dest), blocking=False)


    def _psm1_object_pos(self):
        if self.psm1_object is not None:
            return self.world_to_psm1_tf * self.psm1_object.pos
        else:
            return None


    def _psm2_object_pos(self):
        if self.psm2_object is not None:
            return self.world_to_psm2_tf * self.psm2_object.pos
        else:
            return None


    def _get_objects_for_psms(self):
        '''
        Returns a dict of PSM index -> list of objects that are closest to that PSM
        '''
        result = dict()
        for object in self.world.objects:
            closest_psm_idx = [self.world_to_psm1_tf, self.world_to_psm2_tf].index(
                min([self.world_to_psm1_tf, self.world_to_psm2_tf], 
                    key=lambda tf : (tf * object.pos).Norm()))
            
            if closest_psm_idx not in result:
                result[closest_psm_idx] = list()
            
            result[closest_psm_idx].append(object)

        if self.log_verbose:
            loginfo("Unpicked objects: {}".format(pprint.pformat(result)))

        return result
        

    def __init__(self, psm1_and_tf, psm2_and_tf, world, approach_vec, log_verbose=False):
        self.log_verbose = log_verbose
        if self.log_verbose:
            loginfo("PickAndPlaceDualArmStateMachine:__init__")

        self.state = PickAndPlaceState.OPEN_JAW
        self.psm1, self.world_to_psm1_tf = psm1_and_tf
        self.psm2, self.world_to_psm2_tf = psm2_and_tf

        self.world = world
        self.psm1_approach_vec = self.world_to_psm1_tf * approach_vec
        self.psm2_approach_vec = self.world_to_psm2_tf * approach_vec

        self.obj_dest = world.bowl.pos + PyKDL.Vector(0, 0, 0.05)
        self.state_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object, 
            PickAndPlaceState.CLOSE_JAW : self._close_jaw,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest,
        }

        self.next_functions = {
            PickAndPlaceState.OPEN_JAW : self._open_jaw_next,
            PickAndPlaceState.APPROACH_OBJECT : self._approach_object_next,
            PickAndPlaceState.GRAB_OBJECT : self._grab_object_next,
            PickAndPlaceState.CLOSE_JAW : self._close_jaw_next,
            PickAndPlaceState.APPROACH_DEST : self._approach_dest_next,
        }

        # assign objects to psm1/psm2
        psm_to_object_dict = self._get_objects_for_psms()

        if len(psm_to_object_dict[0]) > 0:
            self.psm1_object = psm_to_object_dict[0][0]
        else:
            self.psm1_object = None

        if len(psm_to_object_dict[0]) > 0:
            self.psm2_object = psm_to_object_dict[1][0]
        else:
            self.psm2_object = None


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
        return self.state == PickAndPlaceState.DONE

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
