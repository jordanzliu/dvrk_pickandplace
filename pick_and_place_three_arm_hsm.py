from pick_and_place_arm_sm import PickAndPlaceStateMachine, PickAndPlaceState, PSM_HOME_POS, vector_eps_eq
from copy import deepcopy, copy
from enum import Enum
from rospy import loginfo, logwarn
import numpy as np
import pprint


class PickAndPlaceParentState(Enum):
    PREPARING = 0
    PICKING = 1
    DROPPING = 2
    DONE = 3


class PickAndPlaceThreeArmHSM:

    def _get_psm_object_assignments(self):
        '''
        Returns a dict of psm idx's-> the object they are assigned to pick up
        '''
        if len(self.world.objects) == 0:
            loginfo("No objects!")
            return dict()

        result = dict()

        if len(self.world.objects) == 0:
            return dict()

        # this spaghetti code means this class doesnt work for >2 arms anymore
        objects = [obj for obj in self.world.objects]
        psm1_objects = []
        psm2_objects = []
        psm3_objects = []
        # PSM1 is on the +y side
        # psm1_objects = filter(lambda obj: obj.pos.y() < self.median_object_y, objects)
        # psm2_objects = filter(lambda obj: obj.pos.y() >= self.median_object_y, objects)
        # psm3_objects = filter(lambda obj: obj.pos.y() >= self.median_object_y, objects)
        # psm1_objects.extend([objects[1], objects[4]])   # red 0
        # psm2_objects.extend([objects[0], objects[3]])   # green 2
        # psm3_objects.extend([objects[2], objects[5]])   # blue 1

        for object in self.world.objects:
            if object.color == 0:
                psm1_objects.append(object)
            if object.color == 2:
                psm2_objects.append(object)
            if object.color == 1:
                psm3_objects.append(object)

        # if self.log_verbose:
        #     loginfo("PSM1 objects left: {}, PSM2 objects left: {}, PSM3 objects left: {}".format(psm1_objects, psm2_objects, psm3_objects))
        
        result = dict()

        if psm1_objects:
            result[0] = min(psm1_objects, 
                            key=lambda obj: (self.psms[0].get_current_position().p \
                                                - (self.world_to_psm_tfs[0] * obj.pos)).Norm())

        if psm2_objects:
            result[1] = min(psm2_objects, 
                            key=lambda obj: (self.psms[1].get_current_position().p \
                                                - (self.world_to_psm_tfs[1] * obj.pos)).Norm())

        if psm3_objects:
            result[2] = min(psm3_objects, 
                            key=lambda obj: (self.psms[2].get_current_position().p \
                                                - (self.world_to_psm_tfs[2] * obj.pos)).Norm()) 

        return result


    def _preparing(self):
        for sm in self.psm_state_machines:
            if (not sm.is_done()) and sm.state == PickAndPlaceState.OPEN_JAW:
                sm.run_once()


    def _preparing_next(self):
        if all([sm.jaw_fully_open() for sm in self.psm_state_machines]):
            return PickAndPlaceParentState.PICKING
        else:
            return PickAndPlaceParentState.PREPARING


    def _picking(self):
        for sm in self.psm_state_machines:
            if not sm.is_done():
                sm.run_once()

    def _picking_next(self):
        # if there are no objects left, go to DONE state 
        if len(self.world.objects) == 0:
            return PickAndPlaceParentState.DONE

        # if a child state machine is done, reset it to APPROACH_OBJECT 
        # with an updated object
        done_sm_idxs = filter(lambda sm_idx : self.psm_state_machines[sm_idx].is_done(), 
                              range(len(self.psm_state_machines)))
                            
        if self.log_verbose:
            loginfo("Done child sms: {}".format(done_sm_idxs))

        psm_to_closest_object_map = self._get_psm_object_assignments()

        for sm_idx in done_sm_idxs:
            if sm_idx in psm_to_closest_object_map:
                closest_obj = psm_to_closest_object_map[sm_idx]

                if self.log_verbose:
                    loginfo("Assigning object {} to {}".format(closest_obj, self.psms[sm_idx].name()))
                
                self.psm_state_machines[sm_idx].object = closest_obj
                self.psm_state_machines[sm_idx].state = PickAndPlaceState.APPROACH_OBJECT

        # if a child state machine is in the APPROACH_DEST state, we transition to the 
        # DROPPING state

        dropping_sm_idxs = filter(lambda sm_idx : \
            self.psm_state_machines[sm_idx].state == PickAndPlaceState.APPROACH_DEST, 
            range(len(self.psm_state_machines)))

        if self.log_verbose:
            loginfo("Child sm states: {}".format([sm.state for sm in self.psm_state_machines]))

        if len(dropping_sm_idxs) > 0:
            self.dropping_sm = self.psm_state_machines[dropping_sm_idxs[0]]

            for sm_idx, sm in enumerate(self.psm_state_machines):
                if sm_idx != dropping_sm_idxs[0] and sm.state == PickAndPlaceState.APPROACH_DEST:
                    sm.halt()
            
            if self.log_verbose:
                loginfo("Entering DROPPING state!")
                
            return PickAndPlaceParentState.DROPPING

        return PickAndPlaceParentState.PICKING


    def _dropping(self):
        if not self.dropping_sm.is_done():
            self.dropping_sm.run_once()

        for sm in self.psm_state_machines:
            if sm != self.dropping_sm and sm.state != PickAndPlaceState.APPROACH_DEST:
                sm.run_once()


    def _dropping_next(self):
        if self.dropping_sm.is_done():
            loginfo(self.dropping_sm.psm.name() + " is done dropping")

            # check if the psm has objects left to pick up
            # we have to make sure that the dropping arm is moving away 
            # from the bowl before starting the next drop
            psm_to_closest_object_map = self._get_psm_object_assignments()
            sm_idx = self.psms.index(self.dropping_sm.psm)

            if sm_idx in psm_to_closest_object_map:
                closest_obj = psm_to_closest_object_map[sm_idx]
                if self.log_verbose:
                    loginfo("Assigning object {} to {}".format(closest_obj, self.psms[sm_idx].name()))
                
                # if there's an object to pick up, move to the next one
                self.psm_state_machines[sm_idx].object = closest_obj
                self.psm_state_machines[sm_idx].state = PickAndPlaceState.APPROACH_OBJECT
            else:
                # no objects left, set arm to home
                self.dropping_sm.state = PickAndPlaceState.HOME
            self.dropping_sm = None
            loginfo([sm.state for sm in self.psm_state_machines])
            return PickAndPlaceParentState.PICKING
        else:
            return PickAndPlaceParentState.DROPPING


    def update_world(self, world):
        self.world = world
        for psm_sm in self.psm_state_machines:
            psm_sm.update_world(world)

    def is_done(self):
        return self.state == PickAndPlaceParentState.DONE

    def __init__(self, psms, world_to_psm_tfs, world, approach_vec, log_verbose=False):
        self.log_verbose = log_verbose
        self.world = world
        if len(world.objects) == 1:
            self.psms = [psms[0]]
            self.world_to_psm_tfs = [world_to_psm_tfs[0]]
        else:
            self.psms = psms
            self.world_to_psm_tfs = world_to_psm_tfs

        self.psm_state_machines = list()

        # find the median object y-position while all the objects are on the table
        # to do object assignment robustly
        self.median_object_y = np.median([obj.pos.y() for obj in self.world.objects])
        print("Median object y-position: {}".format(self.median_object_y))

        # this is a copypaste from _picking_next
        # TODO: reduce code duplication (that's the whole point of having an HSM)
        psm_to_closest_object_map = self._get_psm_object_assignments()
        
        for sm_idx, (psm, world_to_psm_tf) in enumerate(zip(self.psms, self.world_to_psm_tfs)):
            if sm_idx in psm_to_closest_object_map:
                closest_obj = psm_to_closest_object_map[sm_idx]
                if self.log_verbose:
                    loginfo("Assigning object {} to {}".format(closest_obj, self.psms[sm_idx].name()))
                
                self.psm_state_machines.append(
                    PickAndPlaceStateMachine(psm, self.world, world_to_psm_tf, 
                                             closest_obj, approach_vec, closed_loop=False)
                )

        self.state = PickAndPlaceParentState.PREPARING

        self.state_functions = {
            PickAndPlaceParentState.PREPARING : self._preparing,
            PickAndPlaceParentState.PICKING : self._picking,
            PickAndPlaceParentState.DROPPING : self._dropping
        }

        self.next_functions = {
            PickAndPlaceParentState.PREPARING : self._preparing_next,
            PickAndPlaceParentState.PICKING : self._picking_next,
            PickAndPlaceParentState.DROPPING : self._dropping_next
        }

        self.dropping_sm = None

    
    def run_once(self):
        if self.state == PickAndPlaceParentState.DONE:
            return

        if self.log_verbose:
            loginfo("Running state {}".format(self.state))

        self.state_functions[self.state]()
        self.state = self.next_functions[self.state]()
        

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


        
