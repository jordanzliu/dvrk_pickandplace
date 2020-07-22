from pick_and_place_arm_sm import PickAndPlaceStateMachine, PickAndPlaceState, PSM_HOME_POS, vector_eps_eq
from enum import Enum
from rospy import loginfo, logwarn
import pprint


class PickAndPlaceParentState(Enum):
    PREPARING = 0
    PICKING = 1
    DROPPING = 2
    DONE = 3


class PickAndPlaceHSM:

    def _get_objects_for_psms(self):
        '''
        Returns a dict of PSM index -> list of objects that are closest to that PSM
        '''
        result = dict()
        for object in self.world.objects:
            closest_psm_idx = self.world_to_psm_tfs.index(
                min(self.world_to_psm_tfs, key=lambda tf : (tf * object.pos).Norm()))
            
            if closest_psm_idx not in result:
                result[closest_psm_idx] = list()
            
            result[closest_psm_idx].append(object)

        if self.log_verbose:
            loginfo("Unpicked objects: {}".format(pprint.pformat(result)))

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

        psm_to_unpicked_objects_map = self._get_objects_for_psms()

        for sm_idx in done_sm_idxs:
            if sm_idx in psm_to_unpicked_objects_map:
                # find the position of the current psm
                psm_cur_pos = self.world_to_psm_tfs[sm_idx].Inverse() * \
                              self.psms[sm_idx].get_current_position().p

                closest_obj = min(psm_to_unpicked_objects_map[sm_idx], 
                                  key=lambda obj: (psm_cur_pos - obj.pos).Norm())

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


    def _dropping_next(self):
        if self.dropping_sm.is_done():
            loginfo(self.dropping_sm.psm.name() + " is done dropping")
            # check if the psm has objects left to pick up
            # we have to make sure that the dropping arm is moving away 
            # from the bowl before starting the next drop
            psm_to_unpicked_objects_map = self._get_objects_for_psms()
            sm_idx = self.psms.index(self.dropping_sm.psm)
            if sm_idx in psm_to_unpicked_objects_map:
                # find the position of the current psm
                psm_cur_pos = self.world_to_psm_tfs[sm_idx].Inverse() * \
                              self.psms[sm_idx].get_current_position().p

                closest_obj = min(psm_to_unpicked_objects_map[sm_idx], 
                                  key=lambda obj: (psm_cur_pos - obj.pos).Norm())

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

        # this is a copypaste from _picking_next
        # TODO: reduce code duplication (that's the whole point of having an HSM)
        psm_to_unpicked_objects_map = self._get_objects_for_psms()
        for sm_idx, (psm, world_to_psm_tf) in enumerate(zip(self.psms, self.world_to_psm_tfs)):
            if sm_idx in psm_to_unpicked_objects_map:
                # find the position of the current psm
                psm_cur_pos = self.world_to_psm_tfs[sm_idx].Inverse() * \
                              self.psms[sm_idx].get_current_position().p

                closest_obj = min(psm_to_unpicked_objects_map[sm_idx], 
                                  key=lambda obj: (psm_cur_pos - obj.pos).Norm())
                if self.log_verbose:
                    loginfo("Assigning object {} to {}".format(closest_obj, self.psms[sm_idx].name()))
                
                self.psm_state_machines.append(
                    PickAndPlaceStateMachine(psm, self.world, world_to_psm_tf, 
                                             closest_obj, approach_vec, closed_loop=False)
                )

        self.state = PickAndPlaceParentState.PICKING

        self.state_functions = {
            PickAndPlaceParentState.PICKING : self._picking,
            PickAndPlaceParentState.DROPPING : self._dropping
        }

        self.next_functions = {
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


        
