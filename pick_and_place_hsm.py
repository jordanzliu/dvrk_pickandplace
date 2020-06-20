from pick_and_place_arm_sm import PickAndPlaceStateMachine, PickAndPlaceState
from enum import Enum
from rospy import loginfo, logwarn



class PickAndPlaceParentState(Enum):
    PICKING = 0
    DONE = 1

class PickAndPlaceHSM:

    def _picking(self):
        # TODO: this method
        for sm in self.psm_state_machines:
            if not sm.is_done():
                sm.run_once()

    def _picking_next(self):
        # if there are no objects left, go to DONE state 
        if len(self.world.objects) == 0:
            return PickAndPlaceParentState.DONE

        # if a child state machine is done, reset it to OPEN_JAW 
        # with an updated object
        done_state_machines = filter(lambda sm : sm.is_done(), self.psm_state_machines)
        for sm, obj in zip(done_state_machines, self.world.objects):
            sm.object = obj
            sm.state = PickAndPlaceState.OPEN_JAW

        return PickAndPlaceParentState.PICKING

    def update_world(self, world):
        self.world = world


    def __init__(self, psms, world_to_psm_tfs, world, approach_vec):
        if len(world.objects) == 1:
            self.psms = [psms[0]]
            self.world_to_psm_tfs = [world_to_psm_tfs[0]]
        else:
            self.psms = psms
            self.world_to_psm_tfs = world_to_psm_tfs

        self.psm_state_machines = \
            [PickAndPlaceStateMachine(psm, world, tf, world.objects[n], 
                                      approach_vec, closed_loop=False)
            for n, (psm, tf) in enumerate(zip(self.psms, self.world_to_psm_tfs))]

        self.state = PickAndPlaceParentState.PICKING

        self.state_functions = {
            PickAndPlaceParentState.PICKING : self._picking
        }

        self.next_functions = {
            PickAndPlaceParentState.PICKING : self._picking_next
        }

    
    def run_once(self):
        
        if self.state == PickAndPlaceParentState.DONE:
            return
        loginfo("Running state {}".format(self.state))
        self.state_functions[self.state]()
        self.next_functions[self.state]()
        


        
