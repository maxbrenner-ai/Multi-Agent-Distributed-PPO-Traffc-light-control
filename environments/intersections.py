from numpy.random import choice
import random
from sumolib import checkBinary
import traci
import numpy as np
import xml.etree.ElementTree as ET
from environments.environment import Environment
import os, sys
from utils.utils import PiecewiseLinearFunction
from collections import OrderedDict
from utils.env_phases import get_phases, get_current_phase_probs
from utils.net_scrape import *


'''
Assumptions:
1. Single lane roads
2. Even number of intersections (Except for the case of one intersection)
3. Grid like layout, where gen or rem nodes only have one possible edge in and out
'''

# per agent
PER_AGENT_STATE_SIZE = 6
GLOBAL_STATE_SIZE = 1
# per agent
ACTION_SIZE = 2

VEH_LENGTH = 5
VEH_MIN_GAP = 2.5
DET_LENGTH_IN_CARS = 20


def get_rel_net_path(phase_id):
    return phase_id.replace('_rush_hour', '') + '.net.xml'


class IntersectionsEnv(Environment):
    def __init__(self, constants, device, agent_ID, eval_agent, net_path, vis=False):
        super(IntersectionsEnv, self).__init__(constants, device, agent_ID, eval_agent, net_path, vis)
        # For file names
        self.env_name = self.other_C['environment']
        self.phases = get_phases(self.other_C['environment'])
        self.node_edge_dic = get_node_edge_dict(self.net_path)
        self._generate_addfile()

    def _open_connection(self):
        self._generate_routefile()
        sumoB = checkBinary('sumo' if not self.vis else 'sumo-gui')
        # Need to edit .sumocfg to have the right route file
        self._generate_configfile()
        traci.start([sumoB, "-c", "data/{}_{}.sumocfg".format(self.env_name, self.agent_ID)], label=self.conn_label)
        self.connection = traci.getConnection(self.conn_label)

    def _get_sim_step(self, normalize):
        sim_step = self.connection.simulation.getTime()
        if normalize: sim_step /= (self.episode_C['max_ep_steps'] / 10.)  # Normalized between 0 and 10
        return sim_step

    # todo: Might need to normalize vals especially elapsed time!!!!!!!!!!!!!!!!!!!
    def get_state(self):
        # State is made of the jam length for each detector, current phase of each intersection, elapsed time
        # for the current phase of each intersection and the current ep step
        state = self._make_state()
        # Normalize if not rule
        normalize = True if self.agent_type != 'rule' else False
        # Get sim step
        sim_step = self._get_sim_step(normalize)
        for intersection, dets in list(self.intersection_dets.items()):
            # Jam length - WARNING: on sim step 0 this seems to have an issue so just return all zeros
            jam_length = [self.connection.lanearea.getJamLengthVehicle(det) for det in dets] if self.connection.simulation.getTime() != 0 else [0] * len(dets)
            self._add_to_state(state, jam_length, key='jam_length', intersection=intersection)
            # Current phase
            curr_phase = self.connection.trafficlight.getPhase(intersection)
            self._add_to_state(state, curr_phase, key='curr_phase', intersection=intersection)
            # Elapsed time of current phase
            elapsed_phase_time = self.connection.trafficlight.getPhaseDuration(intersection) - \
                               (self.connection.trafficlight.getNextSwitch(intersection) -
                                self.connection.simulation.getTime())
            if normalize: elapsed_phase_time /= 10.  # Slight normalization
            self._add_to_state(state, elapsed_phase_time, key='elapsed_phase_time', intersection=intersection)
            # Add global param of current sim step
            if not self.single_agent and self.agent_type != 'rule':
                self._add_to_state(state, sim_step, key='sim_step', intersection=intersection)
        # DeMorgan's law of above
        if self.single_agent or self.agent_type == 'rule':
            self._add_to_state(state, sim_step, key='sim_step', intersection=None)
        state = self._process_state(state)
        return state

    # Halt
    # ASSUMPTION: this returns the global reward (for now)
    def get_reward(self):
        # Reward is -1 for halted vehicle at intersection, +1 for no halted vehicle
        num_stopped = sum([self.connection.lanearea.getJamLengthVehicle(det) for det in self.all_dets])
        reward = (len(self.all_dets) * DET_LENGTH_IN_CARS) - 2 * num_stopped
        reward /= (len(self.all_dets) * DET_LENGTH_IN_CARS)  # norm.
        assert -1.001 <= reward <= 1.001
        return reward

    # Switch
    # action: {"intersectionNW": 0 or 1, .... }
    def _execute_action(self, action):
        # dont allow ANY switching if in yellow phase (ie in process of switching)
        # Loop through digits, one means switch, zero means stay
        for intersection in self.intersections:
            value = action[intersection]
            currPhase = self.connection.trafficlight.getPhase(intersection)
            if currPhase == 1 or currPhase == 3:  # Yellow, pass
                continue
            if value == 0:  # do nothing
                continue
            else:  # switch
                newPhase = currPhase + 1
                self.connection.trafficlight.setPhase(intersection, newPhase)

    def _generate_configfile(self):
        with open('data/{}_{}.sumocfg'.format(self.env_name, self.agent_ID), 'w') as config:
            print("""<?xml version="1.0" encoding="UTF-8"?>
                <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                    <input>
                        <net-file value="{}"/>
                        <route-files value="{}_{}.rou.xml"/>
                        <additional-files value="{}_{}.add.xml"/>
                    </input>

                    <time>
                        <begin value="0"/>
                    </time>

                    <report>
                        <verbose value="false"/>
                        <no-step-log value="true"/>
                        <no-warnings value="true"/>
                    </report>

                </configuration>
            """.format(get_rel_net_path(self.env_name), self.env_name, self.agent_ID, self.env_name, self.agent_ID), file=config)

    def _add_vehicle(self, t, node_probs, routes_string):
        rem_nodes = []
        rem_probs = []
        for k, v in list(node_probs.items()):
            rem_nodes.append(k)
            rem_probs.append(v['rem'])

        # Pick removal func
        def pick_rem_edge(gen_node):
            while True:
                chosen = choice(rem_nodes, p=rem_probs)
                if chosen != gen_node: return chosen

        # Loop through all gen edges and see if a veh generates
        for gen_k, dic in list(node_probs.items()):
            gen_prob = dic['gen']
            if random.random() < gen_prob:
                # It does generate a veh. so pick a removal edge
                rem_k = pick_rem_edge(gen_k)
                route_id = gen_k + '___' + rem_k
                gen_edge = self.node_edge_dic[gen_k]['gen']
                rem_edge = self.node_edge_dic[rem_k]['rem']
                routes_string += '    <trip id="{}_{}" type="car" from="{}" to="{}" depart="{}" />'.format(route_id, t, gen_edge, rem_edge, t)
        return routes_string

    def _generate_routefile(self):
        # np.random.seed(42)  # make tests reproducible
        routes_string = \
        """
        <routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="{}" minGap="{}" maxSpeed="15" guiShape="passenger"/>
        """.format(VEH_LENGTH, VEH_MIN_GAP)
        # Add the vehicles
        for t in range(self.episode_C['generation_ep_steps']):
            routes_string = self._add_vehicle(t, get_current_phase_probs(t, self.phases, self.episode_C['generation_ep_steps']), routes_string)
        routes_string += '</routes>'
        # Output
        with open("data/{}_{}.rou.xml".format(self.env_name, self.agent_ID), "w") as routes:
            print(routes_string, file=routes)

    def _generate_addfile(self):
        self.all_dets = []  # For reward function
        self.intersection_dets = OrderedDict({k: [] for k in self.intersections})  # For state
        add_string = '<additionals>'
        # Loop through the net file to get all edges that go to an intersection
        tree = ET.parse(self.net_path)
        root = tree.getroot()
        for c in root.iter('edge'):
            id = c.attrib['id']
            # If function key in attib then continue or if not going to intersection
            if 'function' in c.attrib:
                continue
            if not 'intersection' in c.attrib['to']:
                continue
            length = float(c[0].attrib['length'])  # SINGLE LANE ONLY
            pos = length - (DET_LENGTH_IN_CARS * (VEH_LENGTH + VEH_MIN_GAP))
            det_id = 'DET+++'+id
            self.all_dets.append(det_id)
            self.intersection_dets[c.attrib['to']].append(det_id)
            add_string += ' <e2Detector id="{}" lane="{}_0" pos="{}" endPos="{}" freq="100000" ' \
                          'friendlyPos="true" file="{}.out"/>' \
                          ''.format(det_id, id, pos, length, self.env_name)
        add_string += \
        """
            <edgeData id="edgeData_0" file="edgeData_{}.out.xml"/>
         </additionals>
        """.format(self.agent_ID)
        with open("data/{}_{}.add.xml".format(self.env_name, self.agent_ID), "w") as add:
            print(add_string, file=add)
