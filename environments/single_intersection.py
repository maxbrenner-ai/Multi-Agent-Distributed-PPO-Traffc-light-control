from numpy.random import choice
from numpy.random import uniform
from sumolib import checkBinary
import traci
import numpy as np
import xml.etree.ElementTree as ET
from environments.environment import Environment
import os, sys


STATE_SIZE = 1
ACTION_SIZE = 2

DETS = ['detNorthIn', 'detSouthIn', 'detEastIn', 'detWestIn']


class SingleIntersectionEnv(Environment):
    def __init__(self, constants, device, agent_ID, eval_agent, vis=False):
        super(SingleIntersectionEnv, self).__init__(constants, device, agent_ID, eval_agent, vis)

    def _open_connection(self):
        self._generate_routefile()
        self._generate_addfile()
        sumoB = checkBinary('sumo' if not self.vis else 'sumo-gui')
        # Need to edit .sumocfg to have the right route file
        self._generate_configfile()
        traci.start([sumoB, "-c", "data/single_intersection_{}.sumocfg".format(self.agent_ID)], label=self.conn_label)
        self.connection = traci.getConnection(self.conn_label)

    def get_state(self):
        state = self._make_state()
        # Give it the elapsed time of the current phase
        elapsedPhaseTime = self.connection.trafficlight.getPhaseDuration('intersection') - \
                           (self.connection.trafficlight.getNextSwitch('intersection') -
                            self.connection.simulation.getTime())
        state._add_to_state(state, elapsedPhaseTime, key='elapsedPhaseTime', intersection=None)
        return self._process_state(state)

    # TODO: THE 36 BELOW IS SPECIFIC TO THE LENGTH OF THE DETECTOR SO REMEMBER TO CHANGE IN THE FUTURE
    # Halt
    def get_reward(self):
        # Reward is -1 for halted vehicle at intersection, +1 for no halted vehicle
        num_stopped = sum([self.connection.lanearea.getJamLengthVehicle(det) for det in DETS])
        reward = 36 - 2 * num_stopped
        reward /= 36  # norm.
        return reward

    # Phase
    # def _execute_action(self, action):
    #     # if action is 0, then dont switch, o.w. switch
    #     # dont allow ANY switching if in yellow phase (ie in process of switching)
    #     currPhase = self.connection.trafficlight.getPhase('intersection')
    #     if currPhase == 1 or currPhase == 3:
    #         return
    #     elif (action == 0 and currPhase == 0) or (action == 1 and currPhase == 2):  # do nothing
    #         return
    #     else:  # switch
    #         newPhase = currPhase + 1
    #         self.connection.trafficlight.setPhase('intersection', newPhase)

    # Switch
    def _execute_action(self, action):
        # if action is 0, then dont switch, o.w. switch
        # dont allow ANY switching if in yellow phase (ie in process of switching)
        currPhase = self.connection.trafficlight.getPhase('intersection')
        if currPhase == 1 or currPhase == 3:
            return
        if action == 0:  # do nothing
            return
        else:  # switch
            newPhase = currPhase + 1
            self.connection.trafficlight.setPhase('intersection', newPhase)

    def _add_vehicle(self, routes, i):
        # Equal chance for all startChoices
        route = choice(routes)
        return '    <vehicle id="{}_{}" type="car" route="{}" depart="{}" />'.format(route, i, route, i)

    def _generate_configfile(self):
        with open('data/single_intersection_{}.sumocfg'.format(self.agent_ID), 'w') as config:
            print("""<?xml version="1.0" encoding="UTF-8"?>
                <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                    <input>
                        <net-file value="single_intersection.net.xml"/>
                        <route-files value="single_intersection_{}.rou.xml"/>
                        <additional-files value="single_intersection_{}.add.xml"/>
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
            """.format(self.agent_ID, self.agent_ID), file=config)

    def _generate_routefile(self):
        # np.random.seed(42)  # make tests reproducible
        routes_array = ['west_east', 'west_north', 'west_south',
                       'east_west', 'east_north', 'east_south',
                       'north_south', 'north_east', 'north_west',
                       'south_north', 'south_west', 'south_east']
        with open("data/single_intersection_{}.rou.xml".format(self.agent_ID), "w") as routes:
            print("""<routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" \
    guiShape="passenger"/>

            <route id="{}" edges="inWest outEast" />
            <route id="{}" edges="inWest outNorth" />
            <route id="{}" edges="inWest outSouth" />

            <route id="{}" edges="inEast outWest" />
            <route id="{}" edges="inEast outNorth" />
            <route id="{}" edges="inEast outSouth" />

            <route id="{}" edges="inNorth outSouth" />
            <route id="{}" edges="inNorth outEast" />
            <route id="{}" edges="inNorth outWest" />

            <route id="{}" edges="inSouth outNorth" />
            <route id="{}" edges="inSouth outWest" />
            <route id="{}" edges="inSouth outEast" />

            """.format(*routes_array), file=routes)
            trafficProb = 1 / 4.  # Chance that a vehicle will be generated on a timestep
            for i in range(self.episode_C['max_ep_steps']):
                if uniform(0, 1) < trafficProb:
                    print(self._add_vehicle(routes_array, i), file=routes)
            print("</routes>", file=routes)

    def _generate_addfile(self):
        with open("data/single_intersection_{}.add.xml".format(self.agent_ID), "w") as add:
            print("""<additionals>
                        <e2Detector id="detNorthIn" lane="inNorth_0" pos="225" endPos="292" freq="100000" friendlyPos="true" file="single_intersection.out"/>
                        <e2Detector id="detSouthIn" lane="inSouth_0" pos="225" endPos="292" freq="100000" friendlyPos="true" file="single_intersection.out"/>
                        <e2Detector id="detEastIn" lane="inEast_0" pos="225" endPos="292" freq="100000" friendlyPos="true" file="single_intersection.out"/>
                        <e2Detector id="detWestIn" lane="inWest_0" pos="225" endPos="292" freq="100000" friendlyPos="true" file="single_intersection.out"/>
                    
                        <edgeData id="edgeData_0" file="edgeData_{}.out.xml"/>
                    </additionals>
            """.format(self.agent_ID), file=add)

