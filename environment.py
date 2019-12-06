from numpy.random import choice
from numpy.random import uniform
from sumolib import checkBinary
import traci
import numpy as np
import xml.etree.ElementTree as ET

import os, sys


DETS = ['detNorthIn', 'detSouthIn', 'detEastIn', 'detWestIn']


class Environment:
    def __init__(self, constants, device, agent_ID, eval_agent, vis=False):
        self.episode_C, self.model_C, self.agent_C, self.other_C = constants['episode_C'], constants['model_C'], \
                                                   constants['agent_C'], constants['other_C']
        self.device = device
        self.agent_ID = agent_ID
        self.eval_agent = eval_agent
        # For sumo connection
        self.conn_label = 'label_' + str(self.agent_ID)
        self.vis = vis

    def _open_connection(self):
        self._generate_routefile()
        self._generate_addfile()
        sumoB = checkBinary('sumo' if not self.vis else 'sumo-gui')
        # Need to edit .sumocfg to have the right route file
        self._generate_configfile()
        traci.start([sumoB, "-c", "data/cross_{}.sumocfg".format(self.agent_ID)], label=self.conn_label)
        self.connection = traci.getConnection(self.conn_label)

    def _close_connection(self):
        self.connection.close()
        del traci.main._connections[self.conn_label]

    def reset(self):
        # If there is a conn to close, then close it
        if self.conn_label in traci.main._connections:
            self._close_connection()
        # Start a new one
        self._open_connection()
        return self.get_state()

    def step(self, a, ep_step, def_agent=False):
        if not def_agent:
            self._execute_action(a)
        self.connection.simulationStep()
        s_ = self.get_state()
        r = self.get_reward()
        # Check if done (and if so reset)
        done = False
        if self.connection.simulation.getMinExpectedNumber() <= 0 or ep_step >= self.episode_C['max_ep_steps']:
            # Just close the conn without restarting if eval agent
            if self.eval_agent:
                self._close_connection()
            else:
                s_ = self.reset()
            done = True
        return s_, r, done

    def get_state(self):
        # State takes info on if there is at least one car waiting on the detector
        state = [1 if self.connection.lanearea.getJamLengthVehicle(det) > 0 else 0 for det in DETS]
        # Also add the phase ID to the state
        state.append(self.connection.trafficlight.getPhase('intersection'))
        # Give it the elapsed time of the current phase
        # Todo: Might wanna try somehow not giving it info on a yellow phase
        elapsedPhaseTime = self.connection.trafficlight.getPhaseDuration('intersection') - \
                           (self.connection.trafficlight.getNextSwitch('intersection') -
                            self.connection.simulation.getTime())
        state.append(elapsedPhaseTime)
        return np.expand_dims(np.array(state), axis=0)  # Need to have state be 2 dim

    def get_reward(self):
        # Reward is -1 for halted vehicle at intersection, +1 for no halted vehicle
        return sum([-1 if self.connection.lanearea.getJamLengthVehicle(det) > 0 else 1 for det in DETS])

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

    def _generate_configfile(self):
        with open('data/cross_{}.sumocfg'.format(self.agent_ID), 'w') as config:
            print("""<?xml version="1.0" encoding="UTF-8"?>
                <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                    <input>
                        <net-file value="cross.net.xml"/>
                        <route-files value="cross_{}.rou.xml"/>
                        <additional-files value="cross_{}.add.xml"/>
                    </input>
                
                    <time>
                        <begin value="0"/>
                    </time>
                
                    <report>
                        <verbose value="false"/>
                        <no-step-log value="true"/>
                    </report>
                    
                </configuration>
            """.format(self.agent_ID, self.agent_ID), file=config)

    def _add_vehicle(self, routes, i):
        # Equal chance for all startChoices
        route = choice(routes)
        return '    <vehicle id="{}_{}" type="car" route="{}" depart="{}" />'.format(route, i, route, i)

    def _generate_routefile(self):
        # np.random.seed(42)  # make tests reproducible
        routes_array = ['west_east', 'west_north', 'west_south',
                       'east_west', 'east_north', 'east_south',
                       'north_south', 'north_east', 'north_west',
                       'south_north', 'south_west', 'south_east']
        with open("data/cross_{}.rou.xml".format(self.agent_ID), "w") as routes:
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
            trafficProb = 1 / 10.  # Chance that a vehicle will be generated on a timestep
            for i in range(self.episode_C['max_ep_steps']):
                if uniform(0, 1) < trafficProb:
                    print(self._add_vehicle(routes_array, i), file=routes)
            print("</routes>", file=routes)

    def _generate_addfile(self):
        with open("data/cross_{}.add.xml".format(self.agent_ID), "w") as add:
            print("""<additionals>
                        <e2Detector id="detNorthIn" lane="inNorth_0" pos="285" endPos="292" freq="100000" friendlyPos="true" file="cross.out"/>
                        <e2Detector id="detSouthIn" lane="inSouth_0" pos="285" endPos="292" freq="100000" friendlyPos="true" file="cross.out"/>
                        <e2Detector id="detEastIn" lane="inEast_0" pos="285" endPos="292" freq="100000" friendlyPos="true" file="cross.out"/>
                        <e2Detector id="detWestIn" lane="inWest_0" pos="285" endPos="292" freq="100000" friendlyPos="true" file="cross.out"/>
                    
                        <edgeData id="edgeData_0" file="edgeData_{}.out.xml"/>
                    </additionals>
            """.format(self.agent_ID), file=add)

