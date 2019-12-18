from numpy.random import choice
from numpy.random import uniform
from sumolib import checkBinary
import traci
import numpy as np
import xml.etree.ElementTree as ET
import os, sys


# Base class
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
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def _execute_action(self, action):
        raise NotImplementedError

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
                        <no-warnings value="true"/>
                    </report>
                    
                </configuration>
            """.format(self.agent_ID, self.agent_ID), file=config)

    def _generate_routefile(self):
        raise NotImplementedError

    def _generate_addfile(self):
        raise NotImplementedError

