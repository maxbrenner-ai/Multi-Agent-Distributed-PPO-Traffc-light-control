import sys
import optparse
from numpy.random import choice
from numpy.random import uniform
import numpy as np
import xml.etree.ElementTree as ET
import contextlib
import os
from multiprocessing import Process
from multiprocessing import Pool
import time


# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

nogui = True
if nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')


def add_vehicle(routes, i):
    # Equal chance for all startChoices
    route = choice(routes)
    return '    <vehicle id="{}_{}" type="car" route="{}" depart="{}" />'.format(route, i, route, i)


NUM_STEPS = 200
def generate_routefile(ID):
    # np.random.seed(42)  # make tests reproducible
    routes_array = ['west_east', 'west_north', 'west_south',
                    'east_west', 'east_north', 'east_south',
                    'north_south', 'north_east', 'north_west',
                    'south_north', 'south_west', 'south_east']
    with open("data/cross_{}.rou.xml".format(ID), "w") as routes:
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
        for i in range(NUM_STEPS):
            if uniform(0, 1) < trafficProb:
                print(add_vehicle(routes_array, i), file=routes)
        print("</routes>", file=routes)

def generate_configfile(ID):
    with open('data/cross_{}.sumocfg'.format(ID), 'w') as config:
        print("""<?xml version="1.0" encoding="UTF-8"?>
            <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
                <input>
                    <net-file value="cross.net.xml"/>
                    <route-files value="cross_{}.rou.xml"/>
                    <additional-files value="cross.add.xml"/>
                </input>

                <time>
                    <begin value="0"/>
                </time>

                <report>
                    <verbose value="false"/>
                    <no-step-log value="true"/>
                </report>

            </configuration>
        """.format(ID), file=config)

def runAgent(args):
    i = args
    print('Agent: ' + str(i))
    label = 'label_'+str(i)
    generate_routefile(i)
    generate_configfile(i)
    traci.start([sumoBinary, "-c", "data/cross_{}.sumocfg".format(i), '--start'], label=label)
    step = 0
    while traci.getConnection(label).simulation.getMinExpectedNumber() > 0:
        traci.getConnection(label).simulationStep()
        # print('Agent {} -- step: {} -- {}'.format(i, step, traci.getConnection(label).edge.getLastStepOccupancy('inNorth')))
        # time.sleep(1)
        step += 1
    traci.getConnection(label).close()
    print('done with agent ' + str(i))


if __name__ == '__main__':
    p = Pool(3)
    p.map(runAgent, range(3))
