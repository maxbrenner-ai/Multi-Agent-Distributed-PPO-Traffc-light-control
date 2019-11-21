import os

# DISABLE GPU -----
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# -----------

import sys
import optparse
from numpy.random import choice
from numpy.random import uniform
import numpy as np
import xml.etree.ElementTree as ET
import contextlib
from DQN_agent import Agent, RandomAgent, MEMORY_CAPACITY, WARMUP_CAP

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


NUM_EPISODES = 1
NUM_STEPS = 2000
DETS = ['detNorthIn', 'detSouthIn', 'detEastIn', 'detWestIn']

# Default (NE=25, NS=2000, Green Phase Time=5): avg total waiting time of 130 (Simple reward func: 5358)

'''
Things to test:
- Change the yellow phase length
- Add more traffic (might make it so longer green phase is better)
- Look at speed instead of waiting time
'''


def addVehicle(routes, i):
    # Equal chance for all startChoices
    route = choice(routes)
    return '    <vehicle id="{}_{}" type="car" route="{}" depart="{}" />'.format(route, i, route, i)

def generate_routefile():
    # np.random.seed(42)  # make tests reproducible
    routesArray = ['west_east', 'west_north', 'west_south',
              'east_west', 'east_north', 'east_south',
              'north_south', 'north_east', 'north_west',
              'south_north', 'south_west', 'south_east']
    with open("data/cross.rou.xml", "w") as routes:
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

        """.format(*routesArray), file=routes)
        trafficProb = 1/10.  # Chance that a vehicle will be generated on a timestep
        for i in range(NUM_STEPS):
            if uniform(0, 1) < trafficProb:
                print(addVehicle(routesArray, i), file=routes)
        print("</routes>", file=routes)


def reset_ep(test):
    generate_routefile()
    sumoB = sumoBinary
    if test:
        sumoB = checkBinary('sumo-gui')
    traci.start([sumoB, "-c", "data/cross.sumocfg"])

def read_edge_waiting_time(file, collectEdgeIDs):
    tree = ET.parse(file)
    root = tree.getroot()
    waitingTime = []
    for c in root.iter('edge'):
        if c.attrib['id'] in collectEdgeIDs:
            waitingTime.append(float(c.attrib['waitingTime']))
    return sum(waitingTime) / len(waitingTime)

# If warmup then warmup mem is filled, if train then ep max reached
def check_end_run_condition(warmup, agent_mem, curr_ep, test):
    if warmup:
        if agent_mem >= WARMUP_CAP: return True
    elif test:
        if curr_ep == 1: return True  # just runs one episode for test
    else:
        if curr_ep >= NUM_EPISODES: return True
    return False

def get_state():
    # State takes info on if there is at least one car waiting on the detector
    state = [1 if traci.lanearea.getJamLengthVehicle(det) > 0 else 0 for det in DETS]
    # Also add the phase ID to the state
    state.append(traci.trafficlight.getPhase('intersection'))
    # Give it the elapsed time of the current phase
    # Todo: Might wanna try somehow not giving it info on a yellow phase
    elapsedPhaseTime = traci.trafficlight.getPhaseDuration('intersection') - \
                       (traci.trafficlight.getNextSwitch('intersection') -
                        traci.simulation.getTime())
    state.append(elapsedPhaseTime)
    return np.array(state)

def get_reward():
    # Reward is -1 for halted vehicle at intersection, +1 for no halted vehicle
    return sum([-1 if traci.lanearea.getJamLengthVehicle(det) > 0 else 1 for det in DETS])

def execute_action(action):
    # Get the argmax of the two, if the first index is largest then stay in current phase, o.w. switch
    # But dont allow ANY switching if in yellow phase (ie in process of switching)
    currPhase = traci.trafficlight.getPhase('intersection')
    if currPhase == 1 or currPhase == 3:
        return
    if action == 0:  # do nothing
        return
    else:  # switch
        newPhase = currPhase + 1
        traci.trafficlight.setPhase('intersection', newPhase)

def run_ep(agent, def_agent=False, test=False):
    R = 0
    s = get_state()
    done = False
    while not done:
        a = agent.act(s)
        if not def_agent:
            execute_action(a)
        traci.simulationStep()
        s_ = get_state()
        r = get_reward()
        # Check if done
        if traci.simulation.getMinExpectedNumber() <= 0:
            done = True
            s_ = None

        if not test:
            agent.observe((s, a, r, s_))
            agent.replay()

        s = s_
        R += r

    return R

def run(agent, warmup=False, def_agent=False, test=False):
    waitingTimeTotal = 0
    ep = 0
    while not check_end_run_condition(warmup, agent.exp, ep, test):
        reset_ep(test)
        ep_reward = run_ep(agent, def_agent)
        traci.close()
        waitingTime = read_edge_waiting_time('data/edgeData.out', ['inEast', 'inNorth', 'inSouth', 'inWest'])
        waitingTimeTotal += waitingTime
        ep += 1
        print(' ------ \n\nEpisode: {} -- Avg. Waiting time: {}  Reward: {}\n\n ------ '.format(ep, waitingTime, ep_reward))
    print('\n\n\nTotal Avg Waiting Time: {}'.format(waitingTimeTotal / NUM_EPISODES))

class DefaultAgent:
    exp = 0

    def act(self, s):
        return -1

    def observe(self, sample):
        pass

    def replay(self):
        pass

if __name__ == "__main__":
    state_size = 6
    action_size = 2
    agent = Agent(state_size, action_size)
    randomAgent = RandomAgent(action_size)

    # --- Def Agent --- (REMMEBER TO CHANGE THE LOGIC IN NET)
    # defaultAgent = DefaultAgent()
    # run(defaultAgent, def_agent=True)
    # ------

    print(" -----------------\n----------  Initialization with random agent -------------\n----------------")
    run(randomAgent, warmup=True)
    agent.memory = randomAgent.memory
    randomAgent = None
    print("-----------------\n------------- Starting learning ----------------\n----------------")
    run(agent)
    print("-----------------\n------------- Starting testing ----------------\n----------------")
    run(agent, test=True)