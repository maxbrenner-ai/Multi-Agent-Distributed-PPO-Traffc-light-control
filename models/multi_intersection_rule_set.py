from scipy.sparse.csgraph import shortest_path
import numpy as np
from utils.net_scrape import *
from models.rule_set import RuleSet
from random import sample


'''
1. Dont want the light lengths to be less than the yellow phase (3 seconds)
2. Assume NS starting config for lights
'''

class MultiIntersectionRuleSet(RuleSet):
    def __init__(self, params, net_path, constants):
        super(MultiIntersectionRuleSet, self).__init__(params, net_path)
        assert params['phases']
        assert 'cycle_length' in params
        self.net_path = net_path
        self.gen_time = constants['episode_C']['generation_ep_steps']
        self.NS_mult = constants['other_C']['NS_mult']
        self.EW_mult = constants['other_C']['EW_mult']
        self.phase_end_offset = constants['other_C']['phase_end_offset']
        self.cycle_length = params['cycle_length']
        # [{'start_step': t, 'light_lengths': {intersection: {NS: #, EW: #}...}}...]
        self.phases = self._get_light_lengths(params['phases'])
        # print(self.phases)
        self.reset()

    def reset(self):
        self.curr_phase_indx = -1  # Keep track of current phase
        # Keep track of current light for each inetrsection as well as current time with that light
        self.info = {intersection: {'curr_light': 'NS', 'curr_time': 0} for intersection in
                     get_intersections(self.net_path)}

    def _convert_to_dec(self, bin):
        return int("".join(str(x) for x in bin), 2)

    def __call__(self, state):
        old_lights = []
        new_lights = []
        t = state['ep_step']
        next_phase = False
        if self.curr_phase_indx < len(self.phases)-1 and self.phases[self.curr_phase_indx + 1]['start_step'] == t:
            next_phase = True
            self.curr_phase_indx += 1

        for intersection in list(self.info.keys()):
            if self.info[intersection]['curr_light'] == 'EW':
                assert state[intersection]['curr_phase'] == 1 or state[intersection]['curr_phase'] == 2
            elif self.info[intersection]['curr_light'] == 'NS':
                assert state[intersection]['curr_phase'] == 0 or state[intersection]['curr_phase'] == 3
            # First get old lights to compare with new lights to see if switch is needed
            old_lights.append(self.info[intersection]['curr_light'])
            # If the next phase start step is equal to curr time then switch to next phase
            if next_phase:
                # Select random starting light if not yellow light (1 or 3)
                if state[intersection]['curr_phase'] == 0 or state[intersection]['curr_phase'] == 2:
                    choice = sample(['NS', 'EW'], 1)[0]
                else:
                    if state[intersection]['curr_phase'] == 1:
                        choice = 'EW'
                    else:
                        choice = 'NS'
                self.info[intersection]['curr_light'] = choice
                self.info[intersection]['curr_time'] = 0
            else: # o.w. continue in curr phase, switch if end of current light time
                light_lengths = self.phases[self.curr_phase_indx]['light_lengths']
                intersection_light_lengths = light_lengths[intersection]
                curr_light = self.info[intersection]['curr_light']
                curr_time = self.info[intersection]['curr_time']
                # If curr time of light is equal to its length then switch
                if curr_time >= intersection_light_lengths[curr_light]:
                    self.info[intersection]['curr_light'] = 'NS' if curr_light == 'EW' else 'EW'
                    self.info[intersection]['curr_time'] = 0
                else:
                    self.info[intersection]['curr_time'] += 1
            # Get new light
            new_lights.append(self.info[intersection]['curr_light'])
        # Compare old lights with new lights
        bin = [0 if x == y else 1 for x, y in zip(old_lights, new_lights)]
        # Convert to dec
        dec = self._convert_to_dec(bin)
        return dec

    def _get_light_lengths(self, phases):
        NODES_arr, NODES = get_node_arr_and_dict(self.net_path)
        N = len(list(NODES))
        assert N == len(set(list(NODES.keys())))

        # Matrix of edge lengths for dikstras
        M = np.zeros(shape=(N, N))

        # Need dict of edge lengths
        edge_lengths = get_edge_length_dict(self.net_path, jitter=False)

        # Loop through nodes and fill M using incLanes attrib
        # Also make dict of positions for nodes so I can work with phases
        node_locations = {}
        tree = ET.parse(self.net_path)
        root = tree.getroot()
        for c in root.iter('junction'):
            id = c.attrib['id']
            # Ignore if type is internal
            if c.attrib['type'] == 'internal':
                continue
            # Add location for future use
            node_locations[id] = (c.attrib['x'], c.attrib['y'])
            # Current col
            col = NODES[id]
            # Get the edges to this node through the incLanes attrib (remove _0 for the first lane)
            incLanes = c.attrib['incLanes']
            # Space delimited
            incLanes_list = incLanes.split(' ')
            for lane in incLanes_list:
                in_edge = lane.split('_0')[0]
                in_node = in_edge.split('___')[0]
                row = NODES[in_node]
                length = edge_lengths[in_edge]
                M[row, col] = length

        # Get the shortest path matrix
        _, preds = shortest_path(M, directed=True, return_predecessors=True)

        # Fill dict of gen{i}___rem{j} with all nodes on the shortest route
        # Todo: Important assumption, "intersection" is ONLY in the name/id of an intersection
        shortest_routes = {}
        for i in range(N):
            gen = NODES_arr[i]
            if 'intersection' in gen: continue
            for j in range(N):
                if i == j:
                    continue
                rem = NODES_arr[j]
                if 'intersection' in rem: continue
                route = [rem]
                while preds[i, j] != i:  # via
                    j = preds[i, j]
                    route.append(NODES_arr[int(j)])
                route.append(gen)
                shortest_routes['{}___{}'.format(rem, gen)] = route

        def get_weightings(probs):
            weights = {'intersectionNE': {'NS': 0., 'EW': 0.},
                       'intersectionSE': {'NS': 0., 'EW': 0.},
                       'intersectionNW': {'NS': 0., 'EW': 0.},
                       'intersectionSW': {'NS': 0., 'EW': 0.}}
            for intersection, phases in list(weights.items()):
                # Collect all routes that include the intersection
                for v in list(shortest_routes.values()):
                    if intersection in v:
                        int_indx = v.index(intersection)
                        # Route weight is gen + rem probs
                        route_weight = probs[v[0]]['gen'] + probs[v[-1]]['rem']
                        # Detect if NS or EW
                        # If xs are equal between the node before and the intersection then EW is used
                        if node_locations[v[int_indx - 1]][0] == node_locations[intersection][0]:
                            phases['NS'] += self.NS_mult * route_weight
                        else:
                            assert node_locations[v[int_indx - 1]][1] == node_locations[intersection][
                                1], 'xs not equal so ys should be.'
                            phases['EW'] += self.EW_mult * route_weight
            return weights

        def convert_weights_to_lengths(weights, cycle_length=10):
            ret_dic = {}
            for intersection, dic in list(weights.items()):
                total = dic['NS'] + dic['EW']
                NS = round((dic['NS'] / total) * cycle_length)
                EW = round((dic['EW'] / total) * cycle_length)
                assert NS >= 3 and EW >= 3, 'Assuming yellow phases are 3 make sure these lengths are not less.'
                ret_dic[intersection] = {'NS': NS, 'EW': EW}
            return ret_dic

        # For each phase get weights and cycle lengths
        new_phases = []
        curr_t = 0
        for p, phase in enumerate(phases):
            weights = get_weightings(phase['probs'])
            light_lengths = convert_weights_to_lengths(weights, self.cycle_length)
            phase_length = phase['duration'] * self.gen_time
            # Want to offset the end of each phase, except for the first phase which needs to start at the beginning
            offset = self.phase_end_offset
            if p == 0:
                offset = 0
            new_phases.append({'start_step': round(curr_t + offset), 'light_lengths': light_lengths})
            curr_t += round(phase_length)
        return new_phases
