import xml.etree.ElementTree as ET
import random
import numpy as np


# Since tie breaks for shortest path will result in deterministic behavior, i can make it nondeterministic better with jitter
def get_edge_length_dict(filepath, jitter=False):
    edges = {}
    # Parse through .net.xml file for lengths
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('edge'):
        id = c.attrib['id']
        # If function key in attib then continue
        if 'function' in c.attrib:
            continue
        length = float(c[0].attrib['length'])
        if jitter:
            length += random.randint(-1, 1)
        edges[id] =  length # ONLY WORKS FOR SINGLE LANE ROADS
    return edges


# Make a dict {external node: {'gen': gen edge for node, 'rem': rem edge for node} ... } for external nodes
# For adding trips
def get_node_edge_dict(filepath):
    dic = {}
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or 'intersection' in node:
            continue
        incLane = c.attrib['incLanes'].split('_0')[0]
        dic[node] = {'gen': incLane.split('___')[1] + '___' + incLane.split('___')[0], 'rem': incLane}
    return dic


def get_intersections(filepath):
    arr = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal' or not 'intersection' in node:
            continue
        arr.append(node)
    return arr


# Returns a arr of the nodes, and a dict where the key is a node and value is the index in the arr
def get_node_arr_and_dict(filepath):
    arr = []
    dic = {}
    i = 0
    tree = ET.parse(filepath)
    root = tree.getroot()
    for c in root.iter('junction'):
        node = c.attrib['id']
        if c.attrib['type'] == 'internal':
            continue
        arr.append(node)
        dic[node] = i
        i += 1
    return arr, dic


# For calc. global/localized discounted rewards
# {intA: {intA: 0, intB: 5, ...} ...}
def get_cartesian_intersection_distances(filepath):
    # First gather the positions
    tree = ET.parse(filepath)
    root = tree.getroot()
    intersection_positions = {}
    for c in root.iter('junction'):
        node = c.attrib['id']
        # If not intersection then continue
        if c.attrib['type'] == 'internal' or not 'intersection' in node:
            continue
        x = float(c.attrib['x'])
        y = float(c.attrib['y'])
        intersection_positions[node] = np.array([x, y])
    # Second calc. euclidean distances
    distances = {}
    for outer_k, outer_v in list(intersection_positions.items()):
        distances[outer_k] = {}
        for inner_k, inner_v in list(intersection_positions.items()):
            dist = np.linalg.norm(outer_v - inner_v)
            distances[outer_k][inner_k] = dist
    return distances