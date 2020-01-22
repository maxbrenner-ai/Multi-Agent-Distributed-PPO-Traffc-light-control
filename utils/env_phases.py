from utils.net_scrape import *


def get_phases(environment_C, net_path):
    if environment_C['rush_hour']:
        if environment_C['shape'] == [2, 2]: return two_two_intersections_rush_hour()
        else: raise AssertionError('Wrong env given for rush hour.')
    else:
        return uniform_intersections(environment_C['uniform_generation_probability'], net_path)

# Normalize all rem probs for each phase to sum to 1
def normalize(dic):
    total = sum([v['rem'] for v in list(dic.values())])
    for v in list(dic.values()):
        v['rem'] /= total
    return dic

def uniform_intersections(gen, net_path):
    node_arr = get_non_intersections(net_path)
    probs = {}
    for node in node_arr:
        probs[node] = {'gen': gen, 'rem': 1}
    phase = [{'duration': 1.0, 'probs': normalize(probs)}]
    return phase

def two_two_intersections_rush_hour():
    high_rush_hour_prob_gen = 0.2  # When an edge is generating a bunch of vehicles for rush hour
    low_rush_hour_prob_gen = 0.01  # When an edge is on the removal side of rush hour
    side_high_rush_hour_prob_gen = 0.05  # When a side edge is generating for rush hour
    side_low_rush_hour_prob_gen = 0.01  # When a side edge is removing for rush hour
    non_rush_hour_prob_gen = 0.1  # For all edges when it isn't rush hour

    high_rush_hour_prob_rem = 0.5
    non_rush_hour_prob_rem = 0.1
    side_high_rush_hour_prob_rem = 0.2

    first_rush_hour_probs = {'northLeft': {'gen': low_rush_hour_prob_gen, 'rem': high_rush_hour_prob_rem},
                             'northRight': {'gen': low_rush_hour_prob_gen, 'rem': high_rush_hour_prob_rem},
                             'southLeft': {'gen': high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'southRight': {'gen': high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'westTop': {'gen': side_low_rush_hour_prob_gen, 'rem': side_high_rush_hour_prob_rem},
                             'eastTop': {'gen': side_low_rush_hour_prob_gen, 'rem': side_high_rush_hour_prob_rem},
                             'westBottom': {'gen': side_high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'eastBottom': {'gen': side_high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem}}

    non_rush_hour_probs = {'northLeft': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'northRight': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'southLeft': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'southRight': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'westTop': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'eastTop': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'westBottom': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'eastBottom': {'gen': non_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem}}

    second_rush_hour_probs = {'northLeft': {'gen': high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'northRight': {'gen': high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'southLeft': {'gen': low_rush_hour_prob_gen, 'rem': high_rush_hour_prob_rem},
                             'southRight': {'gen': low_rush_hour_prob_gen, 'rem': high_rush_hour_prob_rem},
                             'westTop': {'gen': side_high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'eastTop': {'gen': side_high_rush_hour_prob_gen, 'rem': non_rush_hour_prob_rem},
                             'westBottom': {'gen': side_low_rush_hour_prob_gen, 'rem': side_high_rush_hour_prob_rem},
                             'eastBottom': {'gen': side_low_rush_hour_prob_gen, 'rem': side_high_rush_hour_prob_rem}}

    # Phase durations are a proportion of the total generation time
    phases = [{'duration': 0.25, 'probs': normalize(first_rush_hour_probs)},
              {'duration': 0.5, 'probs': normalize(non_rush_hour_probs)},
              {'duration': 0.25, 'probs': normalize(second_rush_hour_probs)}]

    assert sum([p['duration'] for p in phases]) == 1.

    return phases


# Given t figure out which phase its in
def get_current_phase_probs(t, phases, gen_time):
    curr_start = 0
    for p in phases:
        dur = p['duration'] * gen_time
        if curr_start <= t < dur + curr_start:
            return p['probs']
        curr_start += dur
    raise AssertionError()
