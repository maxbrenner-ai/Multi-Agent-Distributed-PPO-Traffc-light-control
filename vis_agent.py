import os
import sys
from agents.PPO_agent import PPOAgent
from agents.rule_based_agent import RuleBasedAgent
from models.ppo_model import NN_Model
from utils.utils import *
from environments.single_intersection import SingleIntersectionEnv
from environments.single_intersection import STATE_SIZE as SI_STATE_SIZE
from environments.single_intersection import ACTION_SIZE as SI_ACTION_SIZE
from environments.four_intersections import FourIntersectionEnv
from environments.four_intersections import STATE_SIZE as FI_STATE_SIZE
from environments.four_intersections import ACTION_SIZE as FI_ACTION_SIZE
from copy import deepcopy


def get_env(env_id):
    if env_id == 'single_intersection': return SingleIntersectionEnv
    elif env_id == 'four_intersections': return FourIntersectionEnv
    raise AssertionError('Intersection id in constants is wrong.')


def get_state_size(env_id):
    if env_id == 'single_intersection': return SI_STATE_SIZE
    elif env_id == 'four_intersections': return FI_STATE_SIZE
    raise AssertionError('Intersection id in constants is wrong.')


def get_action_size(env_id):
    if env_id == 'single_intersection': return SI_ACTION_SIZE
    elif env_id == 'four_intersections': return FI_ACTION_SIZE
    raise AssertionError('Intersection id in constants is wrong.')


def test_agent(agent):
    agent.eval_episode({})
    # Kill connection to sumo server
    agent.env.connection.close()

def test_PPO_agent(constants, device, loaded_model):
    env_id = constants['other_C']['environment']
    env = get_env(env_id)(constants, device, 'vis_ppo', eval_agent=True, vis=True)
    local_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
    local_NN.load_state_dict(loaded_model)
    agent = PPOAgent(constants, device, env, None, None, local_NN, None, 'ppo', dont_reset=True)
    test_agent(agent)

# verbose means test prints at end of each batch of eps
def test_rule_based_agent(constants, device):
    env = get_env(constants['other_C']['environment'])(constants, device, 'vis_rule_based', eval_agent=True, vis=True)
    # Check rule set
    rule_set_class = get_rule_set_class(constants['other_C']['rule_set'])
    rule_set_params = deepcopy(constants['other_C']['rule_set_params'])
    rule_set_params['phases'] = env.phases
    net_path = None
    if constants['other_C']['rule_set'] == 'multi_intersection':
        net_path = 'data/four_intersection.net.xml'
    agent = RuleBasedAgent(constants, device, env, rule_set_class(rule_set_params, net_path, constants), None, 'rule_based')
    test_agent(agent)

def run(load_model_file=None):
    # Load constants
    constants = load_constants('constants/constants.json')

    loaded_model = None
    if load_model_file:
        loaded_model = torch.load('models/saved_models/' + load_model_file)

    if loaded_model:
        assert not constants['other_C']['rule_set']
        test_PPO_agent(constants, device, loaded_model)
    else:
        test_rule_based_agent(constants, device)

if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    run(load_model_file=None)
