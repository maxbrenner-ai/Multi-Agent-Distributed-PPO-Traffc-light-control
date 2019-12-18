import os
import sys

from agents.PPO_agent import PPOAgent
from agents.rule_based_agent import RuleBasedAgent
from environments.single_intersection import SingleIntersectionEnv as Environment
from models.ppo_model import NN_Model
from utils.utils import *


def test_agent(agent):
    agent.eval_episode({})
    # Kill connection to sumo server
    agent.env.connection.close()

def test_PPO_agent(constants, device, loaded_model):
    local_NN = NN_Model(constants['model_C']['state_size'], constants['model_C']['action_size'], device).to(device)
    local_NN.load_state_dict(loaded_model)
    agent = PPOAgent(constants, device, Environment(constants, device, 'vis_ppo', eval_agent=True, vis=True), None, None,
                     local_NN, None, 'ppo', dont_reset=True)
    test_agent(agent)

# verbose means test prints at end of each batch of eps
def test_rule_based_agent(constants, device):
    # Check rule set
    rule_set = rule_set_creator(constants['other_C']['rule_set'], constants['other_C']['rule_set_params'])
    agent = RuleBasedAgent(constants, device, Environment(constants, device, 'vis_rule_based', eval_agent=True, vis=True), rule_set,
                           None, 'rule_based')
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

    run(load_model_file='grid_3-3.pt')
