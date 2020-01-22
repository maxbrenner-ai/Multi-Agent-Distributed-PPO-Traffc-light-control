import os
import sys
from workers.ppo_worker import PPOWorker
from workers.rule_worker import RuleBasedWorker
from models.ppo_model import NN_Model
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from copy import deepcopy


def test_worker(worker):
    worker.eval_episode({})
    # Kill connection to sumo server
    worker.env.connection.close()

def test_PPO_agent(constants, device, loaded_model):
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, constants)
    env = IntersectionsEnv(constants, device, 'vis_ppo', True, get_net_path(constants), vis=True)
    local_NN = NN_Model(s_a['s'], s_a['a'], device).to(device)
    local_NN.load_state_dict(loaded_model)
    worker = PPOWorker(constants, device, env, None, None, local_NN, None, 'ppo', dont_reset=True)
    test_worker(worker)

# verbose means test prints at end of each batch of eps
def test_rule_based_agent(constants, device):
    env = IntersectionsEnv(constants, device, 'vis_rule_based', True, get_net_path(constants), vis=True)
    # Check rule set
    rule_set_class = get_rule_set_class(constants['rule']['rule_set'])
    rule_set_params = deepcopy(constants['rule']['rule_set_params'])
    rule_set_params['phases'] = env.phases
    worker = RuleBasedWorker(constants, device, env, rule_set_class(rule_set_params, get_net_path(constants), constants), None, 'rule_based')
    test_worker(worker)

def run(load_model_file=None):
    # Load constants
    constants = load_constants('constants/constants.json')

    loaded_model = None
    if load_model_file:
        loaded_model = torch.load('models/saved_models/' + load_model_file)

    if loaded_model:
        test_PPO_agent(constants, device, loaded_model)
    else:
        assert constants['agent']['agent_type'] == 'rule'
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
