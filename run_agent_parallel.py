import time
from agents.PPO_agent import PPOAgent
from agents.rule_based_agent import RuleBasedAgent
from environment import Environment
from models.ppo_model import NN_Model
from utils.utils import *


# target for multiprocess
def train(id, shared_NN, data_collector, optimizer, rollout_counter, constants, device):
    # Assumes PPO agent
    local_NN = NN_Model(constants['model_C']['state_size'], constants['model_C']['action_size'], device).to(device)
    agent = PPOAgent(constants, device, Environment(constants, device, id, eval_agent=False), None, shared_NN, local_NN, optimizer, id)
    train_step = 0
    while rollout_counter.get() < constants['episode_C']['num_train_rollouts'] + 1:
        agent.train_rollout(train_step)
        rollout_counter.increment()
    # Kill connection to sumo server
    agent.env.connection.close()
    print('...Training agent {} done'.format(id))


# target for multiprocess
def eval(id, shared_NN, data_collector, rollout_counter, constants, device):
    # Assumes PPO agent
    local_NN = NN_Model(constants['model_C']['state_size'], constants['model_C']['action_size'], device).to(device)
    agent = PPOAgent(constants, device, Environment(constants, device, id, eval_agent=True), data_collector, shared_NN, local_NN, None, id)
    last_eval = 0
    while True:
        curr_r = rollout_counter.get()
        if curr_r % constants['episode_C']['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            agent.eval_episodes(curr_r)
        # End the eval agent
        if curr_r >= constants['episode_C']['num_train_rollouts'] + 1:
            break
    # Kill connection to sumo server
    agent.env.connection.close()
    print('...Eval agent {} done'.format(id))


# target for multiprocess
def test(id, ep_counter, constants, device, agent=None):
    while ep_counter.get() < constants['episode_C']['test_num_eps']:
        agent.eval_episodes(None, ep_count=ep_counter.get())
        ep_counter.increment(constants['episode_C']['eval_num_eps'])
    # Kill connection to sumo server
    agent.env.connection.close()
    print('...Testing agent {} done'.format(id))

# ======================================================================================================================

def run_PPO_agent(constants, device, data_collector):
    shared_NN = NN_Model(constants['model_C']['state_size'], constants['model_C']['action_size'], device).to(device)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), constants['agent_C']['learning_rate'])
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []
    # Run eval agent
    id = 'eval_0'
    p = mp.Process(target=eval, args=(id, shared_NN, data_collector, rollout_counter, constants, device))
    p.start()
    processes.append(p)
    # Run training agents
    for i in range(constants['other_C']['num_agents']):
        id = 'train_'+str(i)
        p = mp.Process(target=train, args=(id, shared_NN, data_collector, optimizer, rollout_counter, constants, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# verbose means test prints at end of each batch of eps
def run_rule_based_agent(constants, device, data_collector):
    # Check rule set
    rule_set = rule_set_creator(constants['other_C']['rule_set'], constants['other_C']['rule_set_params'])
    ep_counter = Counter()  # Eps across all agents
    processes = []
    for i in range(constants['other_C']['num_agents']):
        id = 'test_'+str(i)
        agent = RuleBasedAgent(constants, device, Environment(constants, device, id, eval_agent=True), rule_set, data_collector, id)
        p = mp.Process(target=test, args=(id, ep_counter, constants, device, agent))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
