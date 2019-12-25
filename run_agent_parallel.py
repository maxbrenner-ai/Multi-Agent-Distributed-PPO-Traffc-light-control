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


# target for multiprocess
def train(id, shared_NN, data_collector, optimizer, rollout_counter, constants, device):
    # Assumes PPO agent
    env_id = constants['other_C']['environment']
    local_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
    agent = PPOAgent(constants, device, get_env(env_id)(constants, device, id, eval_agent=False), None, shared_NN, local_NN, optimizer, id)
    train_step = 0
    while rollout_counter.get() < constants['episode_C']['num_train_rollouts'] + 1:
        agent.train_rollout(train_step)
        rollout_counter.increment()
    # Kill connection to sumo server
    agent.env.connection.close()
    # print('...Training agent {} done'.format(id))


# target for multiprocess
def eval(id, shared_NN, data_collector, rollout_counter, constants, device):
    # Assumes PPO agent
    env_id = constants['other_C']['environment']
    local_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
    agent = PPOAgent(constants, device, get_env(env_id)(constants, device, id, eval_agent=True), data_collector, shared_NN, local_NN, None, id)
    last_eval = 0
    while True:
        curr_r = rollout_counter.get()
        if curr_r % constants['episode_C']['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            agent.eval_episodes(curr_r, model_state=agent.NN.state_dict())
        # End the eval agent
        if curr_r >= constants['episode_C']['num_train_rollouts'] + 1:
            break
    # Eval at end
    agent.eval_episodes(curr_r, model_state=agent.NN.state_dict())
    # Kill connection to sumo server
    agent.env.connection.close()
    # print('...Eval agent {} done'.format(id))


# target for multiprocess
def test(id, ep_counter, constants, device, agent=None, data_collector=None, shared_NN=None):
    # assume PPO agent if agent=None
    if not agent:
        env_id = constants['other_C']['environment']
        local_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
        agent = PPOAgent(constants, device, get_env(env_id)(constants, device, id, eval_agent=True), data_collector,
                         shared_NN, local_NN, None, id)
    while ep_counter.get() < constants['episode_C']['test_num_eps']:
        agent.eval_episodes(None, ep_count=ep_counter.get())
        ep_counter.increment(constants['episode_C']['eval_num_eps'])
    # Kill connection to sumo server
    agent.env.connection.close()
    # print('...Testing agent {} done'.format(id))

# ======================================================================================================================

def train_PPO_agent(constants, device, data_collector):
    env_id = constants['other_C']['environment']
    shared_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
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

def test_PPO_agent(constants, device, data_collector, loaded_model):
    env_id = constants['other_C']['environment']
    shared_NN = NN_Model(get_state_size(env_id), get_action_size(env_id), device).to(device)
    shared_NN.load_state_dict(loaded_model)
    shared_NN.share_memory()
    ep_counter = Counter()  # Eps across all agents
    processes = []
    for i in range(constants['other_C']['num_agents']):
        id = 'test_'+str(i)
        p = mp.Process(target=test, args=(id, ep_counter, constants, device, None, data_collector, shared_NN))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# verbose means test prints at end of each batch of eps
def test_rule_based_agent(constants, device, data_collector):
    # Check rule set
    rule_set_class = get_rule_set_class(constants['other_C']['rule_set'])
    ep_counter = Counter()  # Eps across all agents
    processes = []
    for i in range(constants['other_C']['num_agents']):
        id = 'test_'+str(i)
        env = get_env(constants['other_C']['environment'])(constants, device, id, eval_agent=True)
        rule_set_params = deepcopy(constants['other_C']['rule_set_params'])
        rule_set_params['phases'] = env.phases
        net_path = None
        if constants['other_C']['rule_set'] == 'multi_intersection':
            net_path = 'data/four_intersection.net.xml'
        agent = RuleBasedAgent(constants, device, env, rule_set_class(rule_set_params, net_path, constants), data_collector, id)
        p = mp.Process(target=test, args=(id, ep_counter, constants, device, agent, None, None))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
