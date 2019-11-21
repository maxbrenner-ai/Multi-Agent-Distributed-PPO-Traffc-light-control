import time

from PPO_agent import PPOAgent
from utils import *
from model import NN_Model
from environment import Environment


# target for multiprocess
def train(id, shared_NN, optimizer, rollout_counter, args):
    episode_C, model_C, agent_C, other_C, device = args
    local_NN = NN_Model(model_C['state_size'], model_C['action_size'], device).to(device)
    agent = PPOAgent(args, Environment(args, id, eval_agent=False), shared_NN, local_NN, optimizer, id)
    train_step = 0
    r = 0
    while True:
        agent.train_rollout(train_step)
        r += 1
        rollout_counter.increment()
        if rollout_counter.get() >= episode_C['num_train_rollouts'] + 1:
            break
    # Kill connection to sumo server
    agent.env.connection.close()
    print('...Training agent {} done'.format(id))


# target for multiprocess
def eval(shared_NN, rollout_counter, args, df, verbose):
    time_start = time.time()
    episode_C, model_C, agent_C, other_C, device = args
    local_gnn = NN_Model(model_C['state_size'], model_C['action_size'], device).to(device)
    id = 'eval'
    agent = PPOAgent(args, Environment(args, id, eval_agent=True), shared_NN, local_gnn, None, id)
    last_eval = 0
    run_info = {}
    run_info['eval_avg_rew'] = -1.
    run_info['eval_avg_waiting_time'] = float('inf')
    while True:
        curr_r = rollout_counter.get()
        if curr_r % episode_C['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            avg_rew, avg_waiting_time = agent.eval_episodes()
            if verbose:
                print('Eval summary at rollout {}: Avg ep rew: {:.2f}  Avg ep waiting time: {:.2f}'
                      .format(curr_r, avg_rew, avg_waiting_time))
            # Only save best run so far based on a single (the most important) value
            if avg_waiting_time < run_info['eval_avg_waiting_time']:
                run_info['eval_avg_waiting_time'] = avg_waiting_time
                run_info['eval_avg_rew'] = avg_rew
        # End the eval agent
        if curr_r >= episode_C['num_train_rollouts'] + 1:
            if df is not None:
                def add_hyp_param_dict(append_letter, dic):
                    for k, v in list(dic.items()):
                        run_info[append_letter + '_' + k] = v
                add_hyp_param_dict('E', episode_C)
                add_hyp_param_dict('M', model_C)
                add_hyp_param_dict('A', agent_C)
                add_hyp_param_dict('O', other_C)
                time_end = time.time()
                time_taken = (time_end - time_start) / 60.
                run_info['total_time_taken(m)'] = time_taken
                df = df.append(run_info, ignore_index=True)
                df.to_excel('run-data.xlsx', index=False)
                print(
                    'Best run summary: Avg ep rew: {:.2f}  Avg ep waiting time: {:.2f}'.format(
                        run_info['eval_avg_rew'], run_info['eval_avg_waiting_time']))
            break
    # Kill connection to sumo server
    agent.env.connection.close()
    print('...Eval agent done')


# verbose means eval prints at end of each batch of eps
def run_agent(episode_C, model_C, agent_C, other_C, device, df, verbose):
    shared_NN = NN_Model(model_C['state_size'], model_C['action_size'], device).to(device)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), agent_C['learning_rate'])
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []
    args = (episode_C, model_C, agent_C, other_C, device)
    # Run eval agent
    p = mp.Process(target=eval, args=(shared_NN, rollout_counter, args, df, verbose))
    p.start()
    processes.append(p)
    # Run training agents
    for i in range(other_C['num_agents']):
        p = mp.Process(target=train, args=(i, shared_NN, optimizer, rollout_counter, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
