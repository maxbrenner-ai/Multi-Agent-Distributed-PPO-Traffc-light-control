import time
from utils import *
from run_agent_parallel import run_PPO_agent, run_rule_based_agent
import sys
import os


def run_normal(verbose, num_experiments=1, use_DF=False):
    # Load constants
    constants = load_constants('constants.json')
    episode_C, model_C, agent_C, other_C = constants['episode_C'], constants['model_C'], \
                                                   constants['agent_C'], constants['other_C']
    for exp in range(num_experiments):
        print(' --- Running experiment {} --- '.format(exp))

        df = None
        if use_DF:
            df = pd.read_excel('run-data.xlsx')

        exp_start = time.time()

        if not other_C['rule_set']:
            run_PPO_agent(episode_C, model_C, agent_C, other_C, device, df, verbose)
        else:
            run_rule_based_agent(episode_C, model_C, agent_C, other_C, device, df, verbose)

        exp_end = time.time()
        print(' - Time taken (m): {:.2f} - '.format((exp_end - exp_start) / 60.))


if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    excel_header = ['total_time_taken(m)', 'eval_avg_rew', 'eval_avg_waiting_time', 'E_num_train_rollouts',
                    'E_rollout_length', 'E_eval_freq', 'E_eval_num_eps', 'E_max_ep_steps', 'E_test_num_eps', 'A_gae_tau',
                    'A_entropy_weight', 'A_minibatch_size', 'A_optimization_epochs', 'A_ppo_ratio_clip',
                    'A_discount', 'A_learning_rate', 'A_clip_grads', 'A_gradient_clip', 'A_value_loss_coef',
                    'O_num_agents', 'O_rule_set', 'O_rule_set_params']

    # df_path = 'run-data.xlsx'

    # print('Num cores: {}'.format(mp.cpu_count()))

    refresh_excel('run-data.xlsx', excel_header)

    # Run given constants file
    run_normal(verbose=True, num_experiments=1, use_DF=True)

    # Shows how to run multiple random search experiments
    # run_random_search(num_diff_experiments=100, num_repeat_experiment=3, df_path=df_path)

    # Shows how to run a grid search expr, that grid searches a single hypparam
    # refresh_excel(df_path)
    # run_grid_search_single_variable(num_repeat_experiment=10, param_set='agent', var='critic_agg_weight',
    #                                 values=[0.0, 0.25, 0.5, 0.75, 1.0], df_path=df_path)

