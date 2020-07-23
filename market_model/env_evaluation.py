import torch
import copy
import time
import os
import random
import numpy as np
from agents import init_agents
from fsc_network_env import FSCNetworkEnv
from functions import create_val_dir

# runtime parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = True             # True if in debug mode
num_samples = 3

# model parameters
AGENTS = ['FSC', 'Shell', 'Gov']
CHANGEABLE_ALLOC = {'FSC':   ['Shell', 'Gov'],
                    'Shell': ['FSC'],
                    'Gov':   []}
ACT_AGT = ['FSC', 'Shell']               # set the active agents
ACTION_SPACE = [0, 1, 2]                # action definition: 0 = decrease, 1 = maintain, 2 = increase
N_STATE_SPACE = {'FSC': 4,
                 'Shell': 4,
                 'Gov': 4}
INIT_SUPPORT = [[1, 0.1, 0.1]]           # initial support by the agents, must be in order as in AGENTS
                                         # initial resource assignment       #       FSC  Shell  Gov
INIT_RESOURCE = [[0.95,   0.05, 0.00],                                       # FSC
                 [0.025,  0.95, 0.025],                                      # Shell
                 [0.05,   0.10, 0.85]]                                       # Gov
SUB_LVL = 0.05                           # level of subsidy

# RL parameters
LENGTH_EPISODE = 157                   # limits are 1 -> 313, 2 -> 157, 3 -> 105, 4 -> 79
NUM_EPISODES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = NUM_EPISODES
GAMMA = 0.99
SAVE_INTERVAL = 100                        # numbers of updates until data/model is saved

# define ranges for parameters
factor = 1000
range_delta_resource = np.array([0.001, 0.02]) * factor
range_betas = np.array([0.001, 0.1]) * factor

SAVE_DIR = create_val_dir(DEBUG)
env = FSCNetworkEnv(AGENTS, INIT_SUPPORT, INIT_RESOURCE, SUB_LVL, LENGTH_EPISODE, None, None, N_STATE_SPACE)
times = []
for sample_step in range(num_samples):
    start_time = time.time()

    # evaluation parameters
    # factor by which resource assignment is changed due to action
    delta_resource = random.randrange(range_delta_resource[0], range_delta_resource[1]+1) / factor
    beta_j = random.randrange(range_betas[0], range_betas[1]+1) / factor
    beta_fsc = random.randrange(range_betas[0], range_betas[1]+1) / factor
    betas = {'FSC': beta_fsc,  # factor influences how fast support is changed due to FSC interaction
             'Shell': beta_j,
             'Gov': beta_j}  # factor influences how fast support is changed due to FSC interaction
    ACTIONS = {'FSC': {'Shell': 0,
                       'Gov': 0},
               'Shell': {'FSC': 0}
               }

    CONFIG = {'agents': AGENTS,
              'active_agents': ACT_AGT,
              'init_support': INIT_SUPPORT,
              'init_resource': INIT_RESOURCE,
              'sub_lvl': SUB_LVL,
              'delta_resource': delta_resource,
              'support_factor': betas,
              'length_ep': LENGTH_EPISODE,
              'n_ep': NUM_EPISODES,
              'lr': LEARNING_RATE,
              'batch_size': BATCH_SIZE,
              'gamma': GAMMA,
              'save_interval': SAVE_INTERVAL}

    # create saving directory and save config
    sample_dir = SAVE_DIR / ('sample_no' + str(sample_step))
    os.mkdir(sample_dir)
    torch.save(CONFIG, (sample_dir / 'config'))
    with open((sample_dir / 'config.txt'), 'w') as file:
        for key in CONFIG.keys():
            file.write(str(key) + ': ' + str(CONFIG[key]) + '\n')

    # set missing parameters for environment
    env.set(delta_resource, betas)

    # initialize agents and network optimizers and store them in dicts
    _, all_agents = init_agents(ACTION_SPACE, N_STATE_SPACE, CHANGEABLE_ALLOC, ACT_AGT, LEARNING_RATE, device)

    # initialise loop variables
    batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
    batch_states = {'FSC':   np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                    'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                    'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
    support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                    'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}
    batch_count = 0
    step_count = 0
    ep = 0
    # loop over all episodes (= rollouts)
    while ep < NUM_EPISODES:
        ep += 1

        state_0 = env.reset()
        states = {'FSC':   np.empty([LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                  'Shell': np.empty([LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                  'Gov':   np.empty([LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
        rewards = {'FSC': [], 'Shell': []}
        actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
        step_actions = {}
        done = False

        # loop over episode steps
        while not done:

            # get action from each agent and store it
            for key in ACT_AGT:
                if 'ACTIONS' in locals():
                    step_actions = ACTIONS
                else:
                    step_actions.update({key: all_agents[key].get_actions(state_0[key])})
                for par_agt in actions[key].keys():
                    actions[key][par_agt].append(copy.copy(step_actions[key][par_agt]))

            # perform action on environment
            state_1, r1, done, sup_calc, r_shell = env.step_calc(step_actions)
            for key in support_calc.keys():
                support_calc[key][batch_count][step_count] = sup_calc[key]

            # store rewards
            for agt in ACT_AGT:
                rewards[agt].append(r1[agt])

            # update old state and save current state
            for key in AGENTS:
                states[key][step_count] = state_0[key]
            state_0 = state_1
            step_count += 1

            # if done (= new rollout complete), data is put into the batch
            if done:
                step_count = 0
                # put data in batch
                for agt in AGENTS:
                    batch_states[agt][batch_count] = states[agt]
                    if agt != 'Gov':
                        for par_agt in actions[agt].keys():
                            batch_actions[agt][par_agt].extend(actions[agt][par_agt])

                batch_count += 1
                # if batch is full, save batch data and empty it
                if batch_count == BATCH_SIZE:
                    batch_count = 0
                    torch.save(batch_states, (sample_dir / 'batch_states'))
                    torch.save(batch_actions, (sample_dir / 'batch_actions'))
                    torch.save(support_calc, (sample_dir / 'support_calc'))

                    batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
                    batch_states = {'FSC': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['FSC']]),
                                    'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Shell']]),
                                    'Gov': np.empty([BATCH_SIZE, LENGTH_EPISODE, N_STATE_SPACE['Gov']])}
                    support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                                    'Gov': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}

    # take time and save it
    times.append(time.time() - start_time)
    torch.save(times, (sample_dir / 'running_times'))

    # Print moving average
    print('Sample {} complete. Avg time of last 10: {:.3f} sec.'.format(sample_step, np.mean(times[-10:])))

print('Total time for {} samples: {:.3f}'.format(sample_step+1, np.sum(times)))
