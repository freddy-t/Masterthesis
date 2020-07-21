import agents
import torch
import copy
import time
import numpy as np
from fsc_network_env import FSCNetworkEnv
from functions import discount_rewards, create_dir
from torch.utils.tensorboard import SummaryWriter

# TODO: if gov should be active: look for "passiv" and change the lines
# TODO: so viel verallgemeinern wie möglich bei den agenten, damit z.B. unterschiedliche states und actions rauskommen können
# to run tensorboard use following cmd command:
# tensorboard --logdir=C:\Users\fredd\PycharmProjects\Masterthesis\saved_data

# runtime parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = True                # True if in debug mode
save_calc = False           # True if support and resource calculations should be saved
store_graph = False         # True if computational graph of network should be saved
MODE = 'train'              # 'train' for training mode, otherwise testing data is used

# model parameters
AGENTS = ['FSC', 'Shell', 'Gov']
CHANGEABLE_ALLOC = {'FSC':   ['Shell', 'Gov'],
                    'Shell': ['FSC'],
                    'Gov':   []}
ACT_AGT = ['FSC', 'Shell']               # set the active agents
ACTION_SPACE = [-1, 0, 1]                # action definition: -1 = decrease, 0 = maintain, 1 = increase

# parameters to evaluate
INIT_SUPPORT = [[1, 0.1, 0.1]]           # initial support by the agents, must be in order as in AGENTS
                                         # initial resource assignment       #       FSC  Shell  Gov
INIT_RESOURCE = [[0.95,   0.05, 0.00],                                       # FSC
                 [0.025,  0.95, 0.025],                                      # Shell
                 [0.05,   0.10, 0.85]]                                       # Gov
SUB_LVL = 0.05
DELTA_RESOURCE = 0.005                   # factor by which resource assignment is changed due to action
SUPPORT_FACTOR_FSC = 0.01                # factor influences how fast support is changed due to FSC interaction
SUPPORT_FACTOR_RATIO = 0.1               # ratio between support influence by agents interaction to FSC interaction

# RL parameters
LENGTH_EPISODE = 100                   # limit is 313
NUM_EPISODES = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 10
GAMMA = 0.99
SAVE_INTERVAL = 5                        # numbers of updates until data/model is saved

CONFIG = {'agents': AGENTS,
          'active_agents': ACT_AGT,
          'init_support': INIT_SUPPORT,
          'init_resource': INIT_RESOURCE,
          'sub_lvl': SUB_LVL,
          'delta_resource': DELTA_RESOURCE,
          'support_factor': SUPPORT_FACTOR_FSC,
          'ratio of support factor': SUPPORT_FACTOR_RATIO,
          'length_ep': LENGTH_EPISODE,
          'n_ep': NUM_EPISODES,
          'lr': LEARNING_RATE,
          'batch_size': BATCH_SIZE,
          'gamma': GAMMA,
          'save_interval': SAVE_INTERVAL}

# create saving directory and save config
SAVE_DIR = create_dir(DEBUG, NUM_EPISODES, LENGTH_EPISODE, LEARNING_RATE)
torch.save(CONFIG, (SAVE_DIR / 'config'))
with open((SAVE_DIR / 'config.txt'), 'w') as file:
    for key in CONFIG.keys():
        file.write(str(key) + ': ' + str(CONFIG[key]) + '\n')

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# create environment
writer = SummaryWriter(SAVE_DIR)
env = FSCNetworkEnv(AGENTS, INIT_SUPPORT, INIT_RESOURCE, SUB_LVL, LENGTH_EPISODE, DELTA_RESOURCE, SUPPORT_FACTOR_FSC,
                    SUPPORT_FACTOR_RATIO, MODE)
print('--------------------------------    ' + str(device) + '    --------------------------------')
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

# initialize agents and network optimizers and store them in dicts
optimizers = dict()
all_agents = dict()
num_states = len(AGENTS) + 1
for agt in AGENTS:
    agt_optim = dict()
    all_agents.update({agt: eval('agents.' + agt)(ACTION_SPACE, num_states, CHANGEABLE_ALLOC[agt], device)})
    # gov is passiv agent, thus need no optimizer
    if agt != 'Gov':
        networks = all_agents[agt].get_networks()
        for par_agt in networks.keys():
            agt_optim.update({par_agt: torch.optim.Adam(networks[par_agt].parameters(), lr=LEARNING_RATE)})
        optimizers.update({agt: agt_optim})

# save initial agents
torch.save(all_agents, (SAVE_DIR / 'agents_init'))

# initialise loop variables
total_rewards = {'FSC': [], 'Shell': []}
batch_returns = {'FSC': [], 'Shell': []}
batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
batch_states = {'FSC':   np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states]),
                'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states]),
                'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states])}
support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}
reward_shell_calc = np.empty([BATCH_SIZE, LENGTH_EPISODE, 4])
batch_count = 0
optim_count = 0
step_count = 0
ep = 0
states_cplt = []
times = []
# loop over all episodes (= rollouts)
while ep < NUM_EPISODES:
    ep += 1
    start_time = time.time()

    state_0_cplt = env.reset()
    states = {'FSC':   np.empty([LENGTH_EPISODE, num_states]),
              'Shell': np.empty([LENGTH_EPISODE, num_states]),
              'Gov':   np.empty([LENGTH_EPISODE, num_states])}
    rewards = {'FSC': [], 'Shell': []}
    actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
    step_actions = {}
    done = False

    # loop over episode steps
    while not done:

        for key in AGENTS:
            # extract support and resource assignment for agent as array
            state_0 = np.array(state_0_cplt[0].loc['support'][key])
            state_0 = np.append(state_0, state_0_cplt[1].loc[key])
            states[key][step_count] = state_0
            if key != 'Gov':
                # get action from each agent and store it
                step_actions.update({key: all_agents[key].get_actions(state_0)})
                for par_agt in actions[key].keys():
                    actions[key][par_agt].append(copy.copy(step_actions[key][par_agt]))

        # perform action on environment
        if save_calc:
            state_1_cplt, r1, done, sup_calc, r_shell = env.step_calc(step_actions)
            for key in support_calc.keys():
                support_calc[key][batch_count][step_count] = sup_calc[key]
            reward_shell_calc[batch_count][step_count] = r_shell
        else:
            state_1_cplt, r1, done = env.step(step_actions)

        # store rewards
        for agt in ACT_AGT:
            rewards[agt].append(r1[agt])

        # store and update old state
        states_cplt.append(state_0_cplt)
        state_0_cplt = state_1_cplt
        step_count += 1

        # if done (= new rollout complete), data is put into the batch
        if done:
            step_count = 0
            # put data in batch
            for agt in AGENTS:
                batch_states[agt][batch_count] = states[agt]
                if agt != 'Gov':
                    batch_returns[agt].extend(discount_rewards(rewards[agt], GAMMA))
                    for par_agt in actions[agt].keys():
                        batch_actions[agt][par_agt].extend(actions[agt][par_agt])
                    total_rewards[agt].append(np.array(rewards[agt]).sum())

                # write rewards to tensorboard
                writer.add_scalar('reward_FSC', np.array(rewards['FSC']).sum(), ep)
                writer.add_scalar('reward_Shell', np.array(rewards['Shell']).sum(), ep)

            if save_calc:
                for i in range(reward_shell_calc[batch_count].shape[1]):
                    reward_shell_calc[batch_count][:, i] = discount_rewards(reward_shell_calc[batch_count][:, i], GAMMA)

            batch_count += 1

            # if batch full (= enough rollouts for optimization), perform optimization on networks
            if batch_count == BATCH_SIZE:

                optim_count += 1
                # loop over all agents and over the changeable allocation
                for agt in ACT_AGT:
                    for par_agt in optimizers[agt].keys():

                        # set gradients of all optimizers to zero
                        optimizers[agt][par_agt].zero_grad()

                        # create tensors for calculation
                        reward_tensor = torch.FloatTensor(batch_returns[agt]).to(device)

                        # create Long Tensor and add +1 to use action_tensor as indices
                        action_tensor = torch.LongTensor(np.array(batch_actions[agt][par_agt])+1).to(device)

                        # calculate the loss
                        logprob = torch.log(all_agents[agt].predict(
                            batch_states[agt].reshape([BATCH_SIZE*LENGTH_EPISODE, num_states]), par_agt)).to(device)
                        # use torch.gather to get the logprob to the corresponding action
                        selected_logprobs = reward_tensor * torch.gather(logprob, 1,
                                                                         action_tensor.unsqueeze(1)).squeeze().to(device)
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()

                        # Apply gradients
                        optimizers[agt][par_agt].step()

                        # write loss to tensorboard after each optimization step
                        writer.add_scalar('mean_loss_' + agt + '_' + par_agt, loss, optim_count)

                        # save weights and its gradients for tensorboard after SAVE_INTERVAL number of
                        # optimization steps
                        # if optim_count % SAVE_INTERVAL == 0:

                        for name, weight in all_agents[agt].get_networks()[par_agt].named_parameters():
                            writer.add_histogram(agt + '_' + par_agt + '_' + name, weight, optim_count)
                            writer.add_histogram(agt + '_' + par_agt + '_' + name + '_grad', weight.grad, optim_count)

                # save agents, all states and rewards after SAVE_INTERVAL number of optimization steps
                if optim_count % SAVE_INTERVAL == 0 or optim_count:
                    torch.save(all_agents, (SAVE_DIR / 'agents_optim{}_ep{}'.format(optim_count, ep)))
                    torch.save(total_rewards, (SAVE_DIR / 'tot_r'))
                    torch.save(batch_states, (SAVE_DIR / 'batch_states_optim{}'.format(optim_count)))
                    if save_calc:
                        torch.save(support_calc, (SAVE_DIR / 'support_calc_optim{}'.format(optim_count)))
                        torch.save(reward_shell_calc, (SAVE_DIR / 'reward_shell_calc_optim{}'.format(optim_count)))

                # empty batch
                batch_returns = {'FSC': [], 'Shell': []}
                batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
                batch_states = {'FSC':   np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states]),
                                'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states]),
                                'Gov':   np.empty([BATCH_SIZE, LENGTH_EPISODE, num_states])}
                batch_count = 0
                if save_calc:
                    support_calc = {'Shell': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1]),
                                    'Gov': np.empty([BATCH_SIZE, LENGTH_EPISODE, len(AGENTS) + 1])}
                    reward_shell_calc = np.empty([BATCH_SIZE, LENGTH_EPISODE, 4])
                print('-------------------------------     Update step completed.     -------------------------------')

    # take time and save it
    times.append(time.time() - start_time)
    torch.save(times, (SAVE_DIR / 'running_times'))

    # Print moving average
    if ep % 10 == 0:
        print('Episode {} complete. Avg time of last 10: {:.3f} sec. Average reward of last 100:'
              ' FSC: {:.3f}, Shell: {:.3f}'.format(ep, np.mean(times[-10:]), np.mean(total_rewards['FSC'][-100:]),
                                                   np.mean(total_rewards['Shell'][-100:])))

    # save a graph to tensorboard (just for visualisation)
    if store_graph:
        store_graph = False
        writer.add_graph(all_agents['FSC'].get_networks()['Shell'], torch.FloatTensor(states['FSC'][0]))

    # flush the writer, so that everything is written to tensorboard
    writer.flush()

writer.close()
print('Total time: {:.3f}'.format(np.sum(times)))
