import gym
import agents
import torch
import copy
import time
import numpy as np
import gym_FSC_network
from functions import discount_rewards, create_dir, save_obj
from torch.utils.tensorboard import SummaryWriter

# TODO: ?resource and support initialize in env and check if keys are matching
# TODO: if gov should be active: look for "passiv" and change the lines
# to run tensorboard use following cmd command:
# tensorboard --logdir=C:\Users\fredd\PycharmProjects\Masterthesis\saved_data

AGENTS = ['FSC', 'Shell', 'Gov']
CHANGEABLE_ALLOC = {'FSC':   ['Shell', 'Gov'],
                    'Shell': ['FSC'],
                    'Gov':   []}
ACT_AGT = ['FSC', 'Shell']               # set the active agents
ACTION_SPACE = [-1, 0, 1]                # action definition: -1 = decrease, 0 = maintain, 1 = increase
INIT_SUPPORT = [[1, 0.1, 0.1]]           # initial support by the agents, must be in order as in AGENTS

# initial resource assignment            #       FSC  Shell  Gov
INIT_RESOURCE = [[0.95,   0.05, 0.00],   # FSC
                 [0.05,   0.90, 0.05],   # Shell
                 [0.05,   0.10, 0.85]]   # Gov
SUB_LVL = 0.05
LENGTH_EPISODE = 100  # limit is 313
NUM_EPISODES = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 10
GAMMA = 0.99
SAVE_INTERVAL = 1  # numbers of updates until data/model is saved

CONFIG = {'init_support': INIT_SUPPORT,
          'init_resource': INIT_RESOURCE,
          'sub_lvl': SUB_LVL,
          'length_ep': LENGTH_EPISODE,
          'n_ep': NUM_EPISODES,
          'lr': LEARNING_RATE,
          'batch_size': BATCH_SIZE,
          'gamma': GAMMA,
          'save_interval': SAVE_INTERVAL}

# create saving directory and save config
DEBUG = False
SAVE_DIR = create_dir(DEBUG, NUM_EPISODES, LENGTH_EPISODE, LEARNING_RATE)
torch.save(CONFIG, (SAVE_DIR / 'config'))

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# start timer
start_time_tot = time.time()

# create and setup environment
writer = SummaryWriter(SAVE_DIR)
env = gym.make('FSC_network-v0')
env.setup(AGENTS, INIT_SUPPORT, INIT_RESOURCE, SUB_LVL, LENGTH_EPISODE)

# initialize agents and network optimizers and store them in dicts
optimizers = dict()
all_agents = dict()
for agt in AGENTS:
    agt_optim = dict()
    all_agents.update({agt: eval('agents.' + agt)(ACTION_SPACE, len(AGENTS) + 1, CHANGEABLE_ALLOC[agt])})
    # gov is passiv agent, thus need no optimizer
    if agt != 'Gov':
        networks = all_agents[agt].get_networks()
        for par_agt in networks.keys():
            agt_optim.update({par_agt: torch.optim.Adam(networks[par_agt].parameters(), lr=LEARNING_RATE)})
        optimizers.update({agt: agt_optim})

# save initial agents
torch.save(all_agents, (SAVE_DIR / 'agents_init'))

# init dicts with the active agents
total_rewards = {'FSC': [], 'Shell': []}
batch_returns = {'FSC': [], 'Shell': []}
batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
batch_states = {'FSC': [], 'Shell': []}

batch_count = 0
optim_count = 0
ep = 0
store_graph = False
while ep < NUM_EPISODES:
    start_time = time.time()

    states_cplt = []
    state_0_cplt = env.reset()
    states = {'FSC': [], 'Shell': []}
    rewards = {'FSC': [], 'Shell': []}
    actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
    step_actions = {'FSC': [], 'Shell': []}
    done = False

    while not done:

        for key in ACT_AGT:
            # extract support and resource assignment for agent as array
            state_0 = np.array(state_0_cplt[0].loc['support'][key])
            state_0 = np.append(state_0, state_0_cplt[1].loc[key])
            states[key].append(state_0)

            # get action from each agent and store it
            step_actions.update({key: all_agents[key].get_actions(state_0)})
            for par_agt in actions[key].keys():
                actions[key][par_agt].append(copy.copy(step_actions[key][par_agt]))

        # perform action on environment
        state_1_cplt, r1, done, _ = env.step(step_actions)
        states_cplt.append(state_0_cplt)

        # store rewards
        for agt in ACT_AGT:
            rewards[agt].append(r1[agt])

        # store and update old state
        states_cplt.append(state_0_cplt)
        state_0_cplt = state_1_cplt

        # if done (= new rollout complete), data is put into the batch
        if done:
            # put data in batch
            for agt in ACT_AGT:
                batch_returns[agt].extend(discount_rewards(rewards[agt], GAMMA))
                batch_states[agt].extend(states[agt])
                for par_agt in actions[agt].keys():
                    batch_actions[agt][par_agt].extend(actions[agt][par_agt])
                total_rewards[agt].append(np.array(rewards[agt]).sum())

                # write rewards to tensorboard
                writer.add_scalar('reward_FSC', np.array(rewards['FSC']).sum(), ep)
                writer.add_scalar('reward_Shell', np.array(rewards['Shell']).sum(), ep)

            batch_count += 1

            # if batch full (= enough rollouts for optimization), perform optimization on networks
            if batch_count == BATCH_SIZE:

                optim_count += 1
                # loop over all agents and over the changeable allocation
                for agt in ACT_AGT:
                    for par_agt in optimizers[agt].keys():

                        # TODO: @Kai correct until apply gradients?
                        # set gradients of all optimizers to zero
                        optimizers[agt][par_agt].zero_grad()

                        # create tensors for calculation
                        reward_tensor = torch.FloatTensor(batch_returns[agt])

                        # create Long Tensor and add +1 to use action_tensor as indices
                        action_tensor = torch.LongTensor(np.array(batch_actions[agt][par_agt])+1)

                        # calculate the loss
                        logprob = torch.log(all_agents[agt].predict(batch_states[agt], par_agt))
                        # use torch.gather to get the logprob to the corresponding action
                        selected_logprobs = reward_tensor * torch.gather(logprob, 1,
                                                                         action_tensor.unsqueeze(1)).squeeze()
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()

                        # Apply gradients
                        optimizers[agt][par_agt].step()

                        # write loss to tensorboard after each optimization step
                        writer.add_scalar('mean_loss_' + agt + '_' + par_agt, loss, optim_count)

                        # save specific data after SAVE_INTERVAL number of optimization steps
                        if optim_count % SAVE_INTERVAL == 0:
                            # save weights and its gradients for tensorboard
                            for name, weight in all_agents[agt].get_networks()[par_agt].named_parameters():
                                writer.add_histogram(agt + '_' + par_agt + '_' + name, weight, optim_count)
                                writer.add_histogram(agt + '_' + par_agt + '_' + name + '_grad', weight.grad,
                                                     optim_count)

                            # save agents, all states and rewards
                            torch.save(all_agents, (SAVE_DIR / 'agents_optim{}_ep{}'.format(optim_count, ep+1)))
                            torch.save(states_cplt, (SAVE_DIR / 'states_cplt_optim{}_ep{}'.format(optim_count, ep+1)))
                            torch.save(total_rewards, (SAVE_DIR / 'tot_r'))

                # empty batch
                batch_returns = {'FSC': [], 'Shell': []}
                batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
                batch_states = {'FSC': [], 'Shell': []}
                batch_count = 0
                print('-------------------------------     Update step completed.     -------------------------------')

    ep += 1

    # Print moving average
    if ep % 10 == 0:
        print('Episode {} complete in {:.3f} sec. Average of last 100: FSC: {:.3f}, Shell: {:.3f}'.
              format(ep, time.time() - start_time, np.mean(total_rewards['FSC'][-100:]),
                     np.mean(total_rewards['Shell'][-100:])))

    # save a graph to tensorboard (just for visualisation)
    if store_graph:
        store_graph = False
        writer.add_graph(all_agents['FSC'].get_networks()['Shell'], torch.FloatTensor(states['FSC'][0]))

    # flush the writer, so that everything is written to tensorboard
    writer.flush()

writer.close()
# TODO: save total time
print('Total time: {:.3f}'.format(time.time() - start_time_tot))
