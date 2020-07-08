import gym
import agents
import torch
import copy
import time
import numpy as np
import gym_FSC_network
from functions import discount_rewards, create_dir, save_obj

# TODO: ?resource and support initialize in env and check if keys are matching
# TODO: if gov should be active: look for "passiv" and change the lines

AGENTS = ['FSC', 'Shell', 'Gov']
CHANGEABLE_ALLOC = {'FSC':   ['Shell', 'Gov'],
                    'Shell': ['FSC'],
                    'Gov':   []}
ACT_AGT = ['FSC', 'Shell']               # set the active agents
INIT_SUPPORT = [[1, 0.1, 0.1]]           # initial support by the agents, must be in order as in AGENTS

# initial resource assignment            #       FSC  Shell  Gov
INIT_RESOURCE = [[0.95,   0.05, 0.00],   # FSC
                 [0.05,   0.90, 0.05],   # Shell
                 [0.05,   0.10, 0.85]]   # Gov
SUB_LVL = 0.05
LENGTH_EPISODE = 10  # limit is 313
NUM_EPISODES = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 5
GAMMA = 0.99
SAVE_INTERVAL = 2  # numbers of updates until data/model is saved

# TODO: andere Parameter zur Namensgebung übergeben?
# create saving directory
SAVE_DIR = create_dir()

# pre-allocation
action_space = [-1, 0, 1]  # action definition: -1 = decrease, 0 = maintain, 1 = increase
num_states = len(AGENTS) + 1

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# create and setup environment
env = gym.make('FSC_network-v0')
env.setup(AGENTS, INIT_SUPPORT, INIT_RESOURCE, SUB_LVL, LENGTH_EPISODE)

# initialize agents and network optimizers and store them in dicts
optimizers = dict()
all_agents = dict()
for agt in AGENTS:
    agt_optim = dict()
    all_agents.update({agt: eval('agents.' + agt)(action_space, num_states, CHANGEABLE_ALLOC[agt])})
    # gov is passiv agent, thus need no optimizer
    if agt != 'Gov':
        networks = all_agents[agt].get_networks()
        for par_agt in networks.keys():
            agt_optim.update({par_agt: torch.optim.Adam(networks[par_agt].parameters(), lr=LEARNING_RATE)})
        optimizers.update({agt: agt_optim})

# save initial agents
torch.save(all_agents, (SAVE_DIR / 'agents_init'))

# init dicts for the both active agents
total_rewards = {'FSC': [], 'Shell': []}
states_cplt = []
batch_returns = {'FSC': [], 'Shell': []}
batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
batch_states = {'FSC': [], 'Shell': []}

batch_counter = 0
save_count = 0
ep = 0
while ep < NUM_EPISODES:
    start_time = time.time()

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
            for par_agt in actions[agt].keys():
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

            batch_counter += 1

            # if batch full (= enough rollouts for optimization), perform optimization on networks
            if batch_counter == BATCH_SIZE:
                save_count += 1
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
                        # torch.gather to get the logprob to the corresponding action
                        selected_logprobs = reward_tensor * torch.gather(logprob, 1,
                                                                         action_tensor.unsqueeze(1)).squeeze()
                        loss = -selected_logprobs.mean()

                        # Calculate gradients
                        loss.backward()

                        # Apply gradients
                        optimizers[agt][par_agt].step()

                # empty batch
                batch_returns = {'FSC': [], 'Shell': []}
                batch_actions = {'FSC': {'Shell': [], 'Gov': []}, 'Shell': {'FSC': []}}
                batch_states = {'FSC': [], 'Shell': []}
                batch_counter = 0
                print('-------------------------------     Update step completed.     -------------------------------')

    ep += 1

    # Print moving average
    print('Episode {} complete in {:.3f} sec. Average of last 100: FSC: {:.3f}, Shell: {:.3f}'.
          format(ep, time.time() - start_time, np.mean(total_rewards['FSC'][-100:]),
                 np.mean(total_rewards['Shell'][-100:])))

    # save data
    if save_count == SAVE_INTERVAL:
        save_count = 0
        torch.save(all_agents, (SAVE_DIR / ('agents_ep_' + str(ep))))
        torch.save(total_rewards, (SAVE_DIR / 'tot_r'))
    # TODO: weitermachen: save state_cplt, gesamter batch oder nur einzelne episoden? bisher states aller episoden gesammelt


# file = open(filename, 'wb')
# pickle.dump(all_agents, file)
# file.close()
# fh = open(filename, 'rb')
# test = pickle.load(fh)
