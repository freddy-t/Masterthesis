import gym
import agents
import numpy as np
import gym_FSC_network
from torch import optim

# TODO: ?resource and support initialize in env and check if keys are matching
# TODO: if gov should be active: look for "passiv" and change the lines

AGENTS = ['FSC', 'Shell', 'Gov']
CHANGEABLE_ALLOC = {'FSC':   ['Shell', 'Gov'],
                    'Shell': ['FSC'],
                    'Gov':   []}

# initial support by the agents, must be in order as in AGENTS
INIT_SUPPORT = [[1, 0.2, 0.1]]
# initial resource assignment
#       FSC  Shell  Gov
# FSC
# Shell
# Gov
INIT_RESOURCE = [[ 0.8,   0.2, 0],
                 [0.05,   0.8, 0.15],
                 [ 0.1,   0.1, 0.8]]

SUB_LVL = 0.05
LENGTH_EPISODE = 100
NUM_EPISODES = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 20

# pre-allocation
action_space = [-1, 0, 1]  # action definition: -1 = decrease, 0 = maintain, 1 = increase
num_states = len(AGENTS) + 1

# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################


# create environment and set seed
env = gym.make('FSC_network-v0')
env.setup(AGENTS, INIT_SUPPORT, INIT_RESOURCE, SUB_LVL, LENGTH_EPISODE)
# env.seed(42)

# initialize agents and network optimizers and store them in dicts
optimizers = dict()
all_agents = dict()
for agt in AGENTS:
    agt_optim = dict()
    all_agents.update({agt: agents.FSC(action_space, num_states, CHANGEABLE_ALLOC[agt])})
    # gov is passiv agent, thus need no optimizer
    if agt != 'Gov':
        networks = all_agents[agt].get_networks()
        for par_agt in networks.keys():
            agt_optim.update({par_agt: optim.Adam(networks[par_agt].parameters(), lr=LEARNING_RATE)})
        optimizers.update({agt: agt_optim})

total_rewards = []
batch_rewards = []
batch_actions = []
batch_states = []
batch_counter = 1

ep = 0
while ep < NUM_EPISODES:

    state_0_cplt = env.reset()
    states = []
    rewards = []
    actions = []
    step_actions = dict()
    done = False

    while not done:

        # get action from each agent
        for key in all_agents.keys():
            # Gov is passiv
            if key != 'Gov':
                # extract support and resource assignment for agent as array
                state_0 = np.array(state_0_cplt[0].loc['support'][key])
                state_0 = np.append(state_0, state_0_cplt[1].loc[key])
                step_actions.update({key: all_agents[key].get_actions(state_0)})

        # perform action on environment
        state_1_cplt, r1, done, _ = env.step(step_actions)
        states.append(state_0_cplt)
        rewards.append(r1)
        actions.append(step_actions)

        # update state
        state_0_cplt = state_1_cplt

        # if done, data is put into the batch
        if done:
            # put data in batch
            batch_states.append(states)
            # batch_rewards.append()  # TODO: rewards must be discounted
            batch_actions.append(actions)
            #TODO : total_rewards

            batch_counter += 1

            # if batch full, perform optimization on networks
            if batch_counter == BATCH_SIZE:

                pass

    ep += 1
