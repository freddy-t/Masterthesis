import gym
import agents
import numpy as np
import gym_FSC_network

AGENTS = ['FSC', 'Shell', 'Gov']
# TODO: ?resource and support initialize in env and check if keys are matching
# initial support by the agents, must be in order as in AGENTS
INIT_SUPPORT = [[1, 0.2, 0.1]]
# initial resource assignment
#       FSC  Shell  Gov
# FSC
# Shell
# Gov
INIT_RESOURCE = [[  0.5,   0.2,   0.3],
                 [  0.1,   0.5,   0.4],
                 [0.1, 0.1, 0.8]]

# pre-allocation
all_agents = dict()
actions = dict()
action_space = [-1, 0, 1]  # action definition: -1 = decrease ,1 = increase, 0 = maintain
num_states = len(AGENTS) + 1

# ######################################################################################################################

# create environment and set seed
env = gym.make('FSC_network-v0')
env.setup(AGENTS, INIT_SUPPORT, INIT_RESOURCE)
env.seed(42)

# create and initialize agents and store them in a dict
for agt in AGENTS:
    if agt == 'FSC':
        all_agents.update({agt: agents.FSC(action_space, num_states, ['Shell', 'Gov'])})
    if agt == 'Shell':
        all_agents.update({agt: agents.Shell(action_space, num_states, ['FSC'])})
    if agt == 'Gov':
        all_agents.update({agt: agents.Gov(action_space, num_states, [])})

state_0_cplt = env.reset()

# perform an action
for key in all_agents.keys():
    # extract support and resource assignment for agent as array
    state_0 = np.array(state_0_cplt[0].loc['support'][key])
    state_0 = np.append(state_0, state_0_cplt[1].loc[key])

    # get action from agent
    actions.update({key: all_agents[key].get_actions(state_0)})

# perform step on environment
state_1_cplt, r, done, _ = env.step(actions)

pass
