import gym
import agents
import gym_FSC_network

AGENTS = ['FSC', 'Shell', 'Gov']
# TODO: ?resource and support initialize in env and check if keys are matching
# initial support by the agents, must be in order as in AGENTS
INIT_SUPPORT = [[1, 0.1, 0.1]]
# initial resource assignment
#       FSC  Shell  Gov
# FSC
# Shell
# Gov
INIT_RESOURCE = [[  1,   0,   0],
                 [  0,   1,   0],
                 [0.1, 0.1, 0.8]]

# create environment and set seed
env = gym.make('FSC_network-v0')
env.setup(AGENTS, INIT_SUPPORT, INIT_RESOURCE)
env.seed(42)

# pre-allocation
all_agents = dict()
actions = dict()
num_agents = len(AGENTS)
n_actions = 3

# create and initialize agents in dict
for agt in AGENTS:
    # TODO: ? define number action space and state space as numbers before in a gernal section?
    if agt == 'FSC':
        all_agents.update({agt: agents.FSC(n_actions, num_agents+1, ['Shell', 'Gov'])})
    if agt == 'Shell':
        all_agents.update({agt: agents.Shell(n_actions, num_agents+1, ['FSC'])})
    if agt == 'Gov':
        all_agents.update({agt: agents.Gov(0, 0, [])})

state_0 = env.reset()
# get actions from agents
for key in all_agents.keys():
    actions.update({key: all_agents[key].get_action(state_0)})
#s_1, r, done, _ = env.step(actions)

pass
