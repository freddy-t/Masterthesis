import gym
import gym_FSC_network

# create environment and set seed
agents = ['FSC', 'Shell', 'Gov']
env = gym.make('FSC_network-v0')
env.setup(agents)
env.seed(42)




