import torch
import matplotlib.pyplot as plt
from pathlib import Path
# TODO: model must be evaluated eval() after loading

LOAD_DIR = Path('../saved_data/2020-07-08/4_n_ep_1000_l_ep_100')

config = torch.load(LOAD_DIR / 'config')
total_rewards = torch.load(LOAD_DIR / 'tot_r')

plots = []
plots.append(plt.figure(1))
plt.plot(total_rewards['FSC'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plots[0].show()

plots.append(plt.figure(2))
plt.plot(total_rewards['Shell'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plots[1].show()

# input()

# TODO: weiter analysieren
agents = [torch.load(LOAD_DIR / 'agents_init'), torch.load(LOAD_DIR / 'agents_ep_1000')]
weight_diff = agents[0]['FSC'].get_networks()['Shell'][0].weight.detach().numpy() - \
              agents[1]['FSC'].get_networks()['Shell'][0].weight.detach().numpy()

# TODO: als graph schön visualisieren?
states_cplt = torch.load(LOAD_DIR / 'states_cplt_ep_1000')
pass
