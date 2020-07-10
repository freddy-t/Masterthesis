import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# TODO: model must be evaluated eval() after loading

LOAD_DIR = Path('../saved_data/2020-07-10/1_n_ep1000_l_ep100_lr0.001')

config = torch.load(LOAD_DIR / 'config')
total_rewards = torch.load(LOAD_DIR / 'tot_r')
# window_width = 50
#
# plots = []
# plots.append(plt.figure(1))
# ma = pd.Series(total_rewards['FSC']).rolling(window_width).mean()
# plt.plot(total_rewards['FSC'])
# plt.plot(ma, label='50ep - ma')
# plt.title('FSC')
# plt.legend()
# plt.xlabel('episode')
# plt.ylabel('reward')
# plots[0].show()
#
# plots.append(plt.figure(2))
# ma = pd.Series(total_rewards['Shell']).rolling(window_width).mean()
# plt.plot(total_rewards['Shell'])
# plt.plot(ma, label='50ep - ma')
# plt.title('Shell')
# plt.xlabel('episode')
# plt.ylabel('reward')
# plots[1].show()
#
# plots.append(plt.figure(3))
# plt.plot(np.array(total_rewards['Shell']) / np.array(total_rewards['Shell']).max(), label='Shell')
# plt.plot(np.array(total_rewards['FSC']) / np.array(total_rewards['FSC']).max(), label='FSC', alpha=0.6)
# plt.legend()
# plt.title('')
# plt.xlabel('episode')
# plt.ylabel('scaled reward')
# plots[2].show()

# input()

# TODO: weiter analysieren
agents = [torch.load(LOAD_DIR / 'agents_init'), torch.load(LOAD_DIR / 'agents_optim_300_ep_1000')]
weight_diff = agents[0]['FSC'].get_networks()['Shell'][0].weight.detach().numpy() - \
              agents[1]['FSC'].get_networks()['Shell'][0].weight.detach().numpy()

# TODO: als graph schön visualisieren?
states_cplt = torch.load(LOAD_DIR / 'states_cplt_optim_300_ep_1000')
pass
