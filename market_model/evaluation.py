import torch
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# TODO: model must be evaluated eval() after loading
# TODO: script verschönern

LOAD_DIR = Path('../saved_data/2020-07-14/1_n_ep1000_l_ep100_lr0.001')

config = torch.load(LOAD_DIR / 'config')
total_rewards = torch.load(LOAD_DIR / 'tot_r')

# plots = []
# plot total rewards of active agents and the rolling mean
# window_width = 50
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
# plot the scaled rewards
# plots.append(plt.figure(3))
# plt.plot(np.array(total_rewards['Shell']) / np.array(total_rewards['Shell']).max(), label='Shell')
# plt.plot(np.array(total_rewards['FSC']) / np.array(total_rewards['FSC']).max(), label='FSC', alpha=0.6)
# plt.legend()
# plt.title('')
# plt.xlabel('episode')
# plt.ylabel('scaled reward')
# plots[2].show()

# determine differences of the neural net weights
# agents = [torch.load(LOAD_DIR / 'agents_init'), torch.load(LOAD_DIR / 'agents_optim100_ep1000')]
# weight_diff = agents[0]['FSC'].get_networks()['Shell'][0].weight.detach().numpy() - \
#               agents[1]['FSC'].get_networks()['Shell'][0].weight.detach().numpy()

# TODO: als graph schön visualisieren?
# plot history of states of specific episode
# versions = ['optim0_ep10', 'optim99_ep1000']
# states_cplt = dict()
# support = dict()
# for version in versions:
#     states_cplt.update({version: torch.load(LOAD_DIR / ('states_cplt_' + version))})
#     tmp = dict()
#     for agt in config['agents']:
#         support.update({agt: [i[0].loc['support', agt] for i in states_cplt]})
#
#
#
state_plots = []
# state_plots.append(plt.figure(1))
# for agt in config['agents']:
#
# plt.plot(sup_fsc, label='FSC')
# plt.plot(sup_shell, label='Shell')
# plt.plot(sup_gov, label='Gov')
# plt.legend()
# plt.title('support of agents - optim0_ep10')
# plt.xlabel('episode')
# plt.ylabel('level of support')
# state_plots[0].show()
#
# # TODO: beides in eine gemeinsame schleife
# resources = dict()
# tmp = dict()
# for agt in config['active_agents']:
#     for par_agt in states_cplt[1][1].index:
#         tmp.update({par_agt: [i[1].loc[agt, par_agt] for i in states_cplt]})
#     resources.update({agt: copy.copy(tmp)})
#
# i = 2
# for agt in config['active_agents']:
#     state_plots.append(plt.figure(i))
#     for par_agt in states_cplt[1][1].index:
#         plt.plot(resources[agt][par_agt], label=par_agt)
#     plt.title('resource allocation - ' + agt + ' - optim0_ep10')
#     plt.xlabel('episode')
#     plt.legend()
#     state_plots[i-1].show()
#     i += 1

version = 'optim99_ep1000'
states_cplt = torch.load(LOAD_DIR / ('states_cplt_' + version))

sup_fsc = [i[0].loc['support', 'FSC'] for i in states_cplt]
sup_shell = [i[0].loc['support', 'Shell'] for i in states_cplt]
sup_gov = [i[0].loc['support', 'Gov'] for i in states_cplt]
i=1
state_plots.append(plt.figure(i))
plt.plot(sup_fsc, label='FSC')
plt.plot(sup_shell, label='Shell')
plt.plot(sup_gov, label='Gov')
plt.legend()
plt.title('support of agents - optim99_ep1000')
plt.xlabel('episode')
plt.ylabel('level of support')
state_plots[i-1].show()
i += 1

# TODO: beides in eine gemeinsame schleife
resources = dict()
tmp = dict()
for agt in config['active_agents']:
    for par_agt in states_cplt[1][1].index:
        tmp.update({par_agt: [i[1].loc[agt, par_agt] for i in states_cplt]})
    resources.update({agt: copy.copy(tmp)})

for agt in config['active_agents']:
    state_plots.append(plt.figure(i))
    for par_agt in states_cplt[1][1].index:
        plt.plot(resources[agt][par_agt], label=par_agt)
    plt.title('resource allocation - ' + agt + ' - optim99_ep1000')
    plt.xlabel('episode')
    plt.legend()
    state_plots[i-1].show()
    i += 1
pass
