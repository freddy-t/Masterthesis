import torch
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATE = '2020-07-17_debug'
EXP = '3_n_ep1000_l_ep100_lr0.001'
LOAD_DIR = Path('../saved_data') / DATE # / EXP

# load general data
config = torch.load(LOAD_DIR / 'config')
total_rewards = torch.load(LOAD_DIR / 'tot_r')
times = torch.load(LOAD_DIR / 'running_times')

# old_plots = []
# plot total rewards of active agents and the rolling mean
# window_width = 50
# old_plots.append(plt.figure(1))
# ma = pd.Series(total_rewards['FSC']).rolling(window_width).mean()
# plt.plot(total_rewards['FSC'])
# plt.plot(ma, label='50ep - ma')
# plt.title('FSC')
# plt.legend()
# plt.xlabel('episode')
# plt.ylabel('reward')
# old_plots[0].show()
#
# old_plots.append(plt.figure(2))
# ma = pd.Series(total_rewards['Shell']).rolling(window_width).mean()
# plt.plot(total_rewards['Shell'])
# plt.plot(ma, label='50ep - ma')
# plt.title('Shell')
# plt.xlabel('episode')
# plt.ylabel('reward')
# old_plots[1].show()
#
# plot the scaled rewards
# old_plots.append(plt.figure(3))
# plt.plot(np.array(total_rewards['Shell']) / np.array(total_rewards['Shell']).max(), label='Shell')
# plt.plot(np.array(total_rewards['FSC']) / np.array(total_rewards['FSC']).max(), label='FSC', alpha=0.6)
# plt.legend()
# plt.title('')
# plt.xlabel('episode')
# plt.ylabel('scaled reward')
# old_plots[2].show()

# determine differences of the neural net weights
# agents = [torch.load(LOAD_DIR / 'agents_init'), torch.load(LOAD_DIR / 'agents_optim100_ep1000')]
# weight_diff = agents[0]['FSC'].get_networks()['Shell'][0].weight.detach().numpy() - \
#               agents[1]['FSC'].get_networks()['Shell'][0].weight.detach().numpy()


# load state data
versions = ['optim1']#, 'optim100']
states = dict()
for version in versions:
    states.update({version: torch.load(LOAD_DIR / ('batch_states_' + version))})

# ------------------ create plots for support ------------------
plots = []
fig_count = 0
plotted_count = 0
# loop over all versions
for version in versions:
    fig = plt.figure(fig_count, figsize=(12, 4))
    plots.append(fig)
    fig_count += 1
    # create two axes
    ax = [fig.add_subplot(121), fig.add_subplot(122)]
    # loop over each agent
    for key in config['agents']:
        # calculate mean and std for support
        mean = states[version][key].mean(axis=0)[:, 0]
        std = states[version][key].std(axis=0)[:, 0]
        x = range(0, states[version][key].shape[1])
        # plot mean
        ax[0].plot(mean, label=key)
        ax[1].plot(mean, label=key)
        # plot std as an area
        ax[0].fill_between(x, mean+std, mean-std, alpha=0.2)
        # plot max and min as an area
        ax[1].fill_between(x, states[version][key].max(axis=0)[:, 0],
                           states[version][key].min(axis=0)[:, 0], alpha=0.2)
    # set some plot properties
    fig.suptitle('support of agents of batch - ' + version)
    ax[0].set_title('mean and standard deviation')
    ax[1].set_title('mean and max-min')
    for axis in fig.get_axes():
        axis.set_xlabel('step')
        axis.set_ylabel('level of support')
        axis.legend()

# plot first two plots
for i in range(fig_count):
    plotted_count += 1
    # plots[i].show()
# state_plots[0].savefig((LOAD_DIR / (versions[0] + '.pdf')))  # svg auch möglich
# state_plots[1].show()

# ------------------ create plots for resource assignment ------------------
for version in versions:
    fig = plt.figure(fig_count, figsize=(12, 4))
    plots.append(fig)
    # create new axes
    ax = [fig.add_subplot(121), fig.add_subplot(122)]
    fig_count += 1

    for i, key in enumerate(config['active_agents']):
        # calculate mean of resource assignment to the different partner agents
        x = range(0, states[version][key].shape[1])
        fsc = states[version][key].mean(axis=0)[:, 1]
        shell = states[version][key].mean(axis=0)[:, 2]
        gov = states[version][key].mean(axis=0)[:, 3]

        # create stacked plots
        ax[i].stackplot(x, gov + shell + fsc)
        ax[i].stackplot(x, gov + shell)
        ax[i].stackplot(x, gov)
        ax[i].set_title(key)
        ax[i].legend(['FSC', 'Shell', 'Gov'], loc='center')
    fig.suptitle('mean resource allocation for batch - ' + version)

# plot last
for i in range(plotted_count, fig_count):
    plotted_count += 1
    # plots[i].show()
# state_plots[2].show()
# state_plots[3].show()

# ------------------ plot support calculation ------------------
support_calc = dict()
for version in versions:
    support_calc.update({version: torch.load(LOAD_DIR / ('support_calc_' + version))})


pass
