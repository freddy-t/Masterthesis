import numpy as np


def discount_rewards(rewards, gamma):
    # discount future rewards with gamma
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # rotate rewards, calculate cumulative sum and rotate again, so that
    r = r[::-1].cumsum()[::-1]

    return r


def get_total_rewards(rewards):
    # return the total reward of the episode
    tot_r = dict()
    for agt in rewards.keys():
        tot_r.update({agt: np.array([rewards[agt][i] for i in range(len(rewards[agt]))]).sum()})
    return tot_r
