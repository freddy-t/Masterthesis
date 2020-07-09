import numpy as np
import torch
import datetime
import os
from pathlib import Path


def discount_rewards(rewards, gamma):
    # discount future rewards with gamma
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # rotate rewards, calculate cumulative sum and rotate again, so that the return is calculated
    r = r[::-1].cumsum()[::-1]

    return r


def save_obj(path, obj):
    torch.save(obj, path)


def create_dir(n_ep, l_ep, debug_flag):
    # function creates directory to store data of the training/testing
    dir_path = Path('../saved_data') / str(datetime.datetime.now().date())

    if debug_flag:
        bug_path = Path('../saved_data') / (str(datetime.datetime.now().date()) + '_debug')
        if not bug_path.is_dir():
            os.mkdir(bug_path)
        return bug_path

    # create directory for the current date or create a new directory in the current date directory
    n_folders = 0
    if not dir_path.is_dir():
        os.mkdir(dir_path)
        save_dir = dir_path / (str(1) + '_n_ep_' + str(n_ep) + '_l_ep_' + str(l_ep))
        os.mkdir(save_dir)
    else:
        for _, dirnames, _ in os.walk(dir_path):
            n_folders += len(dirnames)
        save_dir = dir_path / (str(n_folders+1) + '_n_ep_' + str(n_ep) + '_l_ep_' + str(l_ep))
        os.mkdir(save_dir)

    return save_dir
