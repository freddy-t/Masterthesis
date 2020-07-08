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
    # test = torch.load(filename)
    # TODO: model must be evaluated eval() after loading


def create_dir():
    # function creates directory to store data of the training/testing
    dir_path = Path('../saved_data') / str(datetime.datetime.now().date())

    # create directory for the current date or create a new directory in the current date directory
    n_folders = 0
    if not dir_path.is_dir():
        os.mkdir(dir_path)
        save_dir = dir_path / (str(n_folders) + '_')
        os.mkdir(save_dir)
    else:
        for _, dirnames, _ in os.walk(dir_path):
            n_folders += len(dirnames)
        save_dir = dir_path / (str(n_folders+1) + '_')
        os.mkdir(save_dir)

    return save_dir
