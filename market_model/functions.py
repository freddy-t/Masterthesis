import numpy as np
import datetime
import os
import shutil
from pathlib import Path

PATH_SAVED = Path('../saved_data')


def discount_rewards(rewards, gamma):
    # discount future rewards with gamma
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # rotate rewards, calculate cumulative sum and rotate again, so that the return is calculated
    r = r[::-1].cumsum()[::-1]

    return r


def create_dir(debug_flag, lr, del_search, beta, del_res, env_type='EnvAlt'):
    # function creates directory to store data of the training/testing
    dir_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type)

    # create directory for debugging purposes by deleting or just creating it
    if debug_flag:
        bug_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type + '_debug')
        if not bug_path.is_dir():
            os.mkdir(bug_path)
        else:
            shutil.rmtree(bug_path)
            os.mkdir(bug_path)
        return bug_path

    # create directory for the current date or create a new directory in the current date directory
    n_folders = 0
    if not dir_path.is_dir():
        os.mkdir(dir_path)
        save_dir = dir_path / ('exp' + str(1) + '_lr' + str(lr) + '_rch' + str(del_search) + '_bta' + str(beta) +
                               '_rce' + str(del_res))
        os.mkdir(save_dir)
    else:
        for _, dirnames, _ in os.walk(dir_path):
            n_folders += len(dirnames)
        save_dir = dir_path / ('exp' + str(n_folders+1) + '_lr' + str(lr) + '_rch' + str(del_search) +
                               '_bta' + str(beta) + '_rce' + str(del_res))
        os.mkdir(save_dir)

    return save_dir


def create_dir_ex_val(debug_flag, lr, lambda_, kappa, sup, env_type='EnvAlt'):
    # function creates directory to store data of the training/testing
    dir_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type)

    # create directory for debugging purposes by deleting or just creating it
    if debug_flag:
        bug_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type + '_debug')
        if not bug_path.is_dir():
            os.mkdir(bug_path)
        else:
            shutil.rmtree(bug_path)
            os.mkdir(bug_path)
        return bug_path

    # create directory for the current date or create a new directory in the current date directory
    n_folders = 0
    if not dir_path.is_dir():
        os.mkdir(dir_path)
        save_dir = dir_path / ('exp' + str(1) + '_lr' + str(lr) + '_lba' + str(lambda_) + '_kpa' + str(kappa) +
                               '_sup' + str(sup))
        os.mkdir(save_dir)
    else:
        for _, dirnames, _ in os.walk(dir_path):
            n_folders += len(dirnames)
        save_dir = dir_path / ('exp' + str(n_folders+1) + '_lr' + str(lr) + '_lba' + str(lambda_) +
                               '_kpa' + str(kappa) + '_sup' + str(sup))
        os.mkdir(save_dir)

    return save_dir


def create_val_dir(debug_flag, suffix, env_type='EnvAlt'):
    # function creates directory to store data of the training/testing
    dir_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type + '_val')

    # create directory for debugging purposes by deleting or just creating it
    if debug_flag:
        bug_path = PATH_SAVED / (str(datetime.datetime.now().date()) + '_' + env_type + '_val_debug')
        if not bug_path.is_dir():
            os.mkdir(bug_path)
        else:
            shutil.rmtree(bug_path)
            os.mkdir(bug_path)
        return bug_path

    # create directory for the current date or create a new directory in the current date directory
    if not dir_path.is_dir():
        os.mkdir(dir_path)
        save_dir = dir_path / ('evaluation_run' + str(1) + suffix)
        os.mkdir(save_dir)
    else:
        n_folders = len(next(os.walk(dir_path))[1])
        save_dir = dir_path / ('evaluation_run' + str(n_folders+1) + suffix)
        os.mkdir(save_dir)

    return save_dir


def create_dict(required_nets, act_agt):
    new_dict = dict()
    for agt in act_agt:
        tmp = dict()
        for par_agt in required_nets[agt]:
            tmp[par_agt] = []
        new_dict[agt] = tmp
    return new_dict
