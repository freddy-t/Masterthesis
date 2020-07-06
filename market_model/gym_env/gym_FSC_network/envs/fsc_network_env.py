import gym
import numpy as np
import pandas as pd
import copy
from pathlib import Path


DATAFILE = Path("C:/Users/fredd/Desktop/freddy/sciebo/Masterarbeit/03_Konzeptentwicklung/Daten") /\
                "00_interesting_data.xlsx"

# TODO: wird überhaupt eine gym environment benötigt, d.h. iwelche gym Funktionen benutzt?


class FSCNetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__init_support = None
        self.__init_resource = None
        self._shared_data_test = None
        self._shared_data_train = None
        self._shell_data_test = None
        self._shell_data_train = None
        self._support = None
        self._resource = None
        self._episode_len = None
        self._delta_resource = 0.005
        self._support_factor = 0.1
        self._mode = 'train'
        self._current_step = None
        self._sub_lvl = None

    def step(self, actions):

        # set the support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(self._support)
        for agt in self._support.keys():
            add_val = 0
            # FSC has fixed support
            if agt != 'FSC':
                for par_agt in self._support.keys():
                    # agents do not due to resource assignment to itself
                    if agt != par_agt:
                        # change to partner agent is only initiated, support is different
                        if self._support.loc['support', agt] < self._support.loc['support', par_agt]:
                            factor = self._support_factor
                        elif self._support.loc['support', agt] > self._support.loc['support', par_agt]:
                            factor = -self._support_factor
                        else:
                            factor = 0
                        add_val += factor * (self._resource.loc[agt, par_agt] + self._resource.loc[par_agt, agt])/2

                # change in support is based on previous support --> copy is used
                # check that support remains between 0 and 1
                new_val = orig_support.loc['support', agt] + add_val
                if 0.0 > new_val:
                    self._support.loc['support', agt] = 0
                elif new_val > 1.0:
                    self._support.loc['support', agt] = 1
                else:
                    self._support.loc['support', agt] = new_val

        # set the resource assignment based on the agents' actions
        orig_resource = copy.copy(self._resource)
        for act_agt in actions.keys():
            # Gov is passiv
            if act_agt != 'Gov':
                for part_agt in actions[act_agt].keys():
                    # update value for resource assignment if it is between 0 and 1
                    val = self._resource.loc[act_agt, part_agt] + self._delta_resource * actions[act_agt][part_agt]
                    if (0.0 <= val) and (val <= 1.0):
                        self._resource.loc[act_agt, part_agt] = val

                # set the resource assignment of the agent to itself
                # as a result of the resource assignment to the other agents
                ext_assign = self._resource.loc[act_agt].sum() - self._resource.loc[act_agt, act_agt]
                self._resource.loc[act_agt, act_agt] = 1 - ext_assign

                # if actions are not feasible, keep the old resource assignment
                if (True in (self._resource.loc[act_agt][:] < 0).values) or \
                        (True in (self._resource.loc[act_agt][:] > 1).values):
                    print('---------------------    Actions {} of agent {} were not feasible.    ---------------------'\
                          .format(actions[act_agt], act_agt))
                    self._resource.loc[act_agt] = orig_resource.loc[act_agt]

        # check if resource calculations were correct
        if (self._resource.sum(axis=1).sum() / len(self._resource.index) != 1) or \
           (True in (self._resource[:][:] < 0).values) or (True in (self._resource[:][:] > 1).values):
            raise ValueError('Resource assignment calculation went wrong')

        observation = [copy.copy(self._support), copy.copy(self._resource)]
        reward = self.calc_reward(observation, self._mode)
        done = self.check_if_done()
        info = []
        return observation, reward, done, info

    def reset(self) -> list:
        self._current_step = 0
        # set support and resource to initial values and return the state
        self._support = copy.copy(self.__init_support)
        self._resource = copy.copy(self.__init_resource)
        return [copy.copy(self._support), copy.copy(self._resource)]

    def render(self, mode='human', close=False):
        pass

    def setup(self, agt: list, init_sup: list, init_res, sub_lvl, ep_len):
        # setup the environment
        shared, shell = load_data(agt)
        self._sub_lvl = sub_lvl
        self._episode_len = ep_len

        # split to train and test data
        self._shared_data_train, self._shared_data_test = split_data(shared)
        self._shell_data_train, self._shell_data_test = split_data(shell)

        # set the initial states
        self.__init_support = pd.DataFrame(init_sup, index=['support'], columns=agt)
        self.__init_resource = pd.DataFrame(init_res, index=agt, columns=agt)

    def get_shell_data(self):
        return copy.copy(self.__shell_data)

    def get_shared_data(self):
        return copy.copy(self.__shared_data)

    def calc_reward(self, obs, mode):
        step = self._current_step
        if mode == 'train':
            sd_data = self._shared_data_train
            sl_data = self._shell_data_train
        else:
            sd_data = self._shared_data_test
            sl_data = self._shell_data_test

        # calculate reward for FSC
        r = {'FSC': obs[0].loc['support'].sum()}

        # calculate reward for Shell
        own_return = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
                      sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
        r_shell = self._resource['Shell']['Shell'] * own_return + self._sub_lvl * self._resource.loc['Shell', 'FSC'] + \
                  self._sub_lvl * self._support.loc['support', 'Shell'] * self._resource['Shell']['Shell']
        r.update({'Shell': r_shell})

        # update step
        self._current_step += 1

        return r

    def set_mode(self, mod):
        self._mode = mod

    def check_if_done(self):
        # minus one as the first step is performed with step=0, taking the iloc[0] of the data DataFrames
        if (self._current_step == self._episode_len) or (self._current_step == len(self._shell_data_train) - 1):
            return True
        else:
            return False


def load_data(agents: list) -> [pd.DataFrame, pd.DataFrame]:
    # define all possible weeks to see which weekly data is missing
    all_weeks = pd.read_excel(DATAFILE, 'dates').set_index('fridays')
    co2_price = pd.read_excel(DATAFILE, 'EEX_EUA_Spot_Open_USD').set_index('fridays')
    leading_df = all_weeks.join(co2_price)

    # fill missing data with linear interpolation
    leading_df.interpolate(inplace=True, axis='index')

    # remove 2020 data, as there is no quarterly data available
    leading_df.drop(leading_df[leading_df.index > pd.Timestamp(year=2020, month=1, day=1)].index, inplace=True)

    # load agent specific data
    if 'Shell' in agents:
        shell_data = load_shell(leading_df)
        print('--------------------------------    Shell data successfully loaded.    --------------------------------')

    # modify the index, so that it is composed of the year and the week only
    return index_to_m_y(leading_df), index_to_m_y(shell_data)


def load_shell(ld_df: pd.DataFrame) -> pd.DataFrame:
    # load excel and set index
    shell_orig = pd.read_excel(DATAFILE, 'shell', usecols=[1, 3, 5, 7]).set_index(['Date'])

    # shift shell entries and save index values for further data handling
    shell_orig = shell_orig.shift(1, freq='h')
    idx_val = shell_orig.index

    # shift the index by two to get sundays and join DataFrames
    ld_df = ld_df.shift(2, freq='D')
    shell = ld_df.append(shell_orig).sort_index()

    # remove 2020 data, as there is no quarterly data available
    shell.drop(shell[shell.index > pd.Timestamp(year=2020, month=1, day=1)].index, inplace=True)

    # calculate weekly data by dividing quarterly values by number of weeks in the quarter
    tmp = len(idx_val) - 1
    for i in range(tmp):
        num_ent = shell.loc[(idx_val[i+1] < shell.index) & (shell.index < idx_val[i]), 'CO2_price'].count()
        for key in shell.keys():
            if key != 'CO2_price':
                shell.loc[(idx_val[i+1] < shell.index) & (shell.index < idx_val[i]), key] \
                    = shell.loc[idx_val[i], key] / num_ent

    # calculate weekly data for first quarter
    num_ent = shell.loc[shell.index < idx_val[tmp], 'CO2_price'].count()
    for key in shell.keys():
        if key != 'CO2_price':
            shell.loc[shell.index < idx_val[tmp], key] = shell.loc[idx_val[tmp], key] / num_ent

    # remove shifted shell values
    shell.dropna(axis='index', how='any', subset=['CO2_price'], inplace=True)

    # check if translation from quarterly to weekly data was correct
    val_test = [shell[key].sum() == shell_orig[key].sum() for key in shell.keys() if (key != 'CO2_price')]
    if val_test.count(False):
        print('--------------    Translation from quarterly to weekly data went wrong. Please check!    --------------')
        # TODO throw exception

    # reshift to original dates
    shell = shell.shift(-2, freq='D')
    return shell.drop(['CO2_price'], axis='columns')


def index_to_m_y(df: pd.DataFrame) -> pd.DataFrame:
    # modify the index, so that it is composed of the year and the week only
    df['week'] = df.index.weekofyear.astype(str)
    df['year'] = df.index.year.astype(str)
    df = df.set_index(['year', 'week'])

    return df


def split_data(data: pd.DataFrame):
    test = data.loc['2019']
    train = data.drop(index='2019')

    return train, test
