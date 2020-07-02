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
        self.__shared_data = None
        self.__shell_data = None
        self.__init_support = None
        self.__init_resource = None
        self._support = None
        self._resource = None
        self._delta_resource = 0.005
        self._support_factor = 0.1

    def step(self, actions):

        # set the resulting support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(self._support)
        print('orig support:\n {}'.format(orig_support))
        print(self._resource)
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

                        print('agt:' + agt + '   ' + 'par_agt: ' + par_agt)
                        print('added value: {}'.format(add_val))
                # change in support is based on previous support --> use copy
                self._support.loc['support', agt] = orig_support.loc['support', agt] + add_val
                print(self._support)
#TODO: berechnung des supports für mehrere konstellationen ausrechnen und sicherstellen, dass support nicht negativ wird
        # set the resource assignment based on the agents' actions
        for act_agt in actions.keys():
            # Gov is passiv
            if act_agt != 'Gov':
                for part_agt in actions[act_agt].keys():
                    # update value resource assignment if it is between 0 and 1
                    new_val = self._resource.loc[act_agt, part_agt] + self._delta_resource * actions[act_agt][part_agt]
                    if (0.0 <= new_val) and (new_val <= 1.0):
                        self._resource.loc[act_agt, part_agt] += self._delta_resource * actions[act_agt][part_agt]

                # set the resource assignment of the agent to itself
                # as a result of the resource assignment to the other agents
                ext_assign = np.array([self._resource.loc[act_agt, i] for i in actions[act_agt].keys()]).sum()
                self._resource.loc[act_agt, act_agt] = 1 - ext_assign

                print('active agent: {}'.format(act_agt))
                print('partner agent: {}'.format(part_agt))
                print(self._resource)


        observation = 1
        reward = 1
        done = 1 # if there are no steps left
        info = 1
        return observation, reward, done, info

    def reset(self) -> list:
        self._support = self.__init_support
        self._resource = self.__init_resource
        return [self._support, self._resource]

    def render(self, mode='human', close=False):
        pass

    def setup(self, agt: list, init_sup: list, init_res):
        # setup the environment by loading external data and setting the initial states
        self.__shared_data, self.__shell_data = load_data(agt)
        self.__init_support = pd.DataFrame(init_sup, index=['support'], columns=agt)
        self.__init_resource = pd.DataFrame(init_res, index=agt, columns=agt)

    def get_shell_data(self):
        return self.__shell_data

    def get_shared_data(self):
        return self.__shared_data


def load_data(agents: list) -> [pd.DataFrame, pd.DataFrame]:
    # define all possible weeks to see which weekly data is missing
    all_weeks = pd.read_excel(DATAFILE, 'dates').set_index('fridays')
    co2_price = pd.read_excel(DATAFILE, 'EEX_EUA_Spot_Open_USD').set_index('fridays')
    leading_df = all_weeks.join(co2_price)

    # fill missing data with linear interpolation
    leading_df.interpolate(inplace=True, axis='index')

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

    return shell.drop(['CO2_price'], axis='columns')


def index_to_m_y(df: pd.DataFrame) -> pd.DataFrame:
    # modify the index, so that it is composed of the year and the week only
    df['week'] = df.index.weekofyear.astype(str)
    df['year'] = df.index.year.astype(str)
    df = df.set_index(['year', 'week'])

    return df
