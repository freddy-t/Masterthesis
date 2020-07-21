from typing import Dict, List, NoReturn
import numpy as np
import pandas as pd
import copy
from pathlib import Path

DATAFILE = Path("C:/Users/fredd/Desktop/freddy/sciebo/Masterarbeit/03_Konzeptentwicklung/Daten") / \
           "00_interesting_data.xlsx"

np.random.seed(42)


class FSCNetworkEnv(object):

    def __init__(self, agt, init_sup, init_res, sub_lvl, ep_len, delta_res, sup_fsc, sup_ratio,
                 mode='train', agg_weeks=1):
        # setup environment parameters
        shared, shell = load_data(agt, agg_weeks)
        self._sub_lvl = sub_lvl
        self._episode_len = ep_len
        self._delta_resource = delta_res
        self._support_factor_fsc = sup_fsc
        self._support_ratio = sup_ratio
        self._mode = mode

        # split to train and test data
        shared_train, shared_test = split_data(shared)
        shell_train, shell_test = split_data(shell)
        if mode == 'train':
            self._shared_data_orig = shared_train
            self._shell_data_orig = shell_train
        else:
            self._shared_data_orig = shared_test
            self._shell_data_orig = shell_test

        # set initial states
        self.__init_support = pd.DataFrame(init_sup, index=['support'], columns=agt)
        self.__init_resource = pd.DataFrame(init_res, index=agt, columns=agt)

        # create other parameters
        self._shared_data = None
        self._shell_data = None
        self._support = None
        self._resource = None
        self._current_step = None
        self.reward_shell_split = None

    def calc_reward(self, obs) -> Dict[str, float]:
        resource = self._resource
        sub_lvl = self._sub_lvl
        step = self._current_step
        sd_data = self._shared_data
        sl_data = self._shell_data

        # calculate reward for FSC, subtract support of FSC as it is not changing
        r = {'FSC': obs[0].loc['support'].sum() - 1}

        # calculate reward for Shell
        own_return = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
                      sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
        r_shell =\
            resource['Shell']['Shell'] * own_return + sub_lvl * resource.loc['Shell', 'FSC']\
            + sub_lvl * self._support.loc['support', 'Shell'] * resource['Shell']['Shell']
        r.update({'Shell': r_shell})

        self.reward_shell_split = np.array([r_shell, resource['Shell']['Shell'] * own_return,
                                            sub_lvl*resource.loc['Shell', 'FSC'],
                                            sub_lvl*self._support.loc['support', 'Shell']*resource['Shell']['Shell']])
        return r

    def check_if_done(self) -> bool:
        current_step = self._current_step
        # minus one as the first step is performed with step=0, taking the iloc[0] of the data DataFrames
        if current_step == self._episode_len:
            return True
        else:
            return False

    def reset(self) -> List[pd.DataFrame]:
        self._current_step = 0
        support = self.__init_support
        resource = self.__init_resource
        # set support and resource to initial values and return the state
        self._support = copy.copy(support)
        self._resource = copy.copy(resource)
        # set starting point of external data
        self.set_data()
        return [copy.copy(support), copy.copy(resource)]

    def set_data(self) -> NoReturn:
        # set starting point of the external data randomly for reward calculation
        ep_len = self._episode_len
        sl = copy.copy(self._shell_data_orig)
        sd = copy.copy(self._shared_data_orig)
        start = np.random.choice(range(0, sd.shape[0] - ep_len + 1))

        self._shell_data = sl.iloc[start:start + ep_len]
        self._shared_data = sd.iloc[start:start + ep_len]

    def step(self, actions) -> (List[pd.DataFrame], Dict[str, float], bool):
        support = self._support
        support_factor = self._support_factor_fsc
        support_ratio = self._support_ratio
        resource = self._resource
        delta_resource = self._delta_resource

        # set the support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(support)
        for agt in support.keys():
            add_val = 0
            # FSC has fixed support
            if agt != 'FSC':
                for par_agt in support.keys():
                    # agents do not due to resource assignment to itself
                    if agt != par_agt:
                        # change to partner agent is only initiated, support is different
                        if orig_support.loc['support', agt] < orig_support.loc['support', par_agt]:
                            factor = support_factor
                        elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                            factor = -support_factor
                        else:
                            factor = 0
                        if par_agt == 'Gov' or par_agt == 'Shell':
                            factor = factor * support_ratio
                        add_val += factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2

                # change in support is based on previous support --> copy is used
                # check that support remains between 0 and 1
                new_val = orig_support.loc['support', agt] + add_val
                if 0.0 > new_val:
                    support.loc['support', agt] = 0
                elif new_val > 1.0:
                    support.loc['support', agt] = 1
                else:
                    support.loc['support', agt] = new_val

        # set the resource assignment based on the agents' actions
        orig_resource = copy.copy(resource)
        for act_agt in actions.keys():
            # Gov is passiv
            if act_agt != 'Gov':
                for part_agt in actions[act_agt].keys():
                    # update value for resource assignment if it is between 0 and 1
                    val = resource.loc[act_agt, part_agt] + delta_resource * actions[act_agt][part_agt]
                    if (0.0 <= val) and (val <= 1.0):
                        resource.loc[act_agt, part_agt] = val

                # set the resource assignment of the agent to itself
                # as a result of the resource assignment to the other agents
                ext_assign = resource.loc[act_agt].sum() - resource.loc[act_agt, act_agt]
                resource.loc[act_agt, act_agt] = 1 - ext_assign

                # if actions are not feasible, keep the old resource assignment
                if (True in (resource.loc[act_agt][:] < 0).values) or \
                        (True in (resource.loc[act_agt][:] > 1).values):
                    print('---------------------    Actions {} of agent {} were not feasible.    ---------------------' \
                          .format(actions[act_agt], act_agt))
                    resource.loc[act_agt] = orig_resource.loc[act_agt]

        # check if resource calculations were correct
        if (resource.sum(axis=1).sum() / len(resource.index) != 1) or \
                (True in (resource[:][:] < 0).values) or (True in (resource[:][:] > 1).values):
            raise ValueError('Resource assignment calculation went wrong')

        observation = [copy.copy(support), copy.copy(resource)]
        reward = self.calc_reward(observation)
        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return observation, reward, done

    def step_calc(self, actions) -> (List[pd.DataFrame], Dict, bool, Dict, np.ndarray):
        support = self._support
        support_factor = self._support_factor_fsc
        support_ratio = self._support_ratio
        resource = self._resource
        delta_resource = self._delta_resource
        sup_calc = {'Shell': [], 'Gov': []}

        # set the support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(support)
        for agt in support.keys():
            # FSC has fixed support
            if agt != 'FSC':
                add_val = 0
                sup_calc[agt].append(support.loc['support'][agt])
                for par_agt in support.keys():
                    # agents do not due to resource assignment to itself
                    if agt != par_agt:
                        # change to partner agent is only initiated, support is different
                        if orig_support.loc['support', agt] < orig_support.loc['support', par_agt]:
                            factor = support_factor
                        elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                            factor = -support_factor
                        else:
                            factor = 0
                        if par_agt == 'Gov' or par_agt == 'Shell':
                            factor = factor * support_ratio
                        sup_calc[agt].append(factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2)
                        add_val += factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2

                # change in support is based on previous support --> copy is used
                # check that support remains between 0 and 1
                new_val = orig_support.loc['support', agt] + add_val
                if 0.0 > new_val:
                    support.loc['support', agt] = 0
                elif new_val > 1.0:
                    support.loc['support', agt] = 1
                else:
                    support.loc['support', agt] = new_val
                sup_calc[agt].append(support.loc['support', agt])

        # set the resource assignment based on the agents' actions
        orig_resource = copy.copy(resource)
        for act_agt in actions.keys():
            # Gov is passiv
            if act_agt != 'Gov':
                for part_agt in actions[act_agt].keys():
                    # update value for resource assignment if it is between 0 and 1
                    val = resource.loc[act_agt, part_agt] + delta_resource * actions[act_agt][part_agt]
                    if (0.0 <= val) and (val <= 1.0):
                        resource.loc[act_agt, part_agt] = val

                # set the resource assignment of the agent to itself
                # as a result of the resource assignment to the other agents
                ext_assign = resource.loc[act_agt].sum() - resource.loc[act_agt, act_agt]
                resource.loc[act_agt, act_agt] = 1 - ext_assign

                # if actions are not feasible, keep the old resource assignment
                if (True in (resource.loc[act_agt][:] < 0).values) or \
                        (True in (resource.loc[act_agt][:] > 1).values):
                    print('--------------------    Actions {} of agent {} were not feasible.    --------------------' \
                          .format(actions[act_agt], act_agt))
                    resource.loc[act_agt] = orig_resource.loc[act_agt]

        # check if resource calculations were correct
        if (resource.sum(axis=1).sum() / len(resource.index) != 1) or \
                (True in (resource[:][:] < 0).values) or (True in (resource[:][:] > 1).values):
            raise ValueError('Resource assignment calculation went wrong')

        observation = [copy.copy(support), copy.copy(resource)]
        reward = self.calc_reward(observation)
        # update step
        self._current_step += 1
        done = self.check_if_done()
        return observation, reward, done, sup_calc, self.reward_shell_split


def load_data(agents: list, num_weeks) -> [pd.DataFrame, pd.DataFrame]:
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

    # aggregate weekly data of leading_df
    df_lead = pd.DataFrame()
    for j in range(0, leading_df.shape[0], num_weeks):
        df_lead = df_lead.append(pd.DataFrame(
            leading_df['CO2_price'].iloc[[i for i in range(j, j + num_weeks) if i < leading_df.shape[0]]].mean(axis=0),
            columns=['CO2_price'],
            index=[leading_df.index[j]]))

    # aggregate weekly data of shell_data (sum of data is taken instead of mean)
    tmp = []
    for k, col in enumerate(shell_data.columns):
        df = pd.DataFrame()
        for j in range(0, shell_data.shape[0], num_weeks):
            df = df.append(pd.DataFrame(
                shell_data[col].iloc[[i for i in range(j, j + num_weeks) if i < shell_data.shape[0]]].sum(axis=0),
                columns=[col],
                index=[shell_data.index[j]]))
        tmp.append(df)
    df_shell = pd.concat([tmp[0], tmp[1], tmp[2]], axis=1)

    # modify the index, so that it is composed of the year and the week only
    return index_to_m_y(df_lead), index_to_m_y(df_shell)


def load_shell(ld_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: load ROC, ROA und ROE

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
        num_ent = shell.loc[(idx_val[i + 1] < shell.index) & (shell.index < idx_val[i]), 'CO2_price'].count()
        for key in shell.keys():
            if key != 'CO2_price':
                shell.loc[(idx_val[i + 1] < shell.index) & (shell.index < idx_val[i]), key] \
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


def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    test = data.loc['2019']
    train = data.drop(index='2019')

    return train, test
