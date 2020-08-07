from typing import Dict, List, NoReturn
import numpy as np
import pandas as pd
import copy
import torch
from pathlib import Path

DATADIR = Path("C:/Users/fredd/Desktop/freddy/sciebo/Masterarbeit/03_Konzeptentwicklung/Daten")
EXCELFILE = "00_interesting_data.xlsx"

np.random.seed(42)


class FSCNetworkEnvAlternative(object):

    def __init__(self, agt, init_sup, init_res, ep_len, delta_res, sup_fac, n_state_space,
                 delta_search, base_impacts, sub_max=0.1, mode='train', agg_weeks=4, save_calc=False):
        # setup environment parameters
        shared, shell = load_data(agt, agg_weeks)
        self._episode_len = ep_len
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._mode = mode
        self._n_state_space = n_state_space
        self._delta_research = delta_search
        self._base_impacts = base_impacts
        self._sub_max = sub_max
        self._save_calc = save_calc

        # split to train and test data
        shared_train, shared_test = split_data(shared)
        shell_train, shell_test = split_data(shell)
        if mode == 'train':
            self._shared_data_orig = shared_train
            self._shell_data_orig = shell_train
        else:
            self._shared_data_orig = shared_test
            self._shell_data_orig = shell_test

        # store initial states
        self.__init_support = np.array(init_sup)
        self.__init_resource = init_res

        # create other variables and parameters
        self._shared_data = None
        self._shell_data = None
        self._states = dict()
        self._support = None
        self._resource = None
        self._sub_lvl = None
        self._current_step = None
        self.reward_shell_split = None

    def check_if_done(self) -> bool:
        current_step = self._current_step
        # minus one as the first step is performed with step=0, taking the iloc[0] of the data DataFrames
        if current_step == self._episode_len:
            return True
        else:
            return False

    def get_state(self, support, resource, s_fsc) -> Dict[str, np.ndarray]:
        state = dict()
        # extract support and resource assignment for agent as array
        state['FSC'] = np.append(support[1:].sum(), s_fsc)
        state['Shell'] = np.append(support[1], resource['Shell'])
        state['Gov'] = np.append(support[2], resource['Gov'])

        return state

    def reset(self) -> Dict[str, np.ndarray]:
        self._current_step = 0
        support = self.__init_support
        resource = self.__init_resource

        # set support and resource to initial values and return the initial state
        self._support = copy.deepcopy(support)
        self._resource = copy.deepcopy(resource)

        # set starting point of external data
        self.set_data()

        # create initial states
        state = self.get_state(support, resource, np.array([0, 0, 0]))
        self._states = state
        self._sub_lvl = self._sub_max * support[2]

        return state

    def reward(self, obs, prev_sup, prev_res, prev_sub_lvl, r_shares) -> Dict[str, float]:
        resource = self._resource
        step = self._current_step
        sd_data = self._shared_data
        sl_data = self._shell_data
        r = dict()

        # calculate reward for FSC base on difference of previous support
        r['FSC'] = np.array([0 if key == 'FSC' else obs[key][0] for key in obs.keys()]).sum() - \
                   (np.array(prev_sup).sum() - 1)
        # print('reward diff: {}'.format(r['FSC']))
        # reward is based on previous support
        # diff = np.array([0 if key == 'FSC' else obs[key][0] for key in obs.keys()]).sum() - \
        #        np.array([0 if key == 'FSC' else orig_sup[key][0] for key in orig_sup.keys()]).sum()
        # epsilon = 10**-6
        # if epsilon < diff:
        #     r = {'FSC': 1}
        # else:
        #     r = {'FSC': -1}
        # reward is based on difference for each agent which is caused by FSC only
        # first entry of r_shares is change due to partner (network effect) and second entry due to fsc influence
        # r_fsc = 0
        # for key in r_shares.keys():
        #     r_fsc += r_shares[key][1] - np.abs(r_shares[key][0])
        # if r_fsc > 0:
        #     r = {'FSC': 1}
        # else:
        #     r = {'FSC': -1}
        # calculate reward for Shell based on profit at the current time step
        # own_return = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
        #               sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
        # r_shell= \
        #         resource['Shell']['Shell'] * own_return + sub_lvl * resource.loc['Shell', 'FSC'] \
        #         + sub_lvl * self._support.loc['support', 'Shell'] * resource['Shell']['Shell']
        # calculate reward for Shell based on profit of difference between current and previous time step
        if step == 0:
            r['Shell'] = 0
        else:
            sub_lvl = self._sub_lvl
            own_return_t = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
                            sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
            own_return_t_1 = (sl_data.iloc[step-1]['NIAT_USD'] - sl_data.iloc[step-1]['CO2_emission_tons'] *
                              sd_data.iloc[step-1]['CO2_price']) / sl_data.iloc[step-1]['TotCap_USD']
            profit_t =\
                      resource['Shell'][1] * own_return_t + sub_lvl * resource['Shell'][0]\
                      + sub_lvl * self._support[1] * resource['Shell'][1]
            profit_t_1 = \
                        prev_res['Shell'][1] * own_return_t_1 + prev_sub_lvl * prev_res['Shell'][0] \
                        + prev_sub_lvl * prev_sup[1] * prev_res['Shell'][1]
            r_shell = profit_t - profit_t_1

            if r_shell > 0:
                r['Shell'] = 1
            else:
                r['Shell'] = -1

        # self.reward_shell_split = np.array([r_shell, resource['Shell']['Shell'] * own_return,
        #                                     sub_lvl * resource.loc['Shell', 'FSC'],
        #                                   sub_lvl * self._support.loc['support', 'Shell']*resource['Shell']['Shell']])

        return r

    def set(self, delta_res, sup_fac, delta_search):
        # set missing parameters, if they haven't been set while initializing the environment
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._delta_research = delta_search

    def set_data(self) -> NoReturn:
        # set starting point of the external data randomly for reward calculation
        ep_len = self._episode_len
        sl = copy.deepcopy(self._shell_data_orig)
        sd = copy.deepcopy(self._shared_data_orig)
        start = np.random.choice(range(0, sd.shape[0] - ep_len + 1))

        self._shell_data = sl.iloc[start:start + ep_len]
        self._shared_data = sd.iloc[start:start + ep_len]

    def step(self, actions) -> (Dict[str, np.ndarray], Dict, bool, Dict, np.ndarray):

        # the following variable assignment leads to a direct update of the private variables, as no copy is passed
        support = self._support
        resource = self._resource
        states = self._states
        delta_resource = self._delta_resource

        # auxiliary variables
        fsc_par_agt = ['Shell', 'Gov']
        support_factor = self._support_factor['Shell']
        prev_support = copy.deepcopy(support)
        prev_resource = copy.deepcopy(resource)

        # perform actions of FSC
        if actions['FSC']['All'] == 0:
            states['FSC'][1] += self._delta_research
            # set influence of FSC to zero
            for n, key in enumerate(fsc_par_agt):
                states['FSC'][n+2] = 0
        elif actions['FSC']['All'] == 1:
            for n, key in enumerate(fsc_par_agt):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][0]
            states['FSC'][1] = 0
        elif actions['FSC']['All'] == 2:
            for n, key in enumerate(fsc_par_agt):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][1]
            states['FSC'][1] = 0
        else:
            raise ValueError('Action for FSC is not defined: env.step()')

        # calculate support for Shell and Gov
        support[1] = prev_support[1] + prev_resource['Shell'][2] * (prev_support[2] - prev_support[1])\
                     + states['FSC'][2] * prev_resource['Shell'][0]
        support[2] = prev_support[2] + prev_resource['Gov'][1] * (prev_support[1] - prev_support[2])\
                     + states['FSC'][3] * prev_resource['Gov'][0]

        # change negative supports to 0 and larger than 1 to 1
        support = np.array([val if val > 0 else 0 for val in support])
        support = np.array([val if val < 1 else 1 for val in support])

        if self._save_calc:
            sup_calc = dict()
            sup_calc['Shell'] = [prev_support[1], prev_resource['Shell'][2] * (prev_support[2] - prev_support[1]),
                                 states['FSC'][2] * prev_resource['Shell'][0], support[1]]
            sup_calc['Gov'] = [prev_support[2], prev_resource['Gov'][1] * (prev_support[1] - prev_support[2]),
                               states['FSC'][3] * prev_resource['Gov'][0], support[2]]
            r_shares_fsc = {'Shell': [], 'Gov': []}

        # set level of subsidies
        prev_sub = copy.deepcopy(self._sub_lvl)
        self._sub_lvl = self._sub_max * support[2]

        # set the resource assignment based on Shells actions
        for part_agt in actions['Shell'].keys():
            # set index according to partner agent
            if part_agt == 'FSC':
                i = 0
            else:
                i = 2
            # update value for resource assignment if it is between 0 and 1
            val_par_ = prev_resource['Shell'][i] + delta_resource * (actions['Shell'][part_agt] - 1)
            if (0.0 <= val_par_) and (val_par_ <= 1.0):
                resource['Shell'][i] = val_par_

        # set the resource assignment of the agent to itself as a result of the resource assignment to the other agents
        ext_assign = np.array(resource['Shell']).sum() - resource['Shell'][1]
        resource['Shell'][1] = 1 - ext_assign

        # if actions are not feasible, keep the old resource assignment
        for i in resource['Shell']:
            if i < 0 or i > 1:
                resource['Shell'] = prev_resource['Shell']
                print('--------------------    Actions {} of Shell were not feasible.    --------------------' \
                      .format(actions['Shell']))

        # check if resource calculations were correct
        for i in resource['Shell']:
            if i < 0 or i > 1:
                if np.array(resource['Shell']).sum() != 1:
                    raise ValueError('Resource assignment calculation went wrong: FSCNetworkEnv.step()')

        # extract support and resource assignment for agent as array
        states = self.get_state(support, resource, states['FSC'][1::])
        rewards = self.reward(states, prev_support, prev_resource, prev_sub, r_shares_fsc)

        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return states, rewards, done, sup_calc, self.reward_shell_split


def load_data(agents: list, num_weeks) -> [pd.DataFrame, pd.DataFrame]:

    # if data was already created, just load it
    if (DATADIR / 'df_lead_agg_weeks_{}'.format(num_weeks)).is_file():
        print('--------------------------      loaded saved data successfully      --------------------------')
        return torch.load((DATADIR / 'df_lead_agg_weeks_{}'.format(num_weeks))), \
               torch.load((DATADIR / 'df_shell_agg_weeks_{}'.format(num_weeks)))

    # define all possible weeks to see which weekly data is missing
    file = DATADIR / EXCELFILE
    all_weeks = pd.read_excel(file, 'dates').set_index('fridays')
    co2_full = pd.read_excel(file, 'CFI2Zc1_rolling_Dec_contract').set_index('fridays')
    co2_missing = pd.read_excel(file, 'missing_data').set_index('date')
    tmp1 = all_weeks.join(co2_missing).dropna(axis='index')
    tmp2 = all_weeks.join(co2_full).dropna(axis='index')
    leading_df = pd.concat([tmp1, tmp2])

    # remove 2020 data, as there is no quarterly data available
    leading_df.drop(leading_df[leading_df.index > pd.Timestamp(year=2020, month=1, day=1)].index, inplace=True)

    # load agent specific data
    if 'Shell' in agents:
        shell_data = load_shell(leading_df)

    # aggregate weekly data of leading_df
    df_lead = pd.DataFrame()
    for j in range(0, leading_df.shape[0], num_weeks):
        df_lead = df_lead.append(pd.DataFrame(
            leading_df['CO2_price'].iloc[[i for i in range(j, j + num_weeks) if i < leading_df.shape[0]]].mean(axis=0),
            columns=['CO2_price'],
            index=[leading_df.index[j]]))

    # aggregate weekly data of shell_data (sum of data is taken instead of mean for some columns)
    tmp = []
    for k, col in enumerate(shell_data.columns):
        df = pd.DataFrame()
        for j in range(0, shell_data.shape[0], num_weeks):
            # profitability data must be averaged
            if col == 'ROE' or col == 'ROA' or col == 'ROC':
                df = df.append(pd.DataFrame(
                    shell_data[col].iloc[[i for i in range(j, j + num_weeks) if i < shell_data.shape[0]]].mean(axis=0),
                    columns=[col],
                    index=[shell_data.index[j]]))
            else:
                df = df.append(pd.DataFrame(
                    shell_data[col].iloc[[i for i in range(j, j + num_weeks) if i < shell_data.shape[0]]].sum(axis=0),
                    columns=[col],
                    index=[shell_data.index[j]]))
        tmp.append(df)
    df_shell = pd.concat([i for i in tmp], axis=1)

    print('--------------------------------    Data successfully loaded.    --------------------------------')

    # modify the index, so that it is composed of the year and the week only
    df_lead = index_to_m_y(df_lead)
    df_shell = index_to_m_y(df_shell)
    torch.save(df_lead, (DATADIR / 'df_lead_agg_weeks_{}'.format(num_weeks)))
    torch.save(df_shell, (DATADIR / 'df_shell_agg_weeks_{}'.format(num_weeks)))

    return df_lead, df_shell


def load_shell(ld_df: pd.DataFrame) -> pd.DataFrame:
    # load excel and set index
    shell_orig = pd.read_excel(DATADIR / EXCELFILE, 'shell', usecols=[1, 3, 5, 7, 8, 9, 10]).set_index(['Date'])

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
                # profitability data must not be averaged
                if key == 'ROE' or key == 'ROA' or key == 'ROC':
                    shell.loc[(idx_val[i + 1] < shell.index) & (shell.index < idx_val[i]), key] \
                        = shell.loc[idx_val[i], key]
                else:
                    shell.loc[(idx_val[i + 1] < shell.index) & (shell.index < idx_val[i]), key] \
                        = shell.loc[idx_val[i], key] / num_ent

    # calculate weekly data for first quarter
    num_ent = shell.loc[shell.index < idx_val[tmp], 'CO2_price'].count()
    for key in shell.keys():
        if key != 'CO2_price':
            # profitability data must not be averaged
            if key == 'ROE' or key == 'ROA' or key == 'ROC':
                shell.loc[shell.index < idx_val[tmp], key] = shell.loc[idx_val[tmp], key]
            else:
                shell.loc[shell.index < idx_val[tmp], key] = shell.loc[idx_val[tmp], key] / num_ent

    # remove shifted shell values
    shell.dropna(axis='index', how='any', subset=['CO2_price'], inplace=True)

    # check if translation from quarterly to weekly data was correct
    val_test = [(shell[key].sum() - shell_orig[key].sum())**2 < 1.0 for key in shell.keys()
                if (key != 'CO2_price' and key != 'ROE' and key != 'ROA' and key != 'ROC')]
    if val_test.count(False):
        raise ValueError('Translation from quarterly to weekly data went wrong: fsc_network_env.load_shell()')

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
    test = data.loc[['2018', '2019']]
    train = data.drop(index=['2018', '2019'])

    return train, test
