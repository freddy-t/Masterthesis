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

    def __init__(self, init_sup, init_res, ep_len, delta_res, beta, n_state_space,
                 delta_search, base_impacts, sub_max=0.1, mode='train', agg_weeks=4, save_calc=False):
        # setup environment parameters
        ext_data = load_data(agg_weeks)
        self._episode_len = ep_len
        self._delta_resource = delta_res
        self._beta = beta
        self._mode = mode
        self._n_state_space = n_state_space
        self._delta_research = delta_search
        self._base_impacts = base_impacts
        self._sub_max = sub_max
        self._save_calc = save_calc
        self._impacts = {'Shell': np.zeros(ep_len),
                         'Gov': np.zeros(ep_len)}

        # split to train and test data
        ext_train, ext_test = split_data(ext_data)

        # make pre-calculation for shell reward, so that it is not done in the loop
        if mode == 'train':
            self._own_return_orig = reward_pre_calc(ext_train)
        else:
            self._own_return_orig = reward_pre_calc(ext_test)

        # store initial states
        self.__init_support = np.array(init_sup)
        self.__init_resource = init_res

        # create other variables and parameters
        self._own_return = None
        self._states = dict()
        self._support = None
        self._resource = None
        self._sub_lvl = None
        self._current_step = None

    def check_if_done(self) -> bool:
        current_step = self._current_step

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
        own_return = self._own_return
        r = dict()

        # calculate reward for FSC
        # diff = np.array([0 if key == 'FSC' else obs[key][0] for key in obs.keys()]).sum()-(np.array(prev_sup).sum() - 1)

        # V1.1: based on difference of previous support and current support
        # r['FSC'] = diff

        # V1.2: based on difference of previous support and current support, but with epsilon
        # epsilon = 10**-6
        # if epsilon < diff:
        #     r['FSC'] = 1
        # else:
        #     r['FSC'] = -1

        # V2.1: reward is based on impact of FSC
        # first entry of r_shares is change due to partner (network effect) and second entry due to fsc influence
        r_fsc = r_shares.sum()
        # r = {'FSC': r_fsc}

        # V2.2: reward is based on impact of FSC
        eps = 10**-4
        if r_fsc > eps:
            r = {'FSC': r_fsc}
        else:
            r = {'FSC': -0.05}

        # calculate reward for Shell
        # as the start for the episode is random in the external data, it does not matter if we use the difference of
        # the return from step minus step-1 or step+1 and step, as it is only shifted "another time" by the reward
        # --> not using external data anymore
        # calculation
        sub_lvl = self._sub_lvl
        own_return = 0.073
        sup = self._support[1]

        profit_t = resource['Shell'][1] * (own_return * (1 - sup) + sub_lvl * sup)
        profit_t_1 = prev_res['Shell'][1] * (own_return * (1 - prev_sup[1]) + prev_sub_lvl * prev_sup[1])
        # profit_t = resource['Shell'][1] * own_return * (1 - self._support[1]) + sub_lvl * resource['Shell'][0]\
        #            + sub_lvl * self._support[1] * resource['Shell'][1]
        # profit_t_1 = prev_res['Shell'][1] * own_return * (1 - prev_sup[1]) + prev_sub_lvl * prev_res['Shell'][0] \
        #              + prev_sub_lvl * prev_sup[1] * prev_res['Shell'][1]
        r_shell = profit_t - profit_t_1

        # V1: calculation based on profit of current time step
        r['Shell'] = r_shell
        r_shell_split = np.array([r_shell, own_return * (1 - sup), sub_lvl * sup, 0])

        # r_shell_split = np.array([r_shell,
        #                           resource['Shell'][1] * own_return,
        #                           sub_lvl * resource['Shell'][0],
        #                           prev_sub_lvl * prev_sup[1] * prev_res['Shell'][1]])

        # V2.1: calculation based on difference of profit between current and previous time step
        # r['Shell'] = r_shell

        # V2.2: calculation based on difference of profit between current and previous time step and +1 and -1 rewards
        # if r_shell > 0:
        #     r['Shell'] = 1
        # else:
        #     r['Shell'] = -1
        #
        # r_shell_split = np.array([r_shell,
        #                           prev_res['Shell'][1] * own_return[step+1] - resource['Shell'][1] * own_return[step],
        #                           prev_sub_lvl * prev_res['Shell'][0] - sub_lvl * resource['Shell'][0],
        #                           prev_sub_lvl * prev_sup[1] * prev_res['Shell'][1] -
        #                           sub_lvl * self._support[1] * resource['Shell'][1]])

        return r, r_shell_split

    def set(self, delta_res, beta, delta_research):
        # set missing parameters, if they haven't been set while initializing the environment
        self._delta_resource = delta_res
        self._beta = beta
        self._delta_research = delta_research

    def set_data(self) -> NoReturn:
        # set starting point of the external data randomly for reward calculation
        ep_len = self._episode_len
        ext = copy.deepcopy(self._own_return_orig)
        start = np.random.choice(range(0, len(ext) - ep_len))
        self._own_return = ext[start:start + ep_len + 1]

    def step(self, actions) -> (Dict[str, np.ndarray], Dict, bool, Dict, np.ndarray):

        # the following variable assignment leads to a direct update of the private variables, as no copy is passed
        support = self._support
        resource = self._resource
        states = self._states
        delta_resource = self._delta_resource

        # auxiliary variables
        fsc_par_agt = ['Shell', 'Gov']
        beta = self._beta
        prev_support = copy.deepcopy(support)
        prev_resource = copy.deepcopy(resource)
        step = self._current_step
        new_impact = {'Shell': 0,
                        'Gov': 0}
        orig_impacts = self._impacts
        impacts = {'Shell': np.zeros(step+1),
                   'Gov': np.zeros(step+1)}

        # perform actions of FSC
        if actions['FSC']['All'] == 0:
            states['FSC'][1] += self._delta_research
        elif actions['FSC']['All'] == 1:
            for n, key in enumerate(fsc_par_agt):
                new_impact[key] = states['FSC'][1] * self._base_impacts[key][0]
            # research state is zero
            states['FSC'][1] = 0
        elif actions['FSC']['All'] == 2:
            for n, key in enumerate(fsc_par_agt):
                new_impact[key] = states['FSC'][1] * self._base_impacts[key][1]
            # research state is zero
            states['FSC'][1] = 0
        else:
            raise ValueError('Action for FSC is not defined: env.step()')

        # store the non-decayed impact at the current step
        for key in fsc_par_agt:
            orig_impacts[key][step] = new_impact[key]

        tau = 1.5   # after 1 step 51%, after 6 steps 2%
        # perform exponential decay on impacts of FSC (current step is not included)
        for key in orig_impacts.keys():
            for i in range(step+1):
                impacts[key][i] = orig_impacts[key][i] * np.exp(- (step-i) / tau)

        for n, key in enumerate(fsc_par_agt):
            states['FSC'][n+2] = impacts[key].sum()

        # calculate support for Shell and Gov
        denom1 = prev_resource['Shell'][0] + prev_resource['Shell'][2]
        support[1] = prev_support[1] + beta * \
                     (prev_resource['Shell'][2] / denom1 * (prev_support[2] - prev_support[1])
                      + prev_resource['Shell'][0] / denom1 * states['FSC'][2])
        denom2 = np.array(prev_resource['Gov']).sum()
        support[2] = prev_support[2] + beta * \
                     (prev_resource['Gov'][1] / denom2 * (prev_support[1] - prev_support[2])
                      + prev_resource['Gov'][0] / denom2 * states['FSC'][3])

        r_shares_fsc = np.array([prev_resource['Shell'][0] / denom1 * states['FSC'][2],
                                 prev_resource['Gov'][0] / denom2 * states['FSC'][3]])

        # change negative supports to 0 and larger than 1 to 1
        support = np.array([val if val > 0 else 0 for val in support])
        support = np.array([val if val < 1 else 1 for val in support])

        sup_calc = dict()
        if self._save_calc:
            sup_calc['Shell'] = [prev_support[1],
                                 beta * prev_resource['Shell'][2] / denom1 * (prev_support[2] - prev_support[1]),
                                 beta * prev_resource['Shell'][0] / denom1 * states['FSC'][2],
                                 support[1]]
            sup_calc['Gov'] = [prev_support[2],
                               beta * prev_resource['Gov'][1] / denom2 * (prev_support[1] - prev_support[2]),
                               beta * prev_resource['Gov'][0] / denom2 * states['FSC'][3],
                               support[2]]

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
        rewards, r_shell_split = self.reward(states, prev_support, prev_resource, prev_sub, r_shares_fsc)

        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return states, rewards, done, sup_calc, r_shell_split


def load_data(num_weeks) -> [pd.DataFrame, pd.DataFrame]:

    # if data was already created, just load it
    if (DATADIR / 'df_ext_agg_weeks_{}'.format(num_weeks)).is_file():
        print('--------------------------      loaded saved data successfully      --------------------------')
        return torch.load((DATADIR / 'df_ext_agg_weeks_{}'.format(num_weeks)))
        # return torch.load((DATADIR / 'df_lead_agg_weeks_{}'.format(num_weeks))), \
        #        torch.load((DATADIR / 'df_shell_agg_weeks_{}'.format(num_weeks)))

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
    shell_data = load_shell(leading_df)

    # aggregate weekly data of leading_df
    df_lead = pd.DataFrame()
    for j in range(0, leading_df.shape[0], num_weeks):
        df_lead = df_lead.append(pd.DataFrame(
            leading_df['CO2_price'].iloc[[i for i in range(j, j + num_weeks) if i < leading_df.shape[0]]].mean(axis=0),
            columns=['CO2_price'], index=[leading_df.index[j]]))

    # aggregate weekly data of shell_data (sum of data is taken instead of mean for some columns)
    tmp = []
    for k, col in enumerate(shell_data.columns):
        df = pd.DataFrame()
        for j in range(0, shell_data.shape[0], num_weeks):
            df = df.append(pd.DataFrame(
                shell_data[col].iloc[[i for i in range(j, j + num_weeks) if i < shell_data.shape[0]]].mean(axis=0),
                columns=[col], index=[shell_data.index[j]]))
        tmp.append(df)
    df_shell = pd.concat([i for i in tmp], axis=1)

    print('--------------------------------    Data successfully loaded.    --------------------------------')

    # modify the index, so that it is composed of the year and the week only
    external_data = index_to_m_y(df_lead.join(df_shell))

    torch.save(external_data, (DATADIR / 'df_ext_agg_weeks_{}'.format(num_weeks)))

    return external_data


def load_shell(ld_df: pd.DataFrame) -> pd.DataFrame:
    # load excel and set index
    shell_orig = pd.read_excel(DATADIR / EXCELFILE, 'shell',
                               usecols=[1, 11, 12, 13, 14, 15, 16], nrows=40).set_index(['Date'])

    # shift shell entries and save index values for further data handling
    shell_orig = shell_orig.shift(1, freq='h')
    idx_val = shell_orig.index

    # shift the index by two to get sundays and join DataFrames
    ld_df = ld_df.shift(2, freq='D')
    shell = ld_df.append(shell_orig).sort_index()

    # remove 2020 data, as there is no quarterly data available
    shell.drop(shell[shell.index > pd.Timestamp(year=2020, month=1, day=1)].index, inplace=True)

    # assign quarterly data to the corresponding weeks of the quarter
    tmp = len(idx_val) - 1
    for i in range(tmp):
        for key in shell.keys():
            if key != 'CO2_price':
                shell.loc[(idx_val[i + 1] < shell.index) & (shell.index < idx_val[i]), key] = shell.loc[idx_val[i], key]

    # assign quarterly data of first quarter
    for key in shell.keys():
        if key != 'CO2_price':
            shell.loc[shell.index < idx_val[tmp], key] = shell.loc[idx_val[tmp], key]

    # remove shifted shell values
    shell.dropna(axis='index', how='any', subset=['CO2_price'], inplace=True)

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


def reward_pre_calc(df):
    ln = df.shape[0]
    own_return = np.zeros(ln)
    for s in range(ln):
        own_return[s] = 1/3 * ((df.iloc[s]['ROE'] + df.iloc[s]['ROA'] + df.iloc[s]['ROC']) - df.iloc[s]['CO2_price'] *
                               (df.iloc[s]['m_CO2_TotCap'] + df.iloc[s]['m_CO2_TotAss'] + df.iloc[s]['m_CO2_ShEq']))
    return own_return
