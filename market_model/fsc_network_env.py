from typing import Dict, List, NoReturn
import numpy as np
import pandas as pd
import copy
import torch
from pathlib import Path

DATADIR = Path("C:/Users/fredd/Desktop/freddy/sciebo/Masterarbeit/03_Konzeptentwicklung/Daten")
EXCELFILE = "00_interesting_data.xlsx"

np.random.seed(42)


class FSCNetworkEnv(object):

    def __init__(self, agt, init_sup, init_res, sub_lvl, ep_len, delta_res, sup_fac, n_state_space,
                 mode='train',  agg_weeks=4):
        # setup environment parameters
        shared, shell = load_data(agt, agg_weeks)
        self._sub_lvl = sub_lvl
        self._episode_len = ep_len
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._mode = mode
        self._n_state_space = n_state_space

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
        r = {'FSC': np.array([0 if key == 'FSC' else obs[key][0] for key in obs.keys()]).sum()}

        # calculate reward for Shell
        own_return = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
                      sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
        r_shell =\
            resource['Shell']['Shell'] * own_return + sub_lvl * resource.loc['Shell', 'FSC']\
            + sub_lvl * self._support.loc['support', 'Shell'] * resource['Shell']['Shell']
        r.update({'Shell': r_shell})

        self.reward_shell_split = np.array([r_shell, resource['Shell']['Shell'] * own_return,
                                            sub_lvl * resource.loc['Shell', 'FSC'],
                                            sub_lvl * self._support.loc['support', 'Shell']*resource['Shell']['Shell']])
        return r

    def check_if_done(self) -> bool:
        current_step = self._current_step
        # minus one as the first step is performed with step=0, taking the iloc[0] of the data DataFrames
        if current_step == self._episode_len:
            return True
        else:
            return False

    def get_state(self, support, resource) -> Dict[str, np.ndarray]:
        state = dict()
        n_state_space = self._n_state_space
        # extract support and resource assignment for agent as array
        for key in n_state_space.keys():
            # first state entry for FSC is support of the other agents
            if key == 'FSC':
                state_0 = np.array([0 if agt == 'FSC' else support.loc['support'][agt]
                                    for agt in n_state_space.keys()]).sum()
            else:
                state_0 = np.array(support.loc['support'][key])
            state[key] = np.append(state_0, resource.loc[key])

        return state

    def reset(self) -> Dict[str, np.ndarray]:
        self._current_step = 0
        support = self.__init_support
        resource = self.__init_resource
        # set support and resource to initial values and return the state
        self._support = copy.copy(support)
        self._resource = copy.copy(resource)
        # set starting point of external data
        self.set_data()

        return self.get_state(support, resource)

    def set(self, delta_res, sup_fac, delta_search, sub_lvl):
        # set missing parameters, if they haven't been set while initializing the environment
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._sub_lvl = sub_lvl
        _ = delta_search

    def set_data(self) -> NoReturn:
        # set starting point of the external data randomly for reward calculation
        ep_len = self._episode_len
        sl = copy.copy(self._shell_data_orig)
        sd = copy.copy(self._shared_data_orig)
        start = np.random.choice(range(0, sd.shape[0] - ep_len + 1))

        self._shell_data = sl.iloc[start:start + ep_len]
        self._shared_data = sd.iloc[start:start + ep_len]

    def step(self, actions) -> (Dict[str, np.ndarray], Dict[str, float], bool):
        support = self._support
        support_factor = self._support_factor
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
                            factor = support_factor[par_agt]
                        elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                            factor = -support_factor[par_agt]
                        else:
                            factor = 0
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
                    val = resource.loc[act_agt, part_agt] + delta_resource * (actions[act_agt][part_agt] - 1)
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
            raise ValueError('Resource assignment calculation went wrong: FSCNetworkEnv.step()')

        # extract support and resource assignment for agent as array
        observation = self.get_state(support, resource)
        reward = self.calc_reward(observation)

        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return observation, reward, done

    def step_calc(self, actions) -> (Dict[str, np.ndarray], Dict, bool, Dict, np.ndarray):
        support = self._support
        support_factor = self._support_factor
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
                            factor = support_factor[par_agt]
                        elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                            factor = -support_factor[par_agt]
                        else:
                            factor = 0
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
                    val = resource.loc[act_agt, part_agt] + delta_resource * (actions[act_agt][part_agt] - 1)
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
            raise ValueError('Resource assignment calculation went wrong: FSCNetworkEnv.step_calc()')

        # extract support and resource assignment for agent as array
        observation = self.get_state(support, resource)
        reward = self.calc_reward(observation)

        # update step
        self._current_step += 1
        done = self.check_if_done()
        return observation, reward, done, sup_calc, self.reward_shell_split


class FSCNetworkEnvAlternative(object):

    def __init__(self, agt, init_sup, init_res, sub_lvl, ep_len, delta_res, sup_fac, n_state_space,
                 delta_search, base_impacts, mode='train',  agg_weeks=4):
        # setup environment parameters
        shared, shell = load_data(agt, agg_weeks)
        self._sub_lvl = sub_lvl
        self._episode_len = ep_len
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._mode = mode
        self._n_state_space = n_state_space
        self._delta_research = delta_search
        self._base_impacts = base_impacts

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
        # FSC has no resource assignment
        self.__init_resource = pd.DataFrame(init_res[1::], index=agt[1::], columns=agt)

        # create other variables and parameters
        self._shared_data = None
        self._shell_data = None
        self._support = None
        self._resource = None
        # TODO: komplett umstellen, dass states alle beinhaltet, also auch support und resource?
        self._states = dict()
        self._current_step = None
        self.reward_shell_split = None

    def calc_reward(self, obs) -> Dict[str, float]:
        resource = self._resource
        sub_lvl = self._sub_lvl
        step = self._current_step
        sd_data = self._shared_data
        sl_data = self._shell_data

        # calculate reward for FSC, subtract support of FSC as it is not changing
        r = {'FSC': np.array([0 if key == 'FSC' else obs[key][0] for key in obs.keys()]).sum()}

        # calculate reward for Shell
        own_return = (sl_data.iloc[step]['NIAT_USD'] - sl_data.iloc[step]['CO2_emission_tons'] *
                      sd_data.iloc[step]['CO2_price']) / sl_data.iloc[step]['TotCap_USD']
        r_shell =\
            resource['Shell']['Shell'] * own_return + sub_lvl * resource.loc['Shell', 'FSC']\
            + sub_lvl * self._support.loc['support', 'Shell'] * resource['Shell']['Shell']
        r.update({'Shell': r_shell})

        self.reward_shell_split = np.array([r_shell, resource['Shell']['Shell'] * own_return,
                                            sub_lvl * resource.loc['Shell', 'FSC'],
                                            sub_lvl * self._support.loc['support', 'Shell']*resource['Shell']['Shell']])
        return r

    def check_if_done(self) -> bool:
        current_step = self._current_step
        # minus one as the first step is performed with step=0, taking the iloc[0] of the data DataFrames
        if current_step == self._episode_len:
            return True
        else:
            return False

    def get_state(self, support, resource, s) -> Dict[str, np.ndarray]:
        state = dict()
        n_state_space = self._n_state_space
        # extract support and resource assignment for agent as array
        for key in n_state_space.keys():
            # first state entry for FSC is support of the other agents
            if key == 'FSC':
                state_0 = np.array([0 if agt == 'FSC' else support.loc['support'][agt]
                                    for agt in n_state_space.keys()]).sum()
                state[key] = np.append(state_0, s['FSC'][1::])
            else:
                state_0 = np.array(support.loc['support'][key])
                state[key] = np.append(state_0, resource.loc[key])

        return state

    def reset(self) -> Dict[str, np.ndarray]:
        self._current_step = 0
        support = self.__init_support
        resource = self.__init_resource
        # set support and resource to initial values and return the state
        self._support = copy.copy(support)
        self._resource = copy.copy(resource)
        # set starting point of external data
        self.set_data()

        # create initial state for FSC
        s = {'FSC': np.array([np.nan, 0, 0, 0])}
        state = self.get_state(support, resource, s)
        self._states['FSC'] = state['FSC']

        return state

    def set(self, delta_res, sup_fac, delta_search, sub_lvl):
        # set missing parameters, if they haven't been set while initializing the environment
        self._delta_resource = delta_res
        self._support_factor = sup_fac
        self._delta_research = delta_search
        self._sub_lvl = sub_lvl

    def set_data(self) -> NoReturn:
        # set starting point of the external data randomly for reward calculation
        ep_len = self._episode_len
        sl = copy.copy(self._shell_data_orig)
        sd = copy.copy(self._shared_data_orig)
        start = np.random.choice(range(0, sd.shape[0] - ep_len + 1))

        self._shell_data = sl.iloc[start:start + ep_len]
        self._shared_data = sd.iloc[start:start + ep_len]

    def step(self, actions) -> (Dict[str, np.ndarray], Dict[str, float], bool):
        support_factor = self._support_factor
        # the following variable assignment leads to a direct update of the private variables, as no copy is passed
        support = self._support
        resource = self._resource
        states = self._states
        delta_resource = self._delta_resource
        # all agents except FSC
        keys = support.keys().drop('FSC')

        # perform actions of FSC
        if actions['FSC']['All'] == 0:
            states['FSC'][1] += self._delta_research

        elif actions['FSC']['All'] == 1:
            for n, key in enumerate(keys):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][0]
            states['FSC'][1] = 0

        elif actions['FSC']['All'] == 2:
            for n, key in enumerate(keys):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][1]
            states['FSC'][1] = 0
        else:
            raise ValueError('Action for FSC is not defined: env.step()')

        # set the support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(support)
        for n, agt in enumerate(keys):
            add_val = 0
            for par_agt in keys:
                # agents do not influence themselves
                if agt != par_agt:
                    # change to partner agent is only initiated, support is different
                    if orig_support.loc['support', agt] < orig_support.loc['support', par_agt]:
                        factor = support_factor[par_agt]
                    elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                        factor = -support_factor[par_agt]
                    else:
                        factor = 0
                    add_val += factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2

            # influence of support by FSC
            add_val += resource.loc[agt, 'FSC'] * states['FSC'][n+2]
            # change in support is based on previous support --> copy is used
            # check that support remains between 0 and 1
            new_val = orig_support.loc['support', agt] + add_val
            if 0.0 > new_val:
                support.loc['support', agt] = 0
            elif new_val > 1.0:
                support.loc['support', agt] = 1
            else:
                support.loc['support', agt] = new_val

        # set the resource assignment based on the agents' actions, which assign resources
        orig_resource = copy.copy(resource)
        # resource assignment can be done by all agents except FSC (above) and passiv agents (Gov)
        keys = keys.drop('Gov')
        for act_agt in keys:
            for part_agt in actions[act_agt].keys():
                # update value for resource assignment if it is between 0 and 1
                val = resource.loc[act_agt, part_agt] + delta_resource * (actions[act_agt][part_agt] - 1)
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
            raise ValueError('Resource assignment calculation went wrong: FSCNetworkEnv.step()')

        # extract support and resource assignment for agent as array
        observation = self.get_state(support, resource, states)
        reward = self.calc_reward(observation)

        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return observation, reward, done

    def step_calc(self, actions) -> (Dict[str, np.ndarray], Dict, bool, Dict, np.ndarray):
        support_factor = self._support_factor
        # the following variable assignment leads to a direct update of the private variables, as no copy is passed
        support = self._support
        resource = self._resource
        states = self._states
        delta_resource = self._delta_resource
        sup_calc = {'Shell': [], 'Gov': []}

        # all agents except FSC
        keys = support.keys().drop('FSC')

        # perform actions of FSC
        if actions['FSC']['All'] == 0:
            states['FSC'][1] += self._delta_research

        elif actions['FSC']['All'] == 1:
            for n, key in enumerate(keys):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][0]
            states['FSC'][1] = 0

        elif actions['FSC']['All'] == 2:
            for n, key in enumerate(keys):
                states['FSC'][n+2] = states['FSC'][1] * self._base_impacts[key][1]
            states['FSC'][1] = 0
        else:
            raise ValueError('Action for FSC is not defined: env.step()')

        # set the support based on the previous resource assignment -> Therefore, it is the first calculation.
        orig_support = copy.copy(support)
        for n, agt in enumerate(keys):
            add_val = 0
            sup_calc[agt].append(support.loc['support'][agt])
            for par_agt in keys:
                # agents do not influence themselves
                if agt != par_agt:
                    # change to partner agent is only initiated, support is different
                    if orig_support.loc['support', agt] < orig_support.loc['support', par_agt]:
                        factor = support_factor[par_agt]
                    elif orig_support.loc['support', agt] > orig_support.loc['support', par_agt]:
                        factor = -support_factor[par_agt]
                    else:
                        factor = 0
                    sup_calc[agt].append(factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2)
                    add_val += factor * (resource.loc[agt, par_agt] + resource.loc[par_agt, agt]) / 2

            # influence of support by FSC
            sup_calc[agt].append(resource.loc[agt, 'FSC'] * states['FSC'][n+2])
            add_val += resource.loc[agt, 'FSC'] * states['FSC'][n+2]
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

        # set the resource assignment based on the agents' actions, which assign resources
        orig_resource = copy.copy(resource)
        # resource assignment can be done by all agents except FSC (above) and passiv agents (Gov)
        keys = keys.drop('Gov')
        for act_agt in keys:
            for part_agt in actions[act_agt].keys():
                # update value for resource assignment if it is between 0 and 1
                val = resource.loc[act_agt, part_agt] + delta_resource * (actions[act_agt][part_agt] - 1)
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
            raise ValueError('Resource assignment calculation went wrong: FSCNetworkEnv.step()')

        # extract support and resource assignment for agent as array
        observation = self.get_state(support, resource, states)
        reward = self.calc_reward(observation)

        # update step and check if finished
        self._current_step += 1
        done = self.check_if_done()

        return observation, reward, done, sup_calc, self.reward_shell_split


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
