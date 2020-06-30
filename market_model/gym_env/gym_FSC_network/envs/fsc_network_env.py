import gym
import pandas as pd
import numpy as np
from pathlib import Path


DATAFILE = Path("C:/Users/fredd/Desktop/freddy/sciebo/Masterarbeit/03_Konzeptentwicklung/Daten") /\
                "00_interesting_data.xlsx"


class FSCNetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def setup(self, agt: list):
        load_data(agt)


def load_data(agents: list):
    # define all possible weeks to see which weekly data is missing
    dates = pd.read_excel(DATAFILE, 'dates')
    co2_price = pd.read_excel(DATAFILE, 'EEX_EUA_Spot_Open_USD')
    leading_df = dates.set_index('fridays').join(co2_price.set_index('fridays'))

    # fill missing data with linear interpolation
    leading_df = leading_df.interpolate(axis='index')

    # load agent specific data
    if 'Shell' in agents:
        shell_data = load_shell(leading_df)
        print('--------------------------------    Shell data successfully loaded.    --------------------------------')

    # modify the index, so that it is composed of the year and the week only
    leading_df = index_to_m_y(leading_df)
    pass


def load_shell(ld_df: pd.DataFrame) -> pd.DataFrame:
    shell = pd.read_excel(DATAFILE, 'shell', usecols=[1, 3, 5, 7])
    shell = shell.set_index(['Date'])
    shell = shell.shift(1, freq='h')
    idx_val = shell.index

    # shift the index by two to get sundays
    ld_df = ld_df.shift(2, freq='D')
    shell = ld_df.append(shell).sort_index()
    # remove 2020 data
    shell = shell.drop(shell[shell.index > pd.Timestamp(year=2020, month=1, day=1)].index)
    idx_nan = shell['TotCap_USD'].index[shell['TotCap_USD'].apply(np.isnan)]
    for i in len(idx_val)-1:
        print(shell.loc[idx_val[i+1] < shell.index & shell.index < idx_val[i]])
# slicing ausprobieren ?? df[:5]
    # index zu rows machen und dann set value for rows matching condition?

    return index_to_m_y(shell)


def index_to_m_y(df: pd.DataFrame) -> pd.DataFrame:
    # modify the index, so that it is composed of the year and the week only
    df['week'] = df.index.weekofyear.astype(str)
    df['year'] = df.index.year.astype(str)
    df = df.set_index(['year', 'week'])

    return df

