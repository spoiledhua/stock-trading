from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from StockTradingEnv import StockTradingEnv

from sys import argv, exit, stderr

import numpy as np
import pandas as pd

def main(argv):

    if len(argv) < 2:
        print('Please specify one of the following stocks: MSFT, JNJ, WMT, V, DIS', file=stderr)
        exit(1)

    valid_inputs = ['MSFT', 'JNJ', 'WMT', 'V', 'DIS']

    stock = argv[1]

    if stock not in valid_inputs:
        print('Please specify one of the following stocks: MSFT, JNJ, WMT, V, DIS', file=stderr)
        exit(1)

    # read in training data
    df_train = pd.read_excel('./stocks/{}_Prices.xlsx'.format(stock))

    # convert to date format
    dates = np.array([])
    for i in range(0, len(df_train)):
        curr_date = str(df_train.at[i, 'Date'])
        dates = np.append(dates, np.array([curr_date[4:6] + '/' + curr_date[6:] + '/' + curr_date[:4]]))
    dates = pd.Series(dates)
    df_train['Date'] = pd.to_datetime(dates)

    # filter to train only on data between the specified dates
    date_before = pd.Timestamp('2019-01-01 00:00:00', tz=None)
    date_after = pd.Timestamp('2016-12-31 00:00:00', tz=None)

    df_train = df_train[df_train['Date'] < date_before]
    df_train = df_train[df_train['Date'] > date_after]

    # build the environment
    env_train = DummyVecEnv([lambda: StockTradingEnv(df_train, True)])

    # train the model
    model = PPO2(MlpPolicy, env_train, verbose=1)
    model.learn(total_timesteps=25000)
    model.save('PPO2_agent')


if __name__ == '__main__':
    main(argv)