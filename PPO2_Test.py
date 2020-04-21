from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from StockTradingEnv import StockTradingEnv

from sys import argv, exit, stderr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TIME_STEPS = 152

def main(argv):

    if len(argv) < 2:
        print('Please specify one of the following stocks: MSFT, JNJ, WMT, V, DIS', file=stderr)
        exit(1)

    valid_inputs = ['MSFT', 'JNJ', 'WMT', 'V', 'DIS']

    stock = argv[1]

    if stock not in valid_inputs:
        print('Please specify one of the following stocks: MSFT, JNJ, WMT, V, DIS', file=stderr)
        exit(1)

    # read in testing data
    df_test = pd.read_excel('./stocks/{}_Prices.xlsx'.format(stock))

    # convert to date format
    dates = np.array([])
    for i in range(0, len(df_test)):
        curr_date = str(df_test.at[i, 'Date'])
        dates = np.append(dates, np.array([curr_date[4:6] + '/' + curr_date[6:] + '/' + curr_date[:4]]))
    dates = pd.Series(dates)
    df_test['Date'] = pd.to_datetime(dates)

    # filter to test only on data after the specified date
    date_after = pd.Timestamp('2018-12-31 00:00:00', tz=None)
    df_test = df_test[df_test['Date'] > date_after]

    # load the model
    model = PPO2.load("PPO2_bot")

    # build the environment
    env_test = StockTradingEnv(df_test, False)

    total = 0
    net_worth_history = np.ones(TIME_STEPS)

    for j in range(100):
        obs = env_test.reset()
        for i in range(10000):
            action, _state = model.predict(obs)
            obs, reward, done, info = env_test.step(action)
            if done:
                info = env_test.info()
                total += info[1][-1]
                net_worth_history = net_worth_history + np.array(info[1])
                date_history = info[0]
                long_history = info[2]
                break
    
    # render information
    net_worth_history = net_worth_history / 100

    print('Long Strategy Net Worth: {}'.format(long_history[-1]))
    print('Agent Average Net Worth: {}'.format(total/100))
    print('----------------------------------')

    plt.plot(date_history, long_history)
    plt.plot(date_history, net_worth_history)

    _ = plt.xlabel('Date', fontsize=14)
    _ = plt.ylabel('Net Worth', fontsize=14)
    plt.legend(('Long', 'RL Agent'))

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main(argv)