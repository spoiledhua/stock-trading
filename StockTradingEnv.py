import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import random
import matplotlib.pyplot as plt

INITIAL_WALLET = 10000
MAX_NET_WORTH = 100000

class StockTradingEnv(gym.Env):
    "A stock trading environment for OpenAI gym"
    metadata = {'render.modes': ['human']}

    def __init__(self, df, randomized):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.randomized = randomized
        self.reward_range = (0, MAX_NET_WORTH)

        # Buy = 0, Sell = 1, or Hold = 2
        self.action_space = spaces.Discrete(3)

        high = np.array([])
        for i in range(12):
            high = np.append(high, [math.inf])
        high = np.append(high, 1)

        # Observation space consists of past 5 daily close prices, past 5 MACD histogram values, 
        # current wallet amount, current net worth, and an indicator of past stock peaks/valleys
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # ???
        #self.seed()
    '''
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    '''
    def step(self, action):

        # initialize reward
        reward = 0

        # assume transaction is made at closing price
        current_price = self.df['Close'].iloc[self.current_step]

        # buy action
        if action == 0 and self.wallet >= current_price:

            # buy maximum amount of shares
            max_buy = self.wallet // current_price
            cost = max_buy * current_price
            self.wallet -= cost
            self.shares_owned += max_buy

            # find last sell price, if it exists
            if len(self.buy_history) != 0:
                last_date, last_price = self.buy_history[-1]

                # add reward defined to be difference between last sell price and current buy price
                reward += last_price - current_price

            # add large reward if buy action comes right after a valley
            if self.df['Close'].iloc[self.current_step - 1] < min(self.df['Close'].iloc[self.current_step - 2], self.df['Close'].iloc[self.current_step]):
                reward += 100

            # record the transaction
            self.buy_history.append((str(self.df['Date'].iloc[self.current_step]), current_price))

        # sell action
        elif action == 1 and self.shares_owned != 0:

            # sell maximum amount of shares
            revenue = self.shares_owned * current_price
            self.wallet += revenue
            self.shares_owned = 0

            # find last buy price, if it exists
            if len(self.sell_history) != 0:
                last_date, last_price = self.sell_history[-1]

                # add reward defined to be difference between current sell price and last sell price
                reward += current_price - last_price

            # add large reward if sell action comes right after a peak
            if self.df['Close'].iloc[self.current_step - 1] > max(self.df['Close'].iloc[self.current_step - 2], self.df['Close'].iloc[self.current_step]):
                reward += 100

            # record the transaction
            self.sell_history.append((str(self.df['Date'].iloc[self.current_step]), current_price))

        # hold action
        else:

            current_net_worth = self.wallet + self.shares_owned * current_price

            # add reward defined to be difference between current net worth and yesterday's net worth
            reward += current_net_worth - self.net_worth_history[-1]

        self.net_worth = self.wallet + self.shares_owned * current_price

        # record date, agent's net worth, and LONG strategy's net worth
        self.net_worth_history.append(self.net_worth)
        self.date_history.append(str(self.df['Date'].iloc[self.current_step]))
        self.long_history.append(self.long_constant + self.long_shares * current_price)

        self.current_step += 1

        # done condition defined to be when net worth exceeds the maximum
        # or when end of time frame is reached
        done = False
        if self.net_worth <= 0 or self.net_worth >= MAX_NET_WORTH:
            done = True
        if self.current_step >= len(self.df) - 1:
            done = True

        # update observation
        obs = np.array(self.df['Close'].iloc[self.current_step - 4:self.current_step + 1])
        obs = np.append(obs, np.array(self.df['Histogram'].iloc[self.current_step - 4:self.current_step + 1]))
        obs = np.append(obs, np.array([self.wallet]))
        obs = np.append(obs, np.array([self.net_worth]))

        if self.df['Close'].iloc[self.current_step - 1] < min(self.df['Close'].iloc[self.current_step - 2], self.df['Close'].iloc[self.current_step]):
            obs = np.append(obs, np.array([-1]))
        elif self.df['Close'].iloc[self.current_step - 1] > max(self.df['Close'].iloc[self.current_step - 2], self.df['Close'].iloc[self.current_step]):
            obs = np.append(obs, np.array([1]))
        else:
            obs = np.append(obs, np.array([0]))

        return obs, reward, done, {}

    def reset(self):

        # initialize all necessary environment variables and observation
        self.wallet = INITIAL_WALLET
        self.net_worth = INITIAL_WALLET
        self.shares_owned = 0
        self.net_worth_history = [self.net_worth]
        
        if self.randomized:
            self.current_step = random.randint(75, 125)
        else:
            self.current_step = 100

        current_price = self.df['Close'].iloc[self.current_step - 1]

        self.long_shares = self.wallet // current_price
        self.long_constant = self.wallet - self.long_shares * current_price
        self.long_history = [self.long_constant + self.long_shares * current_price]
        self.date_history = [str(self.df['Date'].iloc[self.current_step - 1])]
        self.buy_history = []
        self.sell_history = []

        obs = np.array(self.df['Close'].iloc[self.current_step - 4:self.current_step + 1])
        obs = np.append(obs, np.array(self.df['Histogram'].iloc[self.current_step - 4:self.current_step + 1]))
        obs = np.append(obs, np.array([self.wallet]))
        obs = np.append(obs, np.array([self.net_worth]))
        obs = np.append(obs, np.array([0]))

        return obs

    def info(self):
        
        return [self.date_history, self.net_worth_history, self.long_history]

    def render(self, mode='human', close=False):
        
        # print variables and compare agent's strategy with LONG strategy on a graph
        profit = self.net_worth - INITIAL_WALLET
        print('Wallet: {}'.format(self.wallet))
        print('Buy history:', end='')
        print(self.buy_history)
        print('Sell history:', end='')
        print(self.sell_history)
        print('Agent Profit: {}'.format(profit))
        print('Agent Net worth: {}'.format(self.net_worth))
        print('Long Strategy Net Worth: {}'.format(self.long_history[-1] - self.long_history[0]))
        print('----------------------------------')

        plt.plot(self.date_history, self.long_history)
        plt.plot(self.date_history, self.net_worth_history)

        _ = plt.xlabel('Year', fontsize=14)
        _ = plt.ylabel('Price', fontsize=14)
        plt.legend(('Long', 'RL Agent'))
        plt.suptitle("Total Profit: {}".format(profit))

        plt.grid()
        plt.show()