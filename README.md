<h1 align="center">
  Reinforcement Learning for Stock Trading
</h1>

## Introduction

Reinforcement learning models have consistently beaten chess grandmasters and broken records playing Atari games. Needless to say, there's a whole lot of buzz around what RL has the potential to achieve, so for retail and institutional investors alike, the natural question is: Can reinforcement learning generate better returns than existing trading strategies? Surprisingly little work has been done in this area, so the goal of this project is to explore how RL can be applied. Each asset class has its own niches and intracacies; this project focuses exclusively on the equity market—blue chip stocks in particular.

## About

This project was independently developed by me in fulfillment of my independent work during Spring 2020. It is written in Python and utilizes OpenAI Gym for environment building, Stable Baselines for RL algorithm implementation, Pandas and NumPy for data processing and computations, and Matplotlib for visualization.

## Brief Overview

Throughout the course of my project, I find that the right combination of RL algorithm, the agent's observation space, reward function, and date range of the training data results in an agent that can slightly outperform the baseline "LONG" strategy, which just entails assuming a maximally long position at the beginning of the trading time range. Now, for many, if not most, individuals who don't trade professionally and are just trying to put their savings to good use, LONG isn't a bad strategy. In fact, given that the stock market averages a 10% return, it's quite a good strategy for anyone just aiming to save for retirement—$100K today becomes $1M in less than 25 years. Obviously, investors can do much better than that by actively trading.

In terms of the environment, I developed a custom Gym environment for stock trading. The agent can "see"—via its observation space—the last five days of closing prices as well as the last five days of the Moving Average Convergence Divergence (MACD) histogram, a very commonly used trend indicator. Each day, it's allowed to either buy or sell the maximum amount at the closing price, or simply hold its position. The agent is given $10K in cash initally, and its net worth at the end of the trading interval is defined to be the number of shares it owns times the closing price on the last day, plus the amount of cash it has. The agent is trained on 2017-2018 data on a single stock—MSFT, JNJ, WMT, V, or DIS—and is tested on 2019 data on that same stock, starting at 100 days after January 1st, 2019.

The long-term upwards trend of the stock market actually makes it difficult for the RL agent to find the optimal time to trade. So my algorithm essentially rewards the agent whenever it takes advantage of peaks and valleys. After training, the DQN algorithm performs quite well and actually makes almost all of its trades in line with the "buy low, sell high" mantra on a microscale. That is, it attempts to take advantage of every single peak or valley, no matter how big or small. In some cases, this does no more than 1-2% better than the LONG strategy. In others, it does much better, as is the case with MSFT where the agent outperforms LONG by over 11%.

There are countless ways to extend this project: portfolio management, reward function improvement, different financial indicators, etc. Nonetheless, it provides good intuition as to how RL can be applied to stock trading.

## Demo

* Clone the repo and cd into it:
```bash
git clone https://github.com/spoiledhua/stock-trading.git
cd stock-trading
```

* Install Stable Baselines, OpenAI Gym, NumPy, Pandas, and Matplotlib:
```bash
pip3 install stable-baselines gym numpy pandas matplotlib
```

* Train a model using one of the available RL algorithms (DQN, PPO2, A2C) on one of the available stocks (MSFT, JNJ, WMT, V, DIS). For instance, to train DQN on MSFT, run the following command:
```bash
python3 DQN_Train.py MSFT
```

* Once training is finished, the model will be saved as a .zip file in the directory. Test the model on one of the available stocks. For instance, to test DQN on MSFT, run the following command:
```bash
python3 DQN_Test.py MSFT
```
Note: Testing will throw an error if the appropriate model isn't trained first or if the corresponding .zip file is deleted.

Once the algorithm terminates, a Matplotlib plot comparing the agent's performance with the LONG strategy's performance will display on a separate window. The y-axis denotes net worth, and the plot represents both strategies' net worths througout the trading interval, in USD. The terminal will display the agent's net worth and the LONG strategy's net worth at the end of the interval, given an initial $10K.

## Example

### DQN on MSFT

<img src="https://raw.githubusercontent.com/spoiledhua/stock-trading/master/DQN_MSFT.png" width="50%" height="50%" /> 

## Additional Information

Official project write-up coming soon.
