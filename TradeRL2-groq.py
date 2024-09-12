import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import math
from numpy.random import choice
import random

# Import Model Packages for reinforcement learning
from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque

# Load the data
dataset = read_csv('c://ml//backtesting//EURUSD240.csv')*1000

# Disable the warnings
import warnings
warnings.filterwarnings('ignore')

# Define the state size (window size)
window_size = 5

# Create the Agent class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        # State size depends and is equal to the the window size, n previous days
        self.state_size = state_size  # normalized previous days,
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Create the model
        self.model = self._model()

    import keras
    from keras.models import Sequential
    from keras.models import load_model
    from keras.layers import Dense
    from keras.optimizers import Adam
    from IPython.core.debugger import set_trace
    # Deep Q Learning model- returns the q-value when given state as input
    def _model(self):
        model = Sequential()
        # Input Layer
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        # Hidden Layers
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        # Output Layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    # Return the action on the value function
    # With probability (1-$\epsilon$) choose the action which has the highest Q-value.
    # With probability ($\epsilon$) choose any action at random.
    # Intitially high epsilon-more random, later less
    # The trained agents were evaluated by different initial random condition
    # and an e-greedy policy with epsilon 0.05. This procedure is adopted to minimize the possibility of overfitting during evaluation.

    def act(self, state):
        # If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        # actions suggested.
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose=None)
        # set_trace()
        # action is based on the action that has the highest value from the q-value function.
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        # the memory during the training phase.
        for state, action, reward, next_state, done in mini_batch:
            target = reward  # reward or Q at time t
            # update the Q table based on Q table equation
            # set_trace()
            if not done:
                # set_trace()
                # max of the array of the predicted.
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # Q-value of the state currently from the table
            target_f = self.model.predict(state, verbose=0)
            # Update the output Q table for the given action in the table
            target_f[0][action] = target
            # train and fit the model where state is X and target_f is Y, where the target is updated.
            self.model.fit(state, target_f, epochs=1, verbose=None)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the function to get the state
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    # block is which is the for [1283.27002, 1283.27002]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

# Define the function to plot the behavior
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    plt.title('Total gains: %f' % (profit))
    plt.legend()
    plt.show()

# Define the function to format the price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# Define the function to calculate the total profit
def calculateTotalProfit(data, states_buy, states_sell):
    total_profit = 0
    for i in range(len(states_buy)):
        total_profit += data[states_sell[i]] - data[states_buy[i]]
    return total_profit

# Define the main function
def main():
    # Load the data
    data = dataset['Close'].values

    # Create the agent
    agent = Agent(window_size)

    # Set the batch size
    batch_size = 32

    # Set the episode count
    episode_count = 1

    for e in range(episode_count + 1):
        print("Running episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        states_sell = []
        states_buy = []
        for t in range(len(data)):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[t])
                states_buy.append(t)
                print("Buy: " + formatPrice(data[t]))

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                states_sell.append(t)
                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
                print('Total Profit --> ' + formatPrice(total_profit))

            done = True if t == len(data) - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")
                plot_behavior(data, states_buy, states_sell, total_profit)
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 2 == 0:
            agent.model.save("model_ep" + str(e))

    print("Total Profit: " + formatPrice(calculateTotalProfit(data, states_buy, states_sell)))

if __name__ == "__main__":
    main()