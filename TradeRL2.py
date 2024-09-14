# Load libraries
import datetime;
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns
import math
import random
import h5py
# import keyboard  # For detecting key presses
import pandas_ta as ta
import tensorflow as tf
import gc  # Added for garbage collection
from keras import layers, models, optimizers
from collections import deque

from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense


# Import data from local CSV file
dataset = read_csv('c://ml//backtesting//EURUSD240-small.csv') * 1000

# Disable the warnings
import warnings

warnings.filterwarnings('ignore')

# Check data types and shape
print(dataset.shape)
# set_option('display.width', 100)
print(dataset.head(5))
print('Null Values =', dataset.isnull().values.any())

# Fill the missing values
dataset = dataset.fillna(method='ffill')
print(dataset.head(2))

# Extract OHLC data and normalize
dataset = dataset[['Open', 'High', 'Low', 'Close']]
X = dataset.values.tolist()

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

# Import Keras libraries for model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam


# Define the Agent class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # Number of inputs (OHLC for 5 days)
        self.action_size = 3  # Sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model(model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            reset_tensorflow_keras_backend()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Define helper functions

# Format price output
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# Return sigmoid of x
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Generate an n-day state representation including OHLC data
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]

    res = []
    # Create state representation by flattening OHLC data for the past n days
    for i in range(n):
        ohlc_diff = [
            sigmoid(block[i][1] - block[i][0]),  # High - Open
            sigmoid(block[i][2] - block[i][0]),  # Low - Open
            sigmoid(block[i][3] - block[i][0])  # Close - Open
        ]
        res.extend(ohlc_diff)

    return np.array([res])


# Plot the behavior of the output
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    plt.title('Total gains: %f' % (profit))
    plt.legend()
    plt.show()


# Initialize and run the agent
window_size = 5  # 5 days of OHLC data
agent = Agent(state_size=window_size * 3)  # 3 values (High-Open, Low-Open, Close-Open) for each day
data = X_train
l = len(data) - 1
batch_size = 32
episode_count = 1


def reset_tensorflow_keras_backend():
    import tensorflow as tf
    import tensorflow.keras as keras
    tf.keras.backend.clear_session()
    _ = gc.collect()


for e in range(episode_count + 1):
    print("Running episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size)
        reward = 0

        if action == 1:  # Buy
            agent.inventory.append(data[t][3])  # Use Close price for buy
            states_buy.append(t)
            print("Buy: " + formatPrice(data[t][3]))

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t][3] - bought_price, 0)
            total_profit += data[t][3] - bought_price
            states_sell.append(t)
            s = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
            print(f"{s} | Sell: {formatPrice(data[t][3] )} | Profit: {formatPrice(reward)} / Total Profit: {formatPrice(total_profit)}")

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            plot_behavior([x[3] for x in data], states_buy, states_sell, total_profit)

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 2 == 0:
        agent.model.save("model_ep" + str(e))
