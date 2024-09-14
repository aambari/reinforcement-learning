# Load libraries
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
# import seaborn as sns
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


# List physical devices
gpus = tf.config.list_physical_devices('GPU')
print(str(datetime.datetime.now().time().strftime("%H:%M:%S")) + " GPUs available: ", gpus)


# Set memory growth limitation (just in case you're using GPU later)

if gpus:
    print(gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth")
    except RuntimeError as e:
        print("Errow" + str(e))



#The data already obtained from yahoo finance is imported.
# dataset = read_csv('https://raw.githubusercontent.com/tatsath/fin-ml/master/Chapter%209%20-%20Reinforcement%20Learning/Case%20Study%201%20-%20Reinforcement%20Learning%20based%20Trading%20Strategy/data/SP500.csv',index_col=0)
dataset = read_csv( 'c://ml//backtesting//EURUSD240-small.csv')*1000
#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

type(dataset)

# shape
print(dataset.shape)


# peek at data
# set_option('display.width', 100)
print(dataset.head(5))

# describe data
# set_option('precision', 3)
dataset.describe()

dataset['Close'].plot()

#Checking for any null values and removing the null values'''
print('Null Values =',dataset.isnull().values.any())

# Fill the missing values with the last value available in the dataset.
dataset=dataset.fillna(method='ffill')
print(dataset.head(2))

X=list(dataset["Close"])
X=[float(x) for x in X]

validation_size = 0.2
#In case the data is not dependent on the time series, then train and test split should be done based on sequential sample
#This can be done by selecting an arbitrary split point in the ordered list of observations and creating two new datasets.
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

# import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
# from IPython.core.debugger import set_trace

import numpy as nppip
import random
from collections import deque


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
        # self.epsilon_decay = 0.9

        # self.model = self._model()



        self.model = load_model(model_name) if is_eval else self._model()

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

    def action(self, state):
        # If it is test and self.epsilon is still very high, once the epsilon become low, there are no random
        # actions suggested.
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose = None)
        # set_trace()
        # action is based on the action that has the highest value from the q-value function.
        return np.argmax(options[0])

    # Save model weights
    def save_model(self, model_name):
        self.model.save(model_name)
        print(f'Model saved as {model_name}')


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
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                # Q-value of the state currently from the table
            target_f = self.model.predict(state, verbose = 0)
            # Update the output Q table for the given action in the table
            target_f[0][action] = target
            # train and fit the model where state is X and target_f is Y, where the target is updated.
            self.model.fit(state, target_f, epochs=1, verbose=0)
            reset_tensorflow_keras_backend()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# _______________________________________________________________________________________________


def reset_tensorflow_keras_backend():
    import tensorflow as tf
    import tensorflow.keras as keras
    tf.keras.backend.clear_session()
    _ = gc.collect()


import math



# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# def formatPrice(n):
#     return ("-$" if n < 0 else "$") + "{0:.5f}".format(abs(n)*100) #multiply by 100 to give some value to the pips

# # returns the vector containing stock data from a fixed file
# def getStockData(key):
#     vec = []
#     lines = open("data/" + key + ".csv", "r").read().splitlines()

#     for line in lines[1:]:
#         vec.append(float(line.split(",")[4])) #Only Close column

#     return vec

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    # block is which is the for [1283.27002, 1283.27002]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

# Plots the behavior of the output
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    plt.title('Total gains: %f/' % (profit))
    plt.legend()
    # plt.savefig('output/'+name+'.png')
    plt.show()



# from IPython.core.debugger import set_trace
window_size = 3
agent = Agent(window_size)
# In this step we feed the closing value of the stock price
data = X_train
l = len(data) - 1
#
batch_size = 32
# An episode represents a complete pass over the data.
episode_count = 10

max_trades = 3
# from keras import config
# config.disable_interactive_logging()
for e in range(episode_count + 1):
    print("Running episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    # set_trace()
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    for t in range(l):
        action = agent.action(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1 and len(agent.inventory) <= max_trades:  # buy
            agent.inventory.append(data[t])
            states_buy.append(t)
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            # reward = max(data[t] - bought_price, 0)
            reward = data[t] - bought_price
            total_profit += data[t] - bought_price
            states_sell.append(t)
            s = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
            print(f"{s} | Sell: {formatPrice(data[t])} | Profit: {formatPrice(reward)} / Total Profit: {formatPrice(total_profit)}")

        done = True if t == l - 1 else False
        # appends the details of the state action etc in the memory, which is used further by the exeReply function
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            # set_trace()
            # pd.DataFrame(np.array(agent.memory)).to_csv("Agent"+str(e)+".csv")
            # Chart to show how the model performs with the stock goin up and down for each
            # plot_behavior(data, states_buy, states_sell, total_profit)
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 2 == 0:
        agent.save_model(f"model_ep{e}.h5")
    reset_tensorflow_keras_backend()
agent.save_model("final_model.h5")