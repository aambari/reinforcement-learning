# Load libraries
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import random
import h5py
import keyboard  # For detecting key presses
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
        print("Error: " + str(e))

# The data already obtained from yahoo finance is imported.
dataset = read_csv('c://ml//backtesting//EURUSD240-small.csv') * 1000

# Disable the warnings
import warnings

warnings.filterwarnings('ignore')

# Checking for null values and removing them
print('Null Values =', dataset.isnull().values.any())

# Fill the missing values with the last value available in the dataset
dataset = dataset.fillna(method='ffill')

# Only select the relevant columns: Open, High, Low, Close
dataset = dataset[['Open', 'High', 'Low', 'Close']]

# Split dataset into training and validation sets
validation_size = 0.2
train_size = int(len(dataset) * (1 - validation_size))
X_train, X_test = dataset[0:train_size], dataset[train_size:len(dataset)]

# Add a new variable for loading and training an existing model
train_from_scratch = True  # Set this to False to load and continue training


class Agent:
    def __init__(self, state_size, is_tradeready=False, model_name="", train_from_scratch=False):
        self.state_size = state_size  # Use the passed state_size instead of calculating it here
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_tradeready = is_tradeready

        self.gamma = 0.95
        self.epsilon = 1.0 if train_from_scratch else 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        if is_tradeready or not train_from_scratch:
            # Load the model
            self.model = load_model(model_name)
            # Check if the input shape matches the current state_size
            if self.model.input_shape[1] != self.state_size:
                print(f"Rebuilding model with new input shape: {self.state_size}")
                self.model = self._model()  # Rebuild the model with the correct input shape
            print(f"Loaded model {model_name} for {'trading' if is_tradeready else 'continued training'}")
        else:
            self.model = self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def reward_function(self, action, next_state, bought_price=None):
        # Define rewards
        buy_reward = 0
        sell_reward = 0
        inactive_reward = -0.1  # Penalize inaction
        trade_reward = 0.1  # Reward for frequent trades

        # Calculate reward
        if action == 1:  # Buy
            reward = buy_reward
        elif action == 2:  # Sell
            # Reward for selling at a higher price than buying
            reward = 2 * (next_state[0][9] - bought_price) if bought_price else 0
            reward += trade_reward  # Additional reward for selling
        elif action == 0:  # Do nothing
            reward = inactive_reward

        return reward

    def action(self, state):
        if not self.is_tradeready and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def save_model(self, model_name):
        self.model.save(model_name)
        print(f'Model saved as {model_name}')

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            # Calculate reward using the reward_function
            if action == 2 and len(self.inventory) > 0:
                bought_price = self.inventory.pop(0)
            else:
                bought_price = None
            reward = self.reward_function(action, next_state, bought_price)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            reset_tensorflow_keras_backend()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("Epsilon -- " + str(self.epsilon))

def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    _ = gc.collect()

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Modify the getState function to include OHLC

def getState(data, t, n):
    d = t - n + 1
    block = data[max(d, 0):t + 1].values.tolist()

    if d < 0:
        block = (-d * [data.iloc[0].values.tolist()]) + block

    res = []
    for i in range(n - 1):
        # 10 features per time step
        res.append(sigmoid(block[i + 1][0] - block[i][0]))  # Open prices
        res.append(sigmoid(block[i + 1][1] - block[i][1]))  # High prices
        res.append(sigmoid(block[i + 1][2] - block[i][2]))  # Low prices
        res.append(sigmoid(block[i + 1][3] - block[i][3]))  # Close prices
        res.append(sigmoid(block[i + 1][3] - block[i + 1][0]))  # Close - Open
        res.append(sigmoid(block[i + 1][1] - block[i + 1][2]))  # High - Low
        res.append(sigmoid(block[i + 1][3] - block[i + 1][2]))  # Close - Low
        res.append(sigmoid(block[i + 1][1] - block[i + 1][3]))  # High - Close
        res.append(sigmoid(block[i + 1][1] - block[i + 1][0]))  # High - Open
        res.append(sigmoid(block[i + 1][0] - block[i + 1][2]))  # Open - Low

    return np.array([res])

# Check for 'Ctrl+M' press to save model
def check_save_model(agent, episode):
    if keyboard.is_pressed('ctrl+m'):
        save_name = os.path.splitext(__file__)[0] + f"_ep{e}.h5"
        agent.save_model(save_name)
        print(f'Model saved manually as {save_name}')

# Update state size based on OHLC values.
window_size = 3  # same as before
state_size = window_size * 10  # Each state will now have high, low, open, close differences
agent = Agent(state_size, model_name='', train_from_scratch=True)
data = X_train
l = len(data) - 1
batch_size = 32
episode_count = 5
max_trades = 3

# Run episodes
for e in range(episode_count + 1):
    print("Running episode " + str(e) + "/" + str(episode_count))
    print(f"Starting episode {e} with epsilon: {agent.epsilon}")
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []

    for t in range(l):
        state = getState(data, t, window_size + 1)
        action = agent.action(state)
        next_state = getState(data, t + 1, window_size + 1)

        reward = 0
        if action == 1 and len(agent.inventory) <= max_trades:  # buy
            agent.inventory.append(data.iloc[t]["Close"])
            states_buy.append(t)
            print("Buy: " + formatPrice(data.iloc[t]["Close"]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = data.iloc[t]["Close"] - bought_price
            total_profit += reward
            states_sell.append(t)
            s = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
            print(f"{s} | Sell: {formatPrice(data.iloc[t]['Close'])} | Profit: {formatPrice(reward)} / Total Profit: {formatPrice(total_profit)}")

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

    # Update reward in memory using the reward_function
        if len(agent.memory) > 1:
            last_state, last_action, _, _, _ = agent.memory[-2]
            updated_reward = agent.reward_function(last_action, state)
            agent.memory[-1] = (last_state, last_action, updated_reward, next_state, done)

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

            check_save_model(agent, e)

    # agent.save_model(f"model_ep{e}.h5")
    agent.save_model(os.path.splitext(__file__)[0] + f"_ep{e}.h5")

    reset_tensorflow_keras_backend()

agent.save_model(os.path.splitext(__file__)[0] + "_final.h5")


# Optional: Plot the trading performance
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.plot(data['Close'], label='Close Price')
plt.plot([data.iloc[i]["Close"] for i in states_buy], label='Buy', marker='^', color='g')
plt.plot([data.iloc[i]["Close"] for i in states_sell], label='Sell', marker='v', color='r')
plt.title('Trading Performance')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()