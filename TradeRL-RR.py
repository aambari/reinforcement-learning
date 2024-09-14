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
    except RuntimeError as e:
        print(e)

# Define EMA periods
emaPeriodFast = 5
emaPeriodSlow = 10

# Import data from local CSV file
dataset = read_csv('c://ml//backtesting//EURUSD240-small.csv') * 1000
dataset["EMAFast"] = ta.ema(dataset.Close, length=emaPeriodFast)
dataset["EMASlow"] = ta.ema(dataset.Close, length=emaPeriodSlow)

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Fill missing values
dataset = dataset.fillna(method='ffill')

# Extract OHLC data and normalize
dataset = dataset[['Open', 'High', 'Low', 'Close', 'EMAFast', 'EMASlow']] #, 'EMAFast', 'EMASlow'
X = dataset.values.tolist()

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

# Define the Agent class
class Agent:
    def __init__(self, state_size, is_eval=False, model_name="", max_inventory=3):
        self.state_size = state_size  # Number of inputs (OHLC for 5 days)
        self.action_size = 3  # Sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.max_inventory = max_inventory  # Maximum allowed buys

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Load saved model if available
        self.model = load_model(model_name) if model_name else self._model()

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

    # Save model weights
    def save_model(self, model_name):
        self.model.save(model_name)
        print(f'Model saved as {model_name}')

# Helper Functions
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# State representation for past n days
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]

    res = []
    for i in range(n):
        ohlc_diff = [
            sigmoid(block[i][1] - block[i][0]),  # High - Open
            sigmoid(block[i][2] - block[i][0]),  # Low - Open
            sigmoid(block[i][3] - block[i][0])   # Close - Open
        ]
        res.extend(ohlc_diff)

    return np.array([res])

# Plot the output behavior
def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2.)
    plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='k', label='Selling signal', markevery=states_sell)
    plt.title('Total gains: %f' % (profit))
    plt.legend()
    plt.show()

# Risk-to-reward logic
def calculate_reward(current_price, bought_price, high_since_bought, low_since_bought, risk_reward_ratio):
    profit = current_price - bought_price
    stop_loss = bought_price - low_since_bought
    take_profit = high_since_bought - bought_price
    if profit >= risk_reward_ratio * stop_loss:
        return profit
    elif stop_loss >= profit:
        return -stop_loss
    else:
        return 0

# Detect key press to save the model
# def check_save_model(agent, episode):
#     if keyboard.is_pressed('ctrl+m'):
#         save_name = f"model_ep{episode}.h5"
#         agent.save_model(save_name)
#         print(f'Model saved manually as {save_name}')

def reset_tensorflow_keras_backend():
    import tensorflow as tf
    import tensorflow.keras as keras
    tf.keras.backend.clear_session()
    _ = gc.collect()

# Initialize agent
window_size = 5
max_inventory = 1  # Max concurrent buys
agent = Agent(state_size=window_size * 3, max_inventory=max_inventory)
data = X_train
l = len(data) - 1
batch_size = 32
episode_count = 10

# Training loop with risk-to-reward
for e in range(episode_count + 1):
    print(f"Running episode {e}/{episode_count}")
    state = getState(data, 0, window_size)
    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    high_since_bought = low_since_bought = None
    risk_reward_ratio = 2  # Define desired risk-to-reward ratio

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size)
        current_price = data[t][3]  # Close price
        high = data[t][1]  # High price
        low = data[t][2]  # Low price
        reward = 0

        if action == 1 and len(agent.inventory) < agent.max_inventory:  # Buy if inventory is below limit
            agent.inventory.append(current_price)
            high_since_bought = high
            low_since_bought = low
            states_buy.append(t)
            print("Buy: " + formatPrice(current_price))

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            high_since_bought = max(high_since_bought, high)
            low_since_bought = min(low_since_bought, low)

            reward = calculate_reward(current_price, bought_price, high_since_bought, low_since_bought, risk_reward_ratio)
            total_profit += reward
            states_sell.append(t)
            s = str(datetime.datetime.now().time().strftime("%H:%M:%S"))
            print(f"{s} | Sell: {formatPrice(current_price)} | Profit: {formatPrice(reward)} / Total Profit: {formatPrice(total_profit)}")

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print(f"Final Profit: {formatPrice(total_profit)}")
            print("--------------------------------")
            plot_behavior([x[3] for x in data], states_buy, states_sell, total_profit)

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

        # check_save_model(agent, e)

    if e % 2 == 0:
        agent.save_model(f"model_ep{e}.h5")


# Save the final model
agent.save_model("final_model.h5")
