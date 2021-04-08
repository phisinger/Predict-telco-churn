# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Imports

# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

import pickle

print("Here is Tensorflow", tf.__version__)


# %%
## Load data

pickle_in = open("..\data\X_train.p", "rb")
X_train = pd.DataFrame(pickle.load(pickle_in))
pickle_in.close()

pickle_in = open("..\data\y_train.p", "rb")
y_train = pd.DataFrame(pickle.load(pickle_in))
pickle_in.close()

pickle_in = open("..\data\X_test.p", "rb")
X_test = pd.DataFrame(pickle.load(pickle_in))
pickle_in.close()

pickle_in = open("..\data\y_test.p", "rb")
y_test = pd.DataFrame(pickle.load(pickle_in))
pickle_in.close()

print("data loaded")


# %%
## create validation set
# kernel failed by execution
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=123)


# %%
# X_val = X_train[(len(X_train)//5):]
# X_train = X_train[:(len(X_train)//5)]

# y_val = y_train[(len(y_train)//5):]
# y_train = y_train[:(len(y_train)//5)]


# %%
print(type(X_train))


# %%
# Input Layer: 20 neurons
# hidden Layers: 1 to 5 => 2
# Outplut Layer: 1 Neuron
# Output Activation function: logistic
# Loss function cross entropy

# %% [markdown]
# # Building the model

# %%
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[20, 1]))
model.add(keras.layers.Dense(13, activation="relu"))
model.add(keras.layers.Dense(7, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))


# %%
model.summary()


# %%
model.compile(loss="binary_crossentropy",
    optimizer="adam", 
    metrics=["precision"])

print("model compiled")


# %%
# training_history = model.fit()


