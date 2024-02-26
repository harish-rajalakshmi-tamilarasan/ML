import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
norm_layer_wide = keras.layers.Normalization()
norm_layer_deep = keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden_layer1 = keras.layers.Dense(30,activation="relu")(norm_deep)
hidden_layer2 = keras.layers.Dense(30,activation="relu")(hidden_layer1)
concate_layer1 = keras.layers.concatenate([norm_wide,hidden_layer2]) 
output_layer = keras.layers.Dense(1)(concate_layer1)
model = keras.Model(inputs=[input_wide,input_deep],outputs=[output_layer])


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep),y_train,epochs=20, validation_data=((X_valid_wide, X_valid_deep), y_valid))
mse_test, rmse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
print(y_pred)