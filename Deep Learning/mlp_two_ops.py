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

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
norm_wide = keras.layers.Normalization()
norm_deep = keras.layers.Normalization()
norm_wide_layer = norm_wide(input_wide)
norm_deep_layer = norm_deep(input_deep)
hidden_layer1 = keras.layers.Dense(30,activation="relu")(norm_deep_layer)
hidden_layer2 = keras.layers.Dense(30,activation="relu")(hidden_layer1)
concate_layer = keras.layers.concatenate([input_wide,hidden_layer2])
output_layer = keras.layers.Dense(1)(concate_layer)
aux_layer = keras.layers.Dense(1)(hidden_layer2)
model = keras.Model(inputs=[input_wide,input_deep],outputs=[output_layer,aux_layer])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=["mse","mse"],optimizer=optimizer,loss_weights=(0.9,0.1),metrics = ["RootMeanSquaredError"])
norm_wide.adapt(X_train_wide)
norm_deep.adapt(X_train_deep)
history = model.fit((X_train_wide,X_train_deep),(y_train,y_train),epochs=20,validation_data=((X_valid_wide,X_valid_deep),(y_valid,y_valid)))
eval_results = model.evaluate((X_test_wide,X_test_deep),(y_test,y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
print(y_pred_main)
print(y_pred_aux)
