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

class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu",**kwargs):
        super().__init__(**kwargs)
        self.norm_wide = keras.layers.Normalization()
        self.norm_deep = keras.layers.Normalization()
        self.hidden1 = keras.layers.Dense(30,activation="relu")
        self.hidden2 = keras.layers.Dense(30,activation="relu")
        self.concate = keras.layers.Concatenate()
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide_layer = self.norm_wide(input_wide)
        norm_deep_layer = self.norm_deep(input_deep)
        hidden1_layer = self.hidden1(norm_deep_layer)
        hidden2_layer = self.hidden2(hidden1_layer)
        concat = self.concate([norm_wide_layer,hidden2_layer])
        output_layer = self.main_output(concat)
        aux_layer = self.aux_output(hidden2_layer)
        return output_layer, aux_layer
    
model = WideAndDeepModel(30,activation="relu",name="my_cool_model")

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=["mse","mse"],optimizer=optimizer,loss_weights=(0.9,0.1),metrics = ["RootMeanSquaredError"])
model.norm_wide.adapt(X_train_wide)
model.norm_deep.adapt(X_train_deep)
model.call((X_train_wide,X_train_deep))
history = model.fit((X_train_wide,X_train_deep),(y_train,y_train),epochs=20,validation_data=((X_valid_wide,X_valid_deep),(y_valid,y_valid)))
eval_results = model.evaluate((X_test_wide,X_test_deep),(y_test,y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
print(y_pred_main)
print(y_pred_aux)
model.save("my_keras_model",save_format="tf")