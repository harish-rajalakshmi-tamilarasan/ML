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

normalization_layer = keras.layers.Normalization()
hidden_layer1 = keras.layers.Dense(30,activation="relu") 
hidden_layer2 = keras.layers.Dense(30,activation="relu") 
concate_layer1 = keras.layers.Concatenate() 
output_layer = keras.layers.Dense(1)

input_ = keras.layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concate = concate_layer1([normalized,hidden2])
output = output_layer(concate)
model = keras.Model(inputs=[input_],outputs=[output])

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse",
optimizer=optimizer,
metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train,y_train,epochs=20, validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)