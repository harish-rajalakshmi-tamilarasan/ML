import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from mlp_hyperparam_tuning import build_model as build_model

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)
    
    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalize"):
            norm_layer = keras.layers.Normalization()
            X=norm_layer(X)
        return model.fit(X,y,**kwargs)

hyperband_tuner = kt.Hyperband(MyClassificationHyperModel(),objective="val_accuracy",seed=42,max_epochs=10, factor=3,hyperband_iterations=2,overwrite=True,directory="my_fashion_mnist",project_name="hyperband")
