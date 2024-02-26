import tensorflow as tf
from tensorflow import keras
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full = X_train_full / 255.0

X_valid = X_train_full[:5000]
X_train = X_train_full[5000:]
y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, padding = "same", activation = "relu", kernel_initializer = "he_normal")

model = keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),
    keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64,activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10,activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
history = model.fit(X_train,y_train,epochs=30, validation_data=(X_valid, y_valid))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy*100:.2f}%, Test loss: {test_loss:.4f}")
print(model.summary())


# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

predictions = model.predict(X_test[:10])

plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap='binary')
    plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
plt.show()



