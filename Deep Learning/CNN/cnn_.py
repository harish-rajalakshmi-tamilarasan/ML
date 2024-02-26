from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt


def print_image(images):
    for image in images:
        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

def display_feature_maps(feature_maps, num_columns=6):
    num_feature_maps = feature_maps.shape[-1]
    num_rows = (num_feature_maps + num_columns - 1) // num_columns

    plt.figure(figsize=(num_columns * 2, num_rows * 2))

    for i in range(num_feature_maps):
        plt.subplot(num_rows, num_columns, i + 1)
        fmap = feature_maps[0, :, :, i]
        fmap_normalized = (fmap - tf.reduce_min(fmap)) / (tf.reduce_max(fmap) - tf.reduce_min(fmap))
        plt.imshow(fmap_normalized, cmap='gray')
        plt.axis('off')

    plt.show()

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1/255)(images)
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7,padding = "same")
fmaps = conv_layer(images)

print(conv_layer.get_weights())

