import keras
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import array_to_img
from Utils import *
from sklearn.model_selection import train_test_split

input_dir = "../Data/dataset-iiit-pet-master/images/"
target_dir = "../Data/dataset-iiit-pet-master/annotations/trimaps/"

input_img_paths, target_paths = get_data_path_list(input_dir, target_dir)

img_size = (200, 200)
num_img = len(input_img_paths)

input_img, targets = stack_data(input_img_paths, target_paths, img_size, num_img)

train_input_img, val_input_img, train_targets, val_targets = train_test_split(input_img, targets, test_size=0.3,
                                                                              random_state=101)
num_classes = 3
inputs = keras.Input(shape=img_size + (3,))
x = layers.Rescaling(1./255)(inputs)

x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer='l2')(x)
x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_regularizer='l2')(x)

x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", kernel_regularizer='l2')(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2, kernel_regularizer='l2')(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", kernel_regularizer='l2')(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

history = model.fit(train_input_img, train_targets, epochs=100, batch_size=128,
                    validation_data=(val_input_img, val_targets))

pred = model.predict(val_input_img, verbose=1)
masks = create_mask(pred)


plt.figure(figsize=(18, 10))
for i in range(1, 11):
    test_image = val_input_img[i]
    plt.subplot(5, 6, 3 * i - 2)
    plt.axis("off")
    plt.imshow(array_to_img(test_image))

    mask = masks[i]
    plt.subplot(5, 6, 3 * i - 1)
    plt.axis("off")
    plt.imshow(mask)

    plt.subplot(5, 6, 3 * i)
    plt.axis("off")
    plt.imshow(val_targets[i])
plt.tight_layout()

plt.figure(figsize=(18, 10))
for i in range(1, 21):
    p = pred[i*20]
    image = val_input_img[i*20]
    img = stack_mask(p, image)
    plt.subplot(5, 8, 2 * i - 1)
    plt.axis("off")
    plt.imshow(masks[i*20])
    plt.subplot(5, 8, 2 * i)
    plt.axis("off")
    plt.imshow(img.astype(np.uint8))

plt.tight_layout()
plt.show()
