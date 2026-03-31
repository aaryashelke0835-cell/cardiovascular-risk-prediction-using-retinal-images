import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

# Settings
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 5

DATA_DIR = "test_images"

# Data generator
datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# Base model
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False

# Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_data,
          validation_data=val_data,
          epochs=EPOCHS)

# Save model
model.save("retrained_graph.keras")


# Save labels
with open("retrained_labels.txt", "w") as f:
    for label in train_data.class_indices:
        f.write(label + "\n")

print("\nModel training complete.")
print("retrained_graph.pb created successfully!")
