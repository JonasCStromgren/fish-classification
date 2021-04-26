import itertools
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import matplotlib.pylab as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

#PARAMETERS
pixels = 456
IMAGE_SIZE = (pixels, pixels)

EPOCHS = 10
BATCH_SIZE = 8


# DATA SETUP
train_path = 'train'
valid_path = 'valid'
test_path = 'test'
fish_classes = os.listdir(train_path)

train_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=IMAGE_SIZE, classes=fish_classes, batch_size=BATCH_SIZE,class_mode="categorical")
valid_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=IMAGE_SIZE, classes=fish_classes, batch_size=BATCH_SIZE,class_mode="categorical")
test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=IMAGE_SIZE, classes=fish_classes, batch_size=BATCH_SIZE, class_mode="categorical")


# MODEL SETUP
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(pixels,pixels,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=9, activation='softmax')
])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# MODEL TRAINING
hist = model.fit(x=train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS,
    verbose=2
).history

# saving the model for easy testing later
model.save('CNN_fish_model')

# PLOT RESULTS
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.ylim([0,2])
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.legend(['Training','Validation'])
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.ylim([0,1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])
plt.legend(['Training','Validation'])
plt.show()
