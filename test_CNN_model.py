import itertools
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import matplotlib.pylab as plt
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from functions import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


#PARAMETERS
pixels = 456
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 8
test_path = 'test'

#DATA SETUP
fish_classes = os.listdir(test_path)
test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=IMAGE_SIZE, classes=fish_classes, batch_size=BATCH_SIZE, class_mode="categorical")

#LOAD MODEL
model = keras.models.load_model("CNN_fish_model")

#PREDICT ON TEST DATA
test_imgs, test_labels = next(test_generator)
predictions = model.predict(x=test_generator, steps=len(test_generator), verbose=0)

#PLOT RESULTS
cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))
plot_confusion_matrix(cm=cm, classes=fish_classes, title='Confusion Matrix')
plt.show()
