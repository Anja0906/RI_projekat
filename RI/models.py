import numpy as np
import keras
from keras.callbacks import CSVLogger
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

EPOCHS = 50
BATCH_SIZE = 64
BATCH_PER_EPOCH = 256
VALID_RATIO = 0.2
DROPOUT = 0  # 0.25

MODELS_DIR = "RI2023/Resources/Models/"
MODEL_TRAIN_PATH = MODELS_DIR + "model_learn.hdf5"
MODEL_VALID_PATH = MODELS_DIR + "model_valid.hdf5"
MODEL_TRAIN_MONITOR = "accuracy"
MODEL_VALID_MONITOR = "val_accuracy"
MODEL_MODE = "max"
HISTORY_PATH = MODELS_DIR + "history.csv"


def define_model_boxing():
    inputs = Input(shape=(28, 28, 1))

    model = Sequential()
    model.add(inputs)
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def define_model_recognition():
    inputs = Input(shape=(28, 28, 1))

    model = Sequential()
    model.add(inputs)
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)

    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))  # Fix the reshape operation here
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    return trainX, trainY, testX, testY
