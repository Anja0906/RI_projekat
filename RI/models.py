import cv2
import numpy as np
import keras
from PIL import Image
from keras.callbacks import CSVLogger
from keras.datasets import mnist
from keras.saving.saving_api import load_model
from keras.utils import img_to_array
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


def main():
    trainX, trainY, testX, testY = load_dataset()

    # Model Boxing
    model_boxing = define_model_boxing()
    chkpt_lrn_callback = keras.callbacks.ModelCheckpoint(filepath=MODEL_TRAIN_PATH, save_weights_only=False,
                                                         monitor=MODEL_TRAIN_MONITOR, mode=MODEL_MODE,
                                                         save_best_only=True)
    hist_log_callback = CSVLogger(filename=HISTORY_PATH, append=False)
    history_boxing = model_boxing.fit(trainX, trainY, validation_split=VALID_RATIO, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, steps_per_epoch=BATCH_PER_EPOCH, shuffle=True,
                                      callbacks=[chkpt_lrn_callback, hist_log_callback], verbose=2)
    model_boxing.save(MODEL_TRAIN_PATH)

    # Model Recognition
    model_recognition = define_model_recognition()
    chkpt_val_callback = keras.callbacks.ModelCheckpoint(filepath=MODEL_VALID_PATH, save_weights_only=False,
                                                         monitor=MODEL_VALID_MONITOR, mode=MODEL_MODE,
                                                         save_best_only=True)
    history_recognition = model_recognition.fit(trainX, trainY, validation_split=VALID_RATIO, epochs=EPOCHS,
                                                batch_size=BATCH_SIZE, steps_per_epoch=BATCH_PER_EPOCH,
                                                shuffle=True, callbacks=[chkpt_val_callback, hist_log_callback],
                                                verbose=2)
    model_recognition.save(MODEL_VALID_PATH)

    # Load Valid Model
    valid_model_boxing = keras.models.load_model(MODEL_TRAIN_PATH)
    valid_model_recognition = keras.models.load_model(MODEL_VALID_PATH)

    # Evaluate the models
    loss_boxing = valid_model_boxing.evaluate(testX, testY, batch_size = BATCH_SIZE, verbose = 2)
    loss_recognition = valid_model_recognition.evaluate(testX, testY, batch_size = BATCH_SIZE, verbose = 2)

    # Predict
    predY_boxing = valid_model_boxing.predict(testX, batch_size=BATCH_SIZE, verbose=2)
    predY_recognition = valid_model_recognition.predict(testX, batch_size=BATCH_SIZE, verbose=2)

    testY = testY.argmax(axis=1)
    predY_boxing = predY_boxing.argmax(axis=1)
    predY_recognition = predY_recognition.argmax(axis=1)

    # Confusion Matrix
    cm_boxing = confusion_matrix(testY, predY_boxing)
    cm_recognition = confusion_matrix(testY, predY_recognition)

    print("Confusion matrix for boxing model:")
    print(cm_boxing)
    print("Confusion matrix for recognition model:")
    print(cm_recognition)


def load_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = np.reshape(resized, (28, 28, 1))

    return reshaped


def predict_image(image_path):
    model_boxing = define_model_boxing()
    model_recognition = define_model_recognition()
    original_image = load_image(image_path)
    bounding_boxes = model_boxing.predict(np.array([original_image]))

    predictions = []
    for box in bounding_boxes:
        cropped_image = original_image[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
        cropped_image = cropped_image / 255.0
        print(cropped_image.shape)
        result_recognition = model_recognition.predict(np.array([cropped_image]))
        label_recognition = np.argmax(result_recognition, axis=1)
        predictions.append(label_recognition)

    return predictions

