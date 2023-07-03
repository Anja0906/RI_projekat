from keras.datasets import mnist
import keras
from keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

EPOCHS = 5
BATCH_SIZE = 64
BATCH_PER_EPOCH = 256
VALID_RATIO = 0.2
DROPOUT = 0 # 0.25


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainY = keras.utils.to_categorical(trainY)
    testY = keras.utils.to_categorical(testY)
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))  # Fix the reshape operation here
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    return trainX, trainY, testX, testY


def define_model():
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


def train():
    trainX, trainY, testX, testY = load_dataset()
    model = define_model()
    chkpt_lrn_callback = keras.callbacks.ModelCheckpoint(filepath="model_learn.hdf5", save_weights_only=False,
                                                         monitor="accuracy", mode="max",
                                                         save_best_only=True)
    chkpt_val_callback = keras.callbacks.ModelCheckpoint(filepath="model_valid.hdf5", save_weights_only=False,
                                                         monitor="val_accuracy", mode="max",
                                                         save_best_only=True)
    hist_log_callback = CSVLogger(filename="historydigits.csv", append=False)
    model.fit(trainX, trainY, validation_split=VALID_RATIO, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        steps_per_epoch=BATCH_PER_EPOCH, shuffle=True,
                        callbacks=[chkpt_lrn_callback, chkpt_val_callback, hist_log_callback], verbose=2)
    valid_model = keras.models.load_model("model_valid.hdf5")
    valid_model.evaluate(testX, testY, batch_size=BATCH_SIZE, verbose=2)
    predY = model.predict(testX, batch_size=BATCH_SIZE, verbose=2)
    testY = testY.argmax(axis=1)
    predY = predY.argmax(axis=1)
    cm = confusion_matrix(testY, predY)
    print(cm)


#train()
