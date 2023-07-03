
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

MODELS_DIR = "D:/Fax/DS/DS/Model"
MODEL_TRAIN_PATH = MODELS_DIR + "model_learn.hdf5"
MODEL_VALID_PATH = MODELS_DIR + "model_valid.hdf5"
MODEL_TRAIN_MONITOR = "accuracy"
MODEL_VALID_MONITOR = "val_accuracy"
MODEL_MODE = "max"
HISTORY_PATH = MODELS_DIR + "history.csv"


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
	chkpt_lrn_callback = keras.callbacks.ModelCheckpoint(filepath = MODEL_TRAIN_PATH, save_weights_only = False, monitor = MODEL_TRAIN_MONITOR, mode = MODEL_MODE, save_best_only = True)
	chkpt_val_callback = keras.callbacks.ModelCheckpoint(filepath = MODEL_VALID_PATH, save_weights_only = False, monitor = MODEL_VALID_MONITOR, mode = MODEL_MODE, save_best_only = True)
	hist_log_callback = CSVLogger(filename = HISTORY_PATH, append = False)
	history = model.fit(trainX, trainY, validation_split = VALID_RATIO, epochs = EPOCHS, batch_size = BATCH_SIZE, steps_per_epoch = BATCH_PER_EPOCH, shuffle = True, callbacks = [chkpt_lrn_callback, chkpt_val_callback, hist_log_callback], verbose = 2)

	valid_model = keras.models.load_model(MODEL_VALID_PATH)
	loss = valid_model.evaluate(testX, testY, batch_size = BATCH_SIZE, verbose = 2)
	predY = model.predict(testX, batch_size = BATCH_SIZE, verbose = 2)

	testY = testY.argmax(axis = 1)
	predY = predY.argmax(axis = 1)

	cm = confusion_matrix(testY, predY)


	print(cm)


if __name__ == "__main__":
	train()
