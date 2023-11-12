import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import KFold

import os
import numpy as np
from numpy import mean
from numpy import std
from PIL import Image
import matplotlib.pyplot as plt

# load and scale training and test data
def load_dataset():
    # split the data of training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10         # 10 digits

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # convert to one-hot
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    return x_train, y_train, x_test, y_test

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9) # Stochastic gradient descent
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    plt.clf()
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()
    plt.savefig("summarize_diagnostics.png")
 
# summarize model performance
def summarize_performance(scores):
    plt.clf()
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()
    plt.savefig("summarize_performance.png")
 
# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    print("loaded dataset")
    # evaluate model using 5-fold Cross-Validation
    scores, histories = evaluate_model(trainX, trainY)
    print("evaluated models")
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)
    print("summaries completed")

def run_final():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    print("loaded dataset")

    if os.path.isfile("./final_model.h5"):
        model = keras.models.load_model("final_model.h5")
        print("model already exists")
    else:
        model = define_model()
        model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
        model.save('final_model.h5')
        print("saved final model")
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('final model accuracy %.3f' % (acc * 100.0))
 
# entry point, run the test harness
#run_test_harness()
run_final()