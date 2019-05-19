import csv
import re
from time import sleep

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import SGD
from skimage.viewer.widgets import history
from sklearn import preprocessing, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def read_data(file_path):
    matrix = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['x', 'y', 'z'])
        for row in csv_reader:
            linie = []

            linie.append(float(row['x']))
            linie.append(float(row['y']))
            linie.append(float(row['z']))

            matrix.append(linie)

    return matrix


def read_labels(file_path):

    id = []
    labels = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id.append(int(row['id']))
            labels.append(int(row['class']))

    return id, labels

def write_data(file_path, labels, ids):

    with open(file_path, mode='w') as csv_file:
        fieldnames = ['id', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for ind, id in enumerate(ids):
            writer.writerow({'id': id, 'class': int(labels[ind])})


def normalize(trainX, testX, minim, maxim):


    trainX = (trainX - minim) / (maxim - minim)
    testX = (testX - minim) / (maxim - minim)

    return trainX, testX


def evaluate(scores):
    m, s = np.mean(scores), np.std(scores)

    print('Accuracy: {} +/- ({})'.format(m, s))


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/train")

    minim = 20
    maxim = -20

    train_smartphones1 = []
    test_smartphones1 = []
    test_labels1 = []
    sizes = []
    id, train_labels1 = read_labels('train_labels.csv')

    for root, dirs, files in os.walk(directory):
        files.sort()
        for pos, file in enumerate(files):
            path = root+"/"+file
            if file.endswith(".csv"):
                matrix = np.array(read_data(path))
                sizes.append(matrix.shape[0])
                minim = min(minim, np.min(matrix))
                maxim = max(maxim, np.max(matrix))
                train_smartphones1.append(matrix)

    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/test")

    id_test = []
    test_sizes = []

    for root, dirs, files in os.walk(directory):
        files.sort()
        for file in files:
            path = root+"/"+file
            if file.endswith(".csv"):
                id_test.append(int(file.split(".")[0]))
                matrix = np.array(read_data(path))
                minim = min(minim, np.min(matrix))
                maxim = max(maxim, np.max(matrix))
                test_sizes.append(matrix.shape[0])
                test_smartphones1.append(matrix)

    mean = np.array(sizes).mean()
    for i in range(len(train_smartphones1)):
        train_smartphones1[i] = np.resize(train_smartphones1[i], (150, 3))

    for i in range(len(test_smartphones1)):
        test_smartphones1[i] = np.resize(test_smartphones1[i], (150, 3))


    train_smartphones = np.array(train_smartphones1, dtype=np.float32)
    test_smartphones = np.array(test_smartphones1, dtype=np.float32)
    train_labels = np.array(train_labels1, dtype=np.int)

    train_smartphones, test_smartphones = normalize(train_smartphones, test_smartphones, minim, maxim)

    alpha = np.random.randint(2, 10, 1)
    num_classes = 20

    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_Y = encoder.transform(train_labels)
    train_labels = keras.utils.to_categorical(encoded_Y)

    hidden_layers = int(train_smartphones.shape[0] / (alpha * (train_smartphones.shape[1] + num_classes)))

    # model = baseline_model2(hidden_layers, num_classes, train_labels, train_smartphones)

    model = load_model('best_model2.h5')

    predicted_labels = model.predict_classes(test_smartphones)

    predicted_labels = encoder.inverse_transform(predicted_labels)

    path = '/home/john/PycharmProjects/ML/Kaggle/submit_best_model4.csv'

    write_data(path, predicted_labels, id_test)


def baseline_model(hidden_layers, num_classes, train_labels, train_smartphones):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                     input_shape=(train_smartphones.shape[1], train_smartphones.shape[2])))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_smartphones, train_labels, epochs=45, verbose=1)
    return model

def baseline_model2(hidden_layers, num_classes, train_labels, train_smartphones):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu',
                     input_shape=(train_smartphones.shape[1], train_smartphones.shape[2])))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True)

    model.fit(train_smartphones, train_labels, epochs=400, verbose=1, validation_split=0.3,
              callbacks=[es, mc])

    return model

main()