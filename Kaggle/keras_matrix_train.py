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
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, \
    GlobalAveragePooling1D, BatchNormalization
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

def write_data(file_path, labels, ids, encoder):

    labels = encoder.inverse_transform(labels)

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

    minim = 20
    maxim = -20

    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/train")

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
                minim = min(minim, np.min(matrix))
                maxim = max(maxim, np.max(matrix))
                sizes.append(matrix.shape[0])
                # matrix = preprocessing.MinMaxScaler(copy=False).fit_transform(matrix)
                train_smartphones1.append(matrix)

    mean = np.array(sizes).mean()
    print(min(sizes))
    print(np.bincount(sizes))
    for i in range(len(train_smartphones1)):
        train_smartphones1[i] = np.resize(train_smartphones1[i], (150, 3))

    train_smartphones, test_smartphones, train_labels, test_labels = train_test_split(train_smartphones1, train_labels1, test_size=0.05)

    train_smartphones = np.array(train_smartphones, dtype=np.float32)
    test_smartphones = np.array(test_smartphones, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int)
    test_labels = np.array(test_labels, dtype=np.int)

    print('Train smartphones shape {}'.format(train_smartphones.shape))

    train_smartphones, test_smartphones = normalize(train_smartphones, test_smartphones, minim, maxim)

    alpha = np.random.randint(2, 10, 1)
    num_classes = 20

    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_Y = encoder.transform(train_labels)
    train_labels = keras.utils.to_categorical(encoded_Y)

    encoder = LabelEncoder()
    encoder.fit(test_labels)
    encoded_Y = encoder.transform(test_labels)
    test_labels = keras.utils.to_categorical(encoded_Y)

    print(train_labels.shape)

    hidden_layers = int(train_smartphones.shape[0] / (alpha * (train_smartphones.shape[1] + num_classes)))

    # model = baseline_model2(hidden_layers, num_classes, train_labels, train_smartphones)
    # _, score1 = model.evaluate(train_smartphones, train_labels)
    # _, score = model.evaluate(test_smartphones, test_labels)
    # score = score * 100.0
    # score1 = score1 * 100.0

    model2 = baseline_model(hidden_layers, num_classes, train_labels, train_smartphones, test_smartphones, test_labels)
    _, score2 = model2.evaluate(train_smartphones, train_labels)
    _, score3 = model2.evaluate(test_smartphones, test_labels)

    score2 = score2 * 100.0
    score3 = score3 * 100.0

    file = open('acc.txt', 'a')
    file.write('SG Test score ------> {}\n'.format(score3))
    file.write('SG Train score -----> {}\n'.format(score2))

    # file.write('ADAM Test score ------> {}\n'.format(score))
    # file.write('ADAM Train score -----> {}\n'.format(score1))
    #
    # os.system("shutdown /s /t 1")

    print('Test score ------> {}'.format(score3))
    print('Train score -----> {}'.format(score2))

    # print('Test score ------> {}'.format(score))
    # print('Train score -----> {}'.format(score1))

    # predicted_labels = model.predict_classes(test_smartphones)
    #
    # path = '/home/john/PycharmProjects/ML/Kaggle/submit2.csv'
    #
    # write_data(path, predicted_labels, id_test, encoder)


def baseline_model2(hidden_layers, num_classes, train_labels, train_smartphones):


    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_smartphones.shape[1], train_smartphones.shape[2])))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_smartphones, train_labels, epochs=60, verbose=1)
    return model


def baseline_model(hidden_layers, num_classes, train_labels, train_smartphones, test_X, test_Y):

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(train_smartphones.shape[1], train_smartphones.shape[2])))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    es  = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=32)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True)

    model.fit(train_smartphones, train_labels, epochs=400, verbose=1, validation_split=0.2, callbacks=[es, mc], shuffle=True)

    saved_model = load_model('best_model.h5')

    return saved_model

main()
