import csv
import re
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import os
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn import preprocessing, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def read_data(file_path, id, train_labels, pos):
    matrix = []
    train_labels1 = train_labels[:pos]
    train_labels2 = train_labels[pos+1:]

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['x', 'y', 'z'])
        for row in csv_reader:
            linie = []

            linie.append(float(row['x']))
            linie.append(float(row['y']))
            linie.append(float(row['z']))

            matrix.append(linie)

    train_value = [train_labels[pos]] * len(matrix)
    train_labels = np.concatenate((train_labels1, train_value))
    train_labels = np.concatenate((train_labels, train_labels2))
    return train_labels, matrix


def read_data2(file_path):

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

def write_data(file_path, labels, ids, row_nr):

    with open(file_path, mode='w') as csv_file:
        fieldnames = ['id', 'class']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for ind, id in enumerate(ids):
            writer.writerow({'id': id, 'class': int(labels[ind])})


def normalize(trainX, testX, type=None):
    print(trainX.shape)
    print(testX.shape)

    Scalar = None

    if type == 'standard':
        Scalar = preprocessing.StandardScaler()
    elif type == 'min_max':
        Scalar = preprocessing.MinMaxScaler()
    elif type == 'l1' or type == 'l2':
        Scalar = preprocessing.Normalizer(norm=type)
    elif type == 'l2_v2':
        trainX = trainX / np.expand_dims(np.sqrt(np.sum(trainX ** 2, axis=1)), axis=1)
        testX = testX / np.expand_dims(np.sqrt(np.sum(testX ** 2, axis=1)), axis=1)
    elif type == 'robust':
        Scalar = preprocessing.RobustScaler()
    elif type == 'min-max':
        trainX = (trainX - np.min(trainX))/(np.max(trainX) - np.min(trainX))
        testX = (testX - np.min(trainX))/(np.max(testX) - np.min(testX))


    if Scalar is not None:
        trainX = Scalar.fit_transform(trainX)
        testX = Scalar.fit_transform(testX)

    return trainX, testX


def main():


    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/train")

    train_smartphones1 = []
    test_smartphones1 = []
    test_labels1 = []
    id, train_labels1 = read_labels('train_labels.csv')

    for root, dirs, files in os.walk(directory):
        files.sort()
        for pos, file in enumerate(files):
            path = root+"/"+file
            if file.endswith(".csv"):
                train_labels1, matrix = np.array(read_data(path, id, train_labels1, pos))
                if(len(train_smartphones1) == 0):
                    train_smartphones1 = np.empty((0, 3), float)
                train_smartphones1 = np.vstack((train_smartphones1, matrix))

    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/test")

    id_test = []
    row_nr = []

    for root, dirs, files in os.walk(directory):
        files.sort()
        for file in files:
            path = root+"/"+file
            if file.endswith(".csv"):
                id_test.append(int(file.split(".")[0]))
                row_nr.append(len(test_smartphones1))
                matrix = np.array(read_data2(path))
                if (len(test_smartphones1) == 0):
                    test_smartphones1 = np.empty((0, 3), float)
                test_smartphones1 = np.vstack((test_smartphones1, matrix))

    print(train_smartphones1.shape)
    print(train_labels1.shape)


    #train_smartphones, test_smartphones, train_labels, test_labels = train_test_split(train_smartphones1, train_labels1, test_size=0.3)



    # train_smartphones = train_smartphones1[0:7200]
    # test_smartphones = train_smartphones1[7200:]
    # train_labels = train_labels1[:7200]
    # test_labels = train_labels1[7200:]

    train_smartphones = np.array(train_smartphones1, dtype=np.float64)
    test_smartphones = np.array(test_smartphones1, dtype=np.float64)
    train_labels = np.array(train_labels1, dtype=np.int)
    #test_labels = np.array(test_labels, dtype=np.int)


    print('Train smartphones shape {}'.format(train_smartphones.shape))


    train_smartphones, test_smartphones = normalize(train_smartphones, test_smartphones, 'min_max')
    print(train_smartphones)

    alpha = np.random.randint(2, 10, 1)
    num_classes = 20

    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_Y = encoder.transform(train_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    train_labels = keras.utils.to_categorical(encoded_Y)

    # encoder = LabelEncoder()
    # encoder.fit(test_labels)
    # encoded_Y = encoder.transform(test_labels)
    # # convert integers to dummy variables (i.e. one hot encoded)
    # test_labels = keras.utils.to_categorical(encoded_Y)


    hidden_layers = int(train_smartphones.shape[0] / (alpha * (train_smartphones.shape[1] + num_classes)))

    print(hidden_layers)

    model = baseline_model(hidden_layers, num_classes, train_labels, train_smartphones)

   # score = model.evaluate(test_smartphones, test_labels, batch_size=128)

    predicted_labels1 = model.predict(test_smartphones, batch_size=None, verbose=0, steps=None)

   # print(score)
    first_index = 0
    last_index = row_nr[1]
    predicted_labels = []

    for indx in range(len(row_nr) - 1):
        values_to_sort = predicted_labels1[first_index:last_index]
        max_value = np.argmax(np.bincount(values_to_sort))
        predicted_labels.append(max_value)
        first_index = last_index
        last_index = row_nr[indx + 1]

    path = '/home/john/PycharmProjects/ML/Kaggle/submit.csv'

    write_data(path, predicted_labels, id_test)


def baseline_model(hidden_layers, num_classes, train_labels, train_smartphones):

    model = Sequential()
    model.add(Dense(hidden_layers, activation='relu', input_dim=3))
    model.add(Dropout(0.5))
    # model.add(Dense(hidden_layers, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(train_smartphones, train_labels, epochs=1, batch_size=14000)
    return model


main()