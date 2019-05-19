import csv
import re
from time import sleep
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import matplotlib.pyplot as plt
import os

from sklearn import preprocessing, linear_model, metrics
from sklearn.model_selection import train_test_split


def read_data(file_path, id, train_labels, pos):
    matrix = []
    train_labels1 = train_labels[:pos]
    train_labels2 = train_labels[pos+1:]

    print('Reading ---> {}'.format(pos))

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
            id.append(row['id'])
            labels.append(row['class'])

    return id, labels

def write_data(file_path, labels, ids):


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
                matrix = preprocessing.MinMaxScaler(copy=False).fit_transform(matrix)
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
                matrix = preprocessing.MinMaxScaler(copy=False).fit_transform(matrix)
                if (len(test_smartphones1) == 0):
                    test_smartphones1 = np.empty((0, 3), float)
                test_smartphones1 = np.vstack((test_smartphones1, matrix))

    # train_smartphones, test_smartphones, train_labels, test_labels = train_test_split(train_smartphones1, train_labels1, test_size=0.3)

    # train_smartphones = train_smartphones1[0:7200]
    # test_smartphones = train_smartphones1[7200:]
    # train_labels = train_labels1[:7200]
    # test_labels = train_labels1[7200:]

    train_smartphones = np.array(train_smartphones1, dtype=np.float64)
    test_smartphones = np.array(test_smartphones1, dtype=np.float64)
    train_labels = np.array(train_labels1, dtype=np.int)
    # test_labels = np.array(test_labels, dtype=np.int)


    print('Train smartphones shape {}'.format(train_smartphones.shape))
    print(id[0])

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mul_lr.fit(train_smartphones, train_labels)

    predicted_labels1 = mul_lr.predict(test_smartphones)
    # print(metrics.accuracy_score(test_labels, predicted_labels))

    first_index = 0
    last_index = row_nr[1]
    predicted_labels = []

    print(predicted_labels1)

    for indx in range(len(row_nr) - 1):
        values_to_sort = predicted_labels1[first_index:last_index]
        print('{} with shape {}'.format(values_to_sort, len(values_to_sort)))
        max_value = np.argmax(np.bincount(values_to_sort))
        predicted_labels.append(max_value)
        first_index = last_index
        last_index = row_nr[indx + 1]

    values_to_sort = predicted_labels1[last_index:]
    max_value = np.argmax(np.bincount(values_to_sort))
    predicted_labels.append(max_value)

    path = '/home/john/PycharmProjects/ML/Kaggle/submit1.csv'

    write_data(path, predicted_labels, id_test)


main()