import csv
import re
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import preprocessing


def read_data(file_path):
    matrix = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=['x', 'y', 'z'])
        for row in csv_reader:
            linie = []

            linie.append(row['x'])
            linie.append(row['y'])
            linie.append(row['z'])

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

    if Scalar is not None:
        Scalar.fit(
            trainX)  # pt scaler sau min max si calc mean and std si pt min max calc max si min, pt normalizer nu face nimic
        trainX = Scalar.transform(trainX)
        testX = Scalar.transform(testX)

    return (trainX, testX)


class Knn_classifier:
    def __init__(self, train_smartphone, train_labels, ids):
        self.ids = ids
        self.train_smartphone = train_smartphone
        self.train_labels = train_labels

    def classify_smartphone(self, test_smartphone, num_neighbors=3, metric='l2', id=[], pos=0):

        try:
            print(id[pos])
        except:
            print('not ok')

        if metric == 'l1':
            distances = np.sum(abs(self.train_smartphone - test_smartphone), axis=1)
        else:
            distances = np.sqrt(np.sum((self.train_smartphone - test_smartphone) ** 2, axis=1))

        sorted_indexes = np.argsort(distances)

        first_n_indexes = sorted_indexes[:num_neighbors]

        first_n_labels = self.train_labels[first_n_indexes]
        freq = np.bincount(first_n_labels)

        return np.argmax(freq)

    def classify_all_smartphones(self, test_smartphones, num_neighbours=3, metric='l2'):
        predicted_labels = np.zeros(len(test_smartphones))

        for idx, test_smartphone in enumerate(test_smartphones):
            predicted_labels[idx] = self.classify_smartphone(test_smartphone, num_neighbours, metric, self.ids, idx)

        return predicted_labels


def main():
    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/train")

    train_smartphones1 = []
    test_smartphones1 = []
    test_labels1 = []
    id, train_labels1 = read_labels('train_labels.csv')

    for root, dirs, files in os.walk(directory):
        files.sort()
        for file in files:
            path = root + "/" + file
            if file.endswith(".csv"):
                matrix = np.array(read_data(path))
                matrix = np.resize(matrix, 150 * 3)
                train_smartphones1.append(matrix)

    directory = os.path.join("/home/john/PycharmProjects/ML/Kaggle/test")

    id_test = []

    for root, dirs, files in os.walk(directory):
        files.sort()
        for file in files:
            path = root + "/" + file
            if file.endswith(".csv"):
                matrix = np.array(read_data(path))
                matrix = np.resize(matrix, 150 * 3)
                id_test.append(int(file.split(".")[0]))
                test_smartphones1.append(matrix)

    train_smartphones = train_smartphones1
    test_smartphones = test_smartphones1
    train_labels = train_labels1

    train_smartphones = np.array(train_smartphones, dtype=np.float)
    test_smartphones = np.array(test_smartphones, dtype=np.float)
    train_labels = np.array(train_labels, dtype=np.int)
    # test_labels = np.array(test_labels, dtype=np.int)

    print('Train smartphones shape {}'.format(train_smartphones.shape))

    train_smartphones, test_smartphones = normalize(train_smartphones, test_smartphones, 'l2')

    Knn_c = Knn_classifier(train_smartphones, train_labels, id)

    num_neighbours = [1]
    # accuracy = np.zeros(len(num_neighbours))
    # for idx, num_neighbour in enumerate(num_neighbours):
    #     predicted_labels = Knn_c.classify_all_smartphones(test_smartphones, num_neighbour, 'l2')
    #     accuracy[idx] = computed_accuracy(test_labels, predicted_labels)

    # accuracy1 = np.zeros(len(num_neighbours))
    for idx, num_neighbour in enumerate(num_neighbours):
        predicted_labels = Knn_c.classify_all_smartphones(test_smartphones, num_neighbour, 'l1')
    #   accuracy1[idx] = computed_accuracy(test_labels, predicted_labels)

    path = '/home/john/PycharmProjects/ML/Kaggle/submit.csv'

    write_data(path, predicted_labels, id_test)

    # plt.plot(num_neighbours, accuracy, 'b', label="l2")
    # plt.plot(num_neighbours, accuracy1, 'r', label="l1")
    # plt.legend()
    # plt.show()
    #
    # print(confussion_matrix(test_labels, predicted_labels))


def computed_accuracy(gt_labels, predicted_labels):
    return (gt_labels == predicted_labels).mean()


def confussion_matrix(gt_labels, predicted_labels):
    num_classes = max(gt_labels) + 1
    num_samples = len(gt_labels)
    conf_mat = np.zeros((num_classes, num_classes))

    for idx in range(num_samples):
        predicted_label = int(predicted_labels[idx])
        gt_label = gt_labels[idx]

        if predicted_label == gt_label:
            conf_mat[gt_label][predicted_label] += 1
        else:
            conf_mat[gt_label][predicted_label] += 1

    return conf_mat


main()