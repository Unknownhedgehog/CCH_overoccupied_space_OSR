import sklearn.datasets as skd
import numpy as np
from river import datasets
import pandas as pd
from ucimlrepo import fetch_ucirepo


def create_dataset(dataset_type, n_samples, n_features, n_classes, std, class_sep, rand_state):
    if dataset_type == 'blobs':
        x, y = skd.make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=std,
                              random_state=rand_state)
    elif dataset_type == 'classification':
        x, y = skd.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features,
                                       n_redundant=0, n_clusters_per_class=1, class_sep=class_sep, n_classes=n_classes,
                                       random_state=rand_state)
    return x, y


def create_blobs_dataset(n_samples, n_features, n_classes, std, rand_state):
    x, y = skd.make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=std,
                          random_state=rand_state)

    return x, y


def create_class_dataset(n_samples, n_features, n_classes, class_sep, rand_state):
    x, y = skd.make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features,
                                   n_redundant=0, n_clusters_per_class=1, class_sep=class_sep, n_classes=n_classes,
                                   random_state=rand_state)
    return x, y


def create_insect_dataset(n_samples):
    X = []
    Y = []
    known_labels = []
    d = datasets.Insects().take(k=n_samples)
    for x, y in d:
        X.append(list(x.values()))
        if y in known_labels:
            label = known_labels.index(y)
        else:
            known_labels.append(y)
            label = len(known_labels) - 1
        Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def create_insect_dataset2(n_samples): # Insects dataset method provided by River currently not working
    file_path = 'datasets/INSECTS_abrupt_balanced.csv'
    data = pd.read_csv(file_path, nrows=n_samples, header=None)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    X = np.array(X)
    Y = np.array(Y)
    Y2 = []
    known_labels = []
    for i in Y:
        if i in known_labels:
            label = known_labels.index(i)
        else:
            known_labels.append(i)
            label = len(known_labels) - 1
        Y2.append(label)

    Y2 = np.array(Y2)

    return X, Y2


def create_covert_dataset(n_samples):
    file_path = 'datasets/covtype.data'
    data = pd.read_csv(file_path, nrows=581012, header=None)
    label_counts = data[54].value_counts()

    subset = data.sample(n=n_samples, random_state=42)
    label_counts = subset[54].value_counts()

    X = subset.iloc[:, :-1]
    Y = subset.iloc[:, -1]

    X = np.array(X)
    Y = np.array(Y)
    Y2 = []
    known_labels = []
    for i in Y:
        if i in known_labels:
            label = known_labels.index(i)
        else:
            known_labels.append(i)
            label = len(known_labels) - 1
        Y2.append(label)

    return X, Y