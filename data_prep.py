import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import random
from random import sample, randint


def scale_feat(x_train, x_test, x_unknown):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    unknown_samples = []
    for s in x_unknown:
        t = s[0].reshape(1, -1)
        unknown_samples.append((scaler.transform(t)[0], s[1]))

    return x_train_scaled, x_test_scaled, unknown_samples


def remove_class(x, y, class_labels):

    unknown_class_samples = []
    unknown_class_samples_indexes = []
    counter = 0
    for sample2, label in zip(x, y):
        if label in class_labels:
            unknown_class_samples.append((sample2, label))
            unknown_class_samples_indexes.append(counter)

        counter += 1

    # Data without samples from unknown classes
    x_without_unknown_classes = np.delete(x, unknown_class_samples_indexes, 0)
    y_without_unknown_classes = np.delete(y, unknown_class_samples_indexes)

    return x_without_unknown_classes, y_without_unknown_classes, unknown_class_samples


def add_class(x_test, y_test, unknown_class_samples, unknown_class_labels, number_unknown_examples=[50]):
    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.tolist()

    unknown_samples_list = []
    for s in unknown_class_samples:
        unknown_samples_list.append(list(s[0]))

    if number_unknown_examples[0] > len(unknown_samples_list):
        number_unknown_examples[0] = len(unknown_samples_list) - 1

    try:
        unknown_class_samples = sample(unknown_samples_list, number_unknown_examples[0])
    except ValueError:
        print(f"Requested number of samples ({number_unknown_examples[0]}) is too large for population size ({len(unknown_samples_list)}).")
        unknown_class_samples = random.sample(unknown_samples_list, len(unknown_samples_list))  # Sample the entire population
        print("Sampled entire population:", unknown_class_samples)

    x_test = x_test.tolist()
    y_test = y_test.tolist()
    # Add samples to x_test
    for u in unknown_class_samples:
        index = randint(0, len(x_test))
        x_test.insert(index, u)
        y_test.insert(index, min(unknown_class_labels))

    return x_test, y_test


def calculate_inverse_distance(x_dist, power):
    """
    Calculates the probability of each point based on the distance to each cluster centroid.
    Parameters:
        x_dist - An array of distances of each point to the cluster centers
        power - An int. The higher, the more skewed the class probability distribution will be.
    """
    i_dist = 1.0 / x_dist ** power
    i_dist = i_dist / i_dist.sum(axis=0, keepdims=True)
    return i_dist


def calculate_entropy(i_dist):
    """
    Calculates the Shannon entropy of an array of probabilities.
    Parameters:
        i_dist - A array of probabilities.
    """
    e = entropy(i_dist)
    return e


def get_cluster_centers(s_clust_model):
    d_centers = s_clust_model.centers
    centers = []
    for k, v in d_centers.items():
        center = []
        for c in v.values():
            center.append(c)

        centers.append(center)

    return centers


def group_known_classes(y, unknown_labels):
    # 0 for known classes, 1 for unknown classes
    y_group = []
    for i in y:
        if i in unknown_labels:
            y_group.append(1)
        else:
            y_group.append(0)

    return y_group


def generate_unknown(openness, classes, prop_unknown_samples=0.1, n_samples=[], unknown_labels='random'):
    """ Generates unknows for a degree of missing classes and given set of classes. Number of unknown samples can be set
     manually or 'random' for random number of samples for each class"""
    if unknown_labels == 'random':
        unknown_class_labels = []
        n_u_samples = []
        for e in range(len(classes)):
            n_u_classes = round(classes[e] * openness)
            if n_u_classes == classes[e]:  # All classes cannot be unknown
                n_u_classes = classes[e] - 2
            if n_u_classes == classes[e] - 1:  # All classes unknown but one means a one-class classification problem
                n_u_classes = classes[e] - 2
            if n_u_classes == 0:  # There is always at least 1 unknown class
                n_u_classes = 1
            n_u_s = random.sample(range(classes[e]), n_u_classes)
            n_u_s.sort()

            if prop_unknown_samples == 'random':
                n_s = random.sample(range(50, round(n_samples[e] / classes[e]) - 10), n_u_classes)
            else:
                # Calculate number of unknown examples
                samples = round(n_samples[e] / classes[e])
                samples = round(samples * prop_unknown_samples)
                n_s = []
                for c in range(n_u_classes):
                    n = samples
                    n_s.append(n)

            unknown_class_labels.append(n_u_s)
            n_u_samples.append(n_s)
    else:
        unknown_class_labels = unknown_labels
        n_u_samples = []
        for e in range(len(unknown_class_labels)):
            n_u_classes = len(unknown_class_labels[e])
            if prop_unknown_samples == 'random':
                n_s = random.sample(range(50, round(n_samples[e] / classes[e]) - 10), n_u_classes)
            else:
                samples = round(n_samples[e] / classes[e])
                samples = round(samples * prop_unknown_samples)
                n_s = []
                for c in range(n_u_classes):
                    n = samples
                    n_s.append(n)

            n_u_samples.append(n_s)

    print(unknown_class_labels)
    print(n_u_samples)
    return unknown_class_labels, n_u_samples
