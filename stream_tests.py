import numpy as np
from river import linear_model, cluster, stream
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score
import metrics as osr_me
import data_prep
import plots


def find_threshold(clust_model, x_test_combined, y_test_combined, thresholds):
    fpr = []
    tpr = []
    for t in thresholds:
        fpr_t, tpr_t = get_roc_point(clust_model, x_test_combined, y_test_combined, t)
        fpr.append(fpr_t)
        tpr.append(tpr_t)

    auc = osr_me.auc_score(fpr, tpr)
    # Plot Roc curve
    # plots.plot_roc_curve(fpr, tpr, auc)
    threshold = osr_me.cutoff_youdens_j(fpr, tpr, thresholds)
    return threshold, auc


def get_roc_point(clust_model, x_test_combined, y_test_combined, e_threshold):
    # ---------- Testing phase ----------
    centers = data_prep.get_cluster_centers(clust_model)
    y_pred = []
    for test_sample in x_test_combined:
        s_dist = euclidean_distances(np.array(centers), np.array([test_sample]))
        s_dist = np.concatenate(s_dist, axis=0)

        i_dist = data_prep.calculate_inverse_distance(s_dist, 2)
        entropy = data_prep.calculate_entropy(i_dist)
        # print("Entropy: ", entropy)
        if entropy < e_threshold:
            y_pred.append(0)
        else:
            # Set unknown class label
            y_pred.append(1)

    # ROC Point
    fp_b, fn_b, tp_b, tn_b, c_matrix_b = osr_me.conf_matrix(y_test_combined, y_pred)

    fpr = fp_b[0]/(fp_b[0] + tn_b[0])
    tpr = tp_b[0]/(tp_b[0] + fn_b[0])

    return fpr, tpr


def run_clust_clas_tests(x_train, x_test, y_train, y_test, unknown_samples, unknown_labels, n_k_classes, n_u_samples,
                         e_threshold):
    # Transform the data into river format (dictionaries)
    train_dataset = stream.iter_array(X=x_train, y=y_train)
    # ---------- Training phase ----------
    k_means_model = cluster.STREAMKMeans(chunk_size=10, n_clusters=n_k_classes)
    lr_model = linear_model.SoftmaxRegression()
    x_preds = []
    for x, y in train_dataset:
        # Fit the clustering model
        x_preds.append(x)
        k_means_model.learn_one(x)
        # Fit classification model
        lr_model.learn_one(x, y)

    x_test_combined, y_test_combined = data_prep.add_class(x_test, y_test, unknown_samples, unknown_labels, n_u_samples)

    x_d_b_samples = []
    train_preds = []
    for x in x_preds:
        d_b_label = k_means_model.predict_one(x)
        x_d_b_samples.append(list(x.values()))
        train_preds.append(d_b_label)

    # ---------- Find threshold phase ----------
    d_b_index = 0.5
    try:
        d_b_index = davies_bouldin_score(x_d_b_samples, train_preds)
    except Exception as e:
        # Exception for when the openness is so high that the problem becomes one-class-classification
        # The index cannot be computed over 1 cluster
        print(e)

    y_test_combined_b = data_prep.group_known_classes(y_test_combined, unknown_labels)
    o_threshold, auc = find_threshold(k_means_model, x_test_combined, y_test_combined_b, e_threshold)

    # print("Threshold chosen: " + str(o_threshold))

    # ---------- Testing phase ----------
    centers = data_prep.get_cluster_centers(k_means_model)
    test_dataset = stream.iter_array(X=x_test_combined, y=y_test_combined)
    y_pred = []
    z = 0
    for x, y in test_dataset:
        s_dist = euclidean_distances(np.array(centers), np.array([x_test_combined[z]]))
        s_dist = np.concatenate(s_dist, axis=0)
        i_dist = data_prep.calculate_inverse_distance(s_dist, 2)
        entropy = data_prep.calculate_entropy(i_dist)

        # print("Entropia: ", entropy)
        if entropy < o_threshold:
            # Predict label with closed set classifier
            pred = lr_model.predict_one(x)
            lr_model.learn_one(x, y)
            k_means_model.learn_one(x)
            y_pred.append(pred)

        else:
            # Set unknown class label
            y_pred.append(min(unknown_labels))

        z = z + 1

    # Metrics
    # Accuracy and normalized accuracy
    # plots.plot_stream_cluster_results(k_means_model, x_test_combined, y_pred)
    acc = osr_me.accuracy(y_test_combined, y_pred)
    acc_k = osr_me.normalized_accuracy(y_test_combined, y_pred, 1, min(unknown_labels))
    acc_u = osr_me.normalized_accuracy(y_test_combined, y_pred, 0, min(unknown_labels))
    n_acc = osr_me.normalized_accuracy(y_test_combined, y_pred, 0.5, min(unknown_labels))

    # Open set f_measure (osr f1), precision and recall
    fp, fn, tp, tn, c_matrix = osr_me.conf_matrix(y_test_combined, y_pred)

    # Remove TP from the unknown class
    tp_u = tp.tolist()
    tp_u[min(unknown_labels)] = 0
    tp_u = np.array(tp_u)

    m_f1 = osr_me.osr_f1_score(tp_u, fp, fn)

    return acc, acc_k, acc_u, n_acc, m_f1, auc, d_b_index


def run_normal_clas_tests(x_train, x_test, y_train, y_test, unknown_samples, unknown_labels, n_u_samples):
    # Transform the data into river format (dictionaries)
    train_dataset = stream.iter_array(X=x_train, y=y_train)

    # ---------- Training phase ----------
    lr_model = linear_model.SoftmaxRegression()
    for x, y in train_dataset:
        lr_model.learn_one(x, y)

    x_test_combined, y_test_combined = data_prep.add_class(x_test, y_test, unknown_samples, unknown_labels, n_u_samples)

    # ---------- Testing phase ----------
    test_dataset = stream.iter_array(X=x_test_combined, y=y_test_combined)
    y_pred = []
    for x, y in test_dataset:
        pred = lr_model.predict_one(x)
        y_pred.append(pred)

    # Metrics
    # Accuracy and normalized accuracy
    # plots.plot_classifier_boundary(lr_model, x_train, x_test_combined, y_pred)
    acc = osr_me.accuracy(y_test_combined, y_pred)
    acc_k = osr_me.normalized_accuracy(y_test_combined, y_pred, 1, min(unknown_labels))
    acc_u = osr_me.normalized_accuracy(y_test_combined, y_pred, 0, min(unknown_labels))
    n_acc = osr_me.normalized_accuracy(y_test_combined, y_pred, 0.5, min(unknown_labels))

    # Open set f_measure (osr f1)
    fp, fn, tp, tn, c_matrix = osr_me.conf_matrix(y_test_combined, y_pred)
    # Remove TP from the unknown class
    tp_u = tp.tolist()
    tp_u[min(unknown_labels)] = 0
    tp_u = np.array(tp_u)

    m_f1 = osr_me.osr_f1_score(tp_u, fp, fn)

    return acc, acc_k, acc_u, n_acc, m_f1


def run_learning_clas_tests(x_train, x_test, y_train, y_test, unknown_samples, unknown_labels, n_u_samples):
    # Transform the data into river format (dictionaries)
    train_dataset = stream.iter_array(X=x_train, y=y_train)

    # ---------- Training phase ----------
    lr_model = linear_model.SoftmaxRegression()

    for x, y in train_dataset:
        lr_model.learn_one(x, y)

    x_test_combined, y_test_combined = data_prep.add_class(x_test, y_test, unknown_samples, unknown_labels, n_u_samples)

    # ---------- Testing phase ----------
    test_dataset = stream.iter_array(X=x_test_combined, y=y_test_combined)
    y_pred = []
    for x, y in test_dataset:
        pred = lr_model.predict_one(x)
        lr_model.learn_one(x, y)
        y_pred.append(pred)

    # Metrics
    # Accuracy and normalized accuracy
    # plots.plot_classifier_boundary(lr_model, x_train, x_test_combined, y_pred)
    acc = osr_me.accuracy(y_test_combined, y_pred)
    acc_k = osr_me.normalized_accuracy(y_test_combined, y_pred, 1, min(unknown_labels))
    acc_u = osr_me.normalized_accuracy(y_test_combined, y_pred, 0, min(unknown_labels))
    n_acc = osr_me.normalized_accuracy(y_test_combined, y_pred, 0.5, min(unknown_labels))

    # Open set f_measure (osr f1), precision and recall
    fp, fn, tp, tn, c_matrix = osr_me.conf_matrix(y_test_combined, y_pred)

    # Remove TP from the unknown class
    tp_u = tp.tolist()
    tp_u[min(unknown_labels)] = 0
    tp_u = np.array(tp_u)

    m_f1 = osr_me.osr_f1_score(tp_u, fp, fn)

    return acc, acc_k, acc_u, n_acc, m_f1