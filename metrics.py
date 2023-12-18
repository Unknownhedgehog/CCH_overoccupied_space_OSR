import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from scipy.stats import ttest_ind
from matplotlib import pyplot


def accuracy(y_test, y_pred):
    """Calculate the accuracy score"""
    acc = accuracy_score(y_test, y_pred)
    return acc


def normalized_accuracy(y_test, y_pred, reg_c, u_label):
    """Calculate the normalized accuracy score from the K-Acc and the U-Acc.
    reg_c: Balancing constraint between K-Acc and U-Acc. 1 is for K-Acc and 0 is for U-Acc.
    """
    y_test_u, y_pred_u, y_test_k, y_pred_k = [], [], [], []
    for t, p in zip(y_test, y_pred):
        if t == u_label:
            y_test_u.append(t)
            y_pred_u.append(p)
        else:
            y_test_k.append(t)
            y_pred_k.append(p)

    acc_k = accuracy_score(y_test_k, y_pred_k)
    acc_u = accuracy_score(y_test_u, y_pred_u)
    n_acc = (reg_c * acc_k) + ((1 - reg_c) * acc_u)

    return n_acc


def show_roc_curve(y_pred_proba, y_true):
    """Print the ROC curve"""
    # Random classifier
    rand_probs = [0 for _ in range(len(y_true))]
    rand_fpr, rand_tpr, _ = roc_curve(y_true, rand_probs)

    # Actual classifier
    actual_fpr, actual_tpr, _ = roc_curve(y_true, y_pred_proba)
    # Draw roc curves
    pyplot.plot(rand_fpr, rand_tpr, linestyle='--', label='Random')
    pyplot.plot(actual_fpr, actual_tpr, marker='.', label='Actual')
    pyplot.xlabel('False positive rate')
    pyplot.ylabel('True positive rate')
    pyplot.legend()
    pyplot.show()


def conf_matrix(y_true, y_pred):
    c_matrix = confusion_matrix(y_true, y_pred)
    FP = c_matrix.sum(axis=0) - np.diag(c_matrix)
    FN = c_matrix.sum(axis=1) - np.diag(c_matrix)
    TP = np.diag(c_matrix)
    TN = c_matrix.sum() - (FP + FN + TP)

    return FP, FN, TP, TN, c_matrix


def osr_f1_score(TP, FP, FN):
    """Computes the open macro F1 score."""
    f1_scores = []
    for e in range(len(TP)):
        if TP[e] + FP[e] == 0:
            prec = 0
        else:
            prec = TP[e] / (TP[e] + FP[e])

        if TP[e] + FN[e] == 0:
            rec = 0
        else:
            rec = TP[e] / (TP[e] + FN[e])

        if prec + rec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec)/(prec + rec)

        f1_scores.append(f1)

    f1_score = sum(f1_scores) / len(TP)

    return f1_score


def auc_score(fpr, tpr):
    """Calculate the AUROC (AUC) score"""
    auc_osr = auc(fpr, tpr)
    return auc_osr


def cutoff_youdens_j(fpr, tpr, thresholds):
    """Returns the best threshold based on the Youden index."""
    j_scores = np.array(tpr)-np.array(fpr)
    j_ordered = sorted(zip(j_scores, thresholds))
    # print(j_ordered)
    return j_ordered[-1][1]


def statistical_significance(a, b):
    """Computes the P-value of 2 sets of variables."""
    t_stat, p_val = ttest_ind(a, b)

    return p_val