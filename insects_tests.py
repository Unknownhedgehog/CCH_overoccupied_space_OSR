from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
import random
import numpy as np
import dataset_generator
import data_prep
import metrics
import plots
import stream_tests

d_clas_table = PrettyTable()
d_clas_table.title = "Results Regular Classifier"
d_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC"]
s_clas_table = PrettyTable()
s_clas_table.title = "Results Incremental Classifier"
s_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC"]
cc_clas_table = PrettyTable()
cc_clas_table.title = "Results CCH"
cc_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC", "DB index"]
c1_c3_clas_table = PrettyTable()
c1_c3_clas_table.title = "Results P-value C1-CCH"
c1_c3_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1"]
c2_c3_clas_table = PrettyTable()
c2_c3_clas_table.title = "Results P-value C2-CCH"
c2_c3_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1"]

n_samples = [52848]
classes = [6]
x, y = dataset_generator.create_insect_dataset(n_samples=n_samples[0])
openness = [0.1, 0.25, 0.5, 0.7]
for o in openness:
    unknown_class_labels, n_u_samples = data_prep.generate_unknown(o, classes, 0.1, n_samples, 'random')

    c1_accs, c1_k_accs, c1_u_accs, c1_n_accs, c1_m_f1s = [], [], [], [], []
    c2_accs, c2_k_accs, c2_u_accs, c2_n_accs, c2_m_f1s = [], [], [], [], []
    c3_accs, c3_k_accs, c3_u_accs, c3_n_accs, c3_m_f1s, c3_aucs, c3_dbs = [], [], [], [], [], [], []

    x_k_classes, y_k_classes, u_class_samples = data_prep.remove_class(x, y, unknown_class_labels[0])
    # plots.plot_scatter_comparison2(x, y, y_k_classes, unknown_class_labels[0])

    c1_acc_scores, c1_acc_k_scores, c1_acc_u_scores, c1_n_acc_scores, c1_macro_f1_scores = \
        [], [], [], [], []
    c2_acc_scores, c2_acc_k_scores, c2_acc_u_scores, c2_n_acc_scores, c2_macro_f1_scores = \
        [], [], [], [], []
    c3_acc_scores, c3_acc_k_scores, c3_acc_u_scores, c3_n_acc_scores, c3_macro_f1_scores, c3_auc, c3_db = \
        [], [], [], [], [], [], []
    c1_c3_stat = []
    c2_c3_stat = []
    loops = 5
    for e in range(loops):
        x_train, x_test, y_train, y_test = train_test_split(x_k_classes, y_k_classes, test_size=0.2,
                                                            stratify=y_k_classes)
        # Standard scaler
        x_train, x_test, unknown_samples = data_prep.scale_feat(x_train, x_test, u_class_samples)

        # Classifier 1 results
        c1_acc, c1_acc_k, c1_acc_u, c1_n_acc, c1_m_f1 = \
            stream_tests.run_normal_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                               unknown_class_labels[0], n_u_samples[0])
        c1_acc_scores.append(c1_acc)
        c1_acc_k_scores.append(c1_acc_k)
        c1_acc_u_scores.append(c1_acc_u)
        c1_n_acc_scores.append(c1_n_acc)
        c1_macro_f1_scores.append(c1_m_f1)

        # Classifier 2 results
        c2_acc, c2_acc_k, c2_acc_u, c2_n_acc, c2_m_f1 = \
            stream_tests.run_learning_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                                 unknown_class_labels[0], n_u_samples[0])
        c2_acc_scores.append(c2_acc)
        c2_acc_k_scores.append(c2_acc_k)
        c2_acc_u_scores.append(c2_acc_u)
        c2_n_acc_scores.append(c2_n_acc)
        c2_macro_f1_scores.append(c2_m_f1)

        # Classifier 3 results
        n_k_classes = classes[0] - len(unknown_class_labels[0])
        e_threshold = np.arange(0.1, 8, 0.025).round(3).tolist()
        # print(e_threshold)
        c3_acc, c3_acc_k, c3_acc_u, c3_n_acc, c3_m_f1, c3_auc_s, c3_db_index = \
            stream_tests.run_clust_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                              unknown_class_labels[0], n_k_classes, n_u_samples[0], e_threshold)
        c3_acc_scores.append(c3_acc)
        c3_acc_k_scores.append(c3_acc_k)
        c3_acc_u_scores.append(c3_acc_u)
        c3_n_acc_scores.append(c3_n_acc)
        c3_macro_f1_scores.append(c3_m_f1)
        c3_auc.append(c3_auc_s)
        c3_db.append(c3_db_index)

    c1_accs.append(round(sum(c1_acc_scores) / loops, 2))
    c1_k_accs.append(round(sum(c1_acc_k_scores) / loops, 2))
    c1_u_accs.append(round(sum(c1_acc_u_scores) / loops, 2))
    c1_n_accs.append(round(sum(c1_n_acc_scores) / loops, 2))
    c1_m_f1s.append(round(sum(c1_macro_f1_scores) / loops, 2))
    c1_sd_acc = round(np.std(np.array(c1_acc_scores)), 2)
    c1_sd_k_acc = round(np.std(np.array(c1_acc_k_scores)), 2)
    c1_sd_u_acc = round(np.std(np.array(c1_acc_u_scores)), 2)
    c1_sd_n_acc = round(np.std(np.array(c1_n_acc_scores)), 2)
    c1_sd_m_f1s = round(np.std(np.array(c1_macro_f1_scores)), 2)

    c2_accs.append(round(sum(c2_acc_scores) / loops, 2))
    c2_k_accs.append(round(sum(c2_acc_k_scores) / loops, 2))
    c2_u_accs.append(round(sum(c2_acc_u_scores) / loops, 2))
    c2_n_accs.append(round(sum(c2_n_acc_scores) / loops, 2))
    c2_m_f1s.append(round(sum(c2_macro_f1_scores) / loops, 2))
    c2_sd_acc = round(np.std(np.array(c2_acc_scores)), 2)
    c2_sd_k_acc = round(np.std(np.array(c2_acc_k_scores)), 2)
    c2_sd_u_acc = round(np.std(np.array(c2_acc_u_scores)), 2)
    c2_sd_n_acc = round(np.std(np.array(c2_n_acc_scores)), 2)
    c2_sd_m_f1s = round(np.std(np.array(c2_macro_f1_scores)), 2)

    c3_accs.append(round(sum(c3_acc_scores) / loops, 2))
    c3_k_accs.append(round(sum(c3_acc_k_scores) / loops, 2))
    c3_u_accs.append(round(sum(c3_acc_u_scores) / loops, 2))
    c3_n_accs.append(round(sum(c3_n_acc_scores) / loops, 2))
    c3_m_f1s.append(round(sum(c3_macro_f1_scores) / loops, 2))
    c3_aucs.append(round(sum(c3_auc) / loops, 2))
    c3_dbs.append(round(sum(c3_db) / loops, 2))
    c3_sd_acc = round(np.std(np.array(c3_acc_scores)), 2)
    c3_sd_k_acc = round(np.std(np.array(c3_acc_k_scores)), 2)
    c3_sd_u_acc = round(np.std(np.array(c3_acc_u_scores)), 2)
    c3_sd_n_acc = round(np.std(np.array(c3_n_acc_scores)), 2)
    c3_sd_m_f1s = round(np.std(np.array(c3_macro_f1_scores)), 2)
    c3_sd_auc = round(np.std(np.array(c3_auc)), 2)
    c3_sd_db = round(np.std(np.array(c3_db)), 2)

    # Fill tables
    d_clas_table.add_row([str(c1_accs) + "+-" + str(c1_sd_acc), str(c1_k_accs) + "+-" + str(c1_sd_k_acc), str(c1_u_accs) + "+-" + str(c1_sd_u_acc),
                          str(c1_n_accs) + "+-" + str(c1_sd_n_acc), str(c1_m_f1s) + "+-" + str(c1_sd_m_f1s), 0])

    s_clas_table.add_row([str(c2_accs) + "+-" + str(c2_sd_acc), str(c2_k_accs) + "+-" + str(c2_sd_k_acc), str(c2_u_accs) + "+-" + str(c2_sd_u_acc),
                          str(c2_n_accs) + "+-" + str(c2_sd_n_acc), str(c2_m_f1s) + "+-" + str(c2_sd_m_f1s), 0])

    cc_clas_table.add_row([str(c3_accs) + "+-" + str(c3_sd_acc), str(c3_k_accs) + "+-" + str(c3_sd_k_acc), str(c3_u_accs) + "+-" + str(c3_sd_u_acc),
                          str(c3_n_accs) + "+-" + str(c3_sd_n_acc), str(c3_m_f1s) + "+-" + str(c3_sd_m_f1s),
                          str(c3_aucs) + "+-" + str(c3_sd_auc), str(c3_dbs) + "+-" + str(c3_sd_db)])

    c1_c3_stat.append(round(metrics.statistical_significance(c1_acc_scores, c3_acc_scores), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_acc_k_scores, c3_acc_k_scores), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_acc_u_scores, c3_acc_u_scores), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_n_acc_scores, c3_n_acc_scores), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_macro_f1_scores, c3_macro_f1_scores), 5))

    c2_c3_stat.append(round(metrics.statistical_significance(c2_acc_scores, c3_acc_scores), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_acc_k_scores, c3_acc_k_scores), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_acc_u_scores, c3_acc_u_scores), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_n_acc_scores, c3_n_acc_scores), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_macro_f1_scores, c3_macro_f1_scores), 5))

    c1_c3_clas_table.add_row(c1_c3_stat)
    c2_c3_clas_table.add_row(c2_c3_stat)

f_name = "Results of insects datasets"
with open(f_name, 'w') as w:
    w.write(d_clas_table.get_string() + "\n")
    w.write(s_clas_table.get_string() + "\n")
    w.write(cc_clas_table.get_string() + "\n")
    w.write(c1_c3_clas_table.get_string() + "\n")
    w.write(c2_c3_clas_table.get_string() + "\n")
