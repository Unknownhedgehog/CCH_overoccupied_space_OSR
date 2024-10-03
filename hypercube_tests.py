from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
import numpy as np
import dataset_generator
import data_prep
import plots
import metrics
import stream_tests
import hypercube_params as cl_p
import argparse

n_samples = cl_p.n_samples
centers_classes = cl_p.centers_classes
features = cl_p.features
class_sep = cl_p.class_sep
rand_state = cl_p.rand_state
mc = cl_p.mc
p_u_s = cl_p.p_u_s

parser = argparse.ArgumentParser(description="Choose model type")
parser.add_argument('--model', type=str, required=True, help="Specify 'linear' or 'tree'")
args = parser.parse_args()
model_type = args.model

d_clas_table = PrettyTable()
d_clas_table.title = "Results Static"
d_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC"]
s_clas_table = PrettyTable()
s_clas_table.title = "Results Incremental"
s_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC"]
cc_clas_table = PrettyTable()
cc_clas_table.title = "Results sOSR"
cc_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1", "AUC", "DB index"]
c1_c3_clas_table = PrettyTable()
c1_c3_clas_table.title = "Results P-value Static-sOSR"
c1_c3_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1"]
c2_c3_clas_table = PrettyTable()
c2_c3_clas_table.title = "Results P-value Incremental-sOSR"
c2_c3_clas_table.field_names = ["Acc", "K-Acc", "U-Acc", "N-Acc", "F1"]

for m in mc:
    print("Computing with MC: ", m)
    unknown_class_labels, n_u_samples = data_prep.generate_unknown(m, centers_classes, p_u_s, n_samples, 'random')
    c1_accs, c1_k_accs, c1_u_accs, c1_n_accs, c1_m_f1s = [], [], [], [], []
    c2_accs, c2_k_accs, c2_u_accs, c2_n_accs, c2_m_f1s = [], [], [], [], []
    c3_accs, c3_k_accs, c3_u_accs, c3_n_accs, c3_m_f1s, c3_aucs, c3_dbs = [], [], [], [], [], [], []
    n_datasets = len(n_samples)
    c1_c3_stat = []
    c2_c3_stat = []
    for i in range(n_datasets):
        x, y = dataset_generator.create_class_dataset(n_samples[i], features[i], centers_classes[i], class_sep[i],
                                                      rand_state[i])
        x_k_classes, y_k_classes, u_class_samples = data_prep.remove_class(x, y, unknown_class_labels[i])
        # plots.plot_scatter_comparison(x, y, y_k_classes, unknown_class_labels[i])

        c1_acc_scores, c1_acc_k_scores, c1_acc_u_scores, c1_n_acc_scores, c1_macro_f1_scores = \
            [], [], [], [], []
        c2_acc_scores, c2_acc_k_scores, c2_acc_u_scores, c2_n_acc_scores, c2_macro_f1_scores = \
            [], [], [], [], []
        c3_acc_scores, c3_acc_k_scores, c3_acc_u_scores, c3_n_acc_scores, c3_macro_f1_scores, c3_auc, c3_db = \
            [], [], [], [], [], [], []

        loops = 20
        for e in range(loops):
            x_train, x_test, y_train, y_test = train_test_split(x_k_classes, y_k_classes, test_size=0.2,
                                                                stratify=y_k_classes)
            # Standard scaler
            x_train, x_test, unknown_samples = data_prep.scale_feat(x_train, x_test, u_class_samples)

            # Classifier 1 results
            c1_acc, c1_acc_k, c1_acc_u, c1_n_acc, c1_m_f1 = \
                stream_tests.run_normal_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                                   unknown_class_labels[i], n_u_samples[i], model_type)
            c1_acc_scores.append(c1_acc)
            c1_acc_k_scores.append(c1_acc_k)
            c1_acc_u_scores.append(c1_acc_u)
            c1_n_acc_scores.append(c1_n_acc)
            c1_macro_f1_scores.append(c1_m_f1)

            # Classifier 2 results
            c2_acc, c2_acc_k, c2_acc_u, c2_n_acc, c2_m_f1 = \
                stream_tests.run_learning_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                                     unknown_class_labels[i], n_u_samples[i], model_type)
            c2_acc_scores.append(c2_acc)
            c2_acc_k_scores.append(c2_acc_k)
            c2_acc_u_scores.append(c2_acc_u)
            c2_n_acc_scores.append(c2_n_acc)
            c2_macro_f1_scores.append(c2_m_f1)

            # CCH results
            n_k_classes = centers_classes[i] - len(unknown_class_labels[i])
            e_threshold = np.arange(0.1, 5, 0.05).round(2).tolist()
            c3_acc, c3_acc_k, c3_acc_u, c3_n_acc, c3_m_f1, c3_auc_s, c3_db_index = \
                stream_tests.run_clust_clas_tests(x_train, x_test, y_train, y_test, unknown_samples,
                                                  unknown_class_labels[i], n_k_classes, n_u_samples[i], e_threshold,
                                                  model_type)
            c3_acc_scores.append(c3_acc)
            c3_acc_k_scores.append(c3_acc_k)
            c3_acc_u_scores.append(c3_acc_u)
            c3_n_acc_scores.append(c3_n_acc)
            c3_macro_f1_scores.append(c3_m_f1)
            c3_auc.append(c3_auc_s)
            c3_db.append(c3_db_index)

        c1_accs.append(sum(c1_acc_scores) / loops)
        c1_k_accs.append(sum(c1_acc_k_scores) / loops)
        c1_u_accs.append(sum(c1_acc_u_scores) / loops)
        c1_n_accs.append(sum(c1_n_acc_scores) / loops)
        c1_m_f1s.append(sum(c1_macro_f1_scores) / loops)

        c2_accs.append(sum(c2_acc_scores) / loops)
        c2_k_accs.append(sum(c2_acc_k_scores) / loops)
        c2_u_accs.append(sum(c2_acc_u_scores) / loops)
        c2_n_accs.append(sum(c2_n_acc_scores) / loops)
        c2_m_f1s.append(sum(c2_macro_f1_scores) / loops)

        c3_accs.append(sum(c3_acc_scores) / loops)
        c3_k_accs.append(sum(c3_acc_k_scores) / loops)
        c3_u_accs.append(sum(c3_acc_u_scores) / loops)
        c3_n_accs.append(sum(c3_n_acc_scores) / loops)
        c3_m_f1s.append(sum(c3_macro_f1_scores) / loops)
        c3_aucs.append(sum(c3_auc) / loops)
        c3_dbs.append(sum(c3_db) / loops)

    c1_mean_acc = round(sum(c1_accs) / n_datasets, 2)
    c1_mean_k_acc = round(sum(c1_k_accs) / n_datasets, 2)
    c1_mean_u_acc = round(sum(c1_u_accs) / n_datasets, 2)
    c1_mean_n_acc = round(sum(c1_n_accs) / n_datasets, 2)
    c1_mean_m_f1s = round(sum(c1_m_f1s) / n_datasets, 2)
    c1_sd_acc = round(np.std(np.array(c1_accs)), 2)
    c1_sd_k_acc = round(np.std(np.array(c1_k_accs)), 2)
    c1_sd_u_acc = round(np.std(np.array(c1_u_accs)), 2)
    c1_sd_n_acc = round(np.std(np.array(c1_n_accs)), 2)
    c1_sd_m_f1s = round(np.std(np.array(c1_m_f1s)), 2)

    c2_mean_acc = round(sum(c2_accs) / n_datasets, 2)
    c2_mean_k_acc = round(sum(c2_k_accs) / n_datasets, 2)
    c2_mean_u_acc = round(sum(c2_u_accs) / n_datasets, 2)
    c2_mean_n_acc = round(sum(c2_n_accs) / n_datasets, 2)
    c2_mean_m_f1s = round(sum(c2_m_f1s) / n_datasets, 2)
    c2_sd_acc = round(np.std(np.array(c2_accs)), 2)
    c2_sd_k_acc = round(np.std(np.array(c2_k_accs)), 2)
    c2_sd_u_acc = round(np.std(np.array(c2_u_accs)), 2)
    c2_sd_n_acc = round(np.std(np.array(c2_n_accs)), 2)
    c2_sd_m_f1s = round(np.std(np.array(c2_m_f1s)), 2)

    c3_mean_acc = round(sum(c3_accs) / n_datasets, 2)
    c3_mean_k_acc = round(sum(c3_k_accs) / n_datasets, 2)
    c3_mean_u_acc = round(sum(c3_u_accs) / n_datasets, 2)
    c3_mean_n_acc = round(sum(c3_n_accs) / n_datasets, 2)
    c3_mean_m_f1s = round(sum(c3_m_f1s) / n_datasets, 2)
    c3_mean_auc = round(sum(c3_aucs) / n_datasets, 2)
    c3_mean_db = round(sum(c3_dbs) / n_datasets, 2)
    c3_sd_acc = round(np.std(np.array(c3_accs)), 2)
    c3_sd_k_acc = round(np.std(np.array(c3_k_accs)), 2)
    c3_sd_u_acc = round(np.std(np.array(c3_u_accs)), 2)
    c3_sd_n_acc = round(np.std(np.array(c3_n_accs)), 2)
    c3_sd_m_f1s = round(np.std(np.array(c3_m_f1s)), 2)
    c3_sd_auc = round(np.std(np.array(c3_aucs)), 2)
    c3_sd_db = round(np.std(np.array(c3_dbs)), 2)

    c1_c3_stat.append(round(metrics.statistical_significance(c1_accs, c3_accs), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_k_accs, c3_k_accs), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_u_accs, c3_u_accs), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_n_accs, c3_n_accs), 5))
    c1_c3_stat.append(round(metrics.statistical_significance(c1_m_f1s, c3_m_f1s), 5))

    c2_c3_stat.append(round(metrics.statistical_significance(c2_accs, c3_accs), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_k_accs, c3_k_accs), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_u_accs, c3_u_accs), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_n_accs, c3_n_accs), 5))
    c2_c3_stat.append(round(metrics.statistical_significance(c2_m_f1s, c3_m_f1s), 5))

    # Fill tables
    d_clas_table.add_row([str(c1_mean_acc) + " +- " + str(c1_sd_acc), str(c1_mean_k_acc) + " +- " + str(c1_sd_k_acc),
                          str(c1_mean_u_acc) + " +- " + str(c1_sd_u_acc), str(c1_mean_n_acc) + " +- " + str(c1_sd_n_acc),
                          str(c1_mean_m_f1s) + " +- " + str(c1_sd_m_f1s), "-"])

    s_clas_table.add_row([str(c2_mean_acc) + " +- " + str(c2_sd_acc), str(c2_mean_k_acc) + " +- " + str(c2_sd_k_acc),
                          str(c2_mean_u_acc) + " +- " + str(c2_sd_u_acc), str(c2_mean_n_acc) + " +- " + str(c2_sd_n_acc),
                          str(c2_mean_m_f1s) + " +- " + str(c2_sd_m_f1s), "-"])

    cc_clas_table.add_row([str(c3_mean_acc) + "+-" + str(c3_sd_acc), str(c3_mean_k_acc) + "+-" + str(c3_sd_k_acc),
                           str(c3_mean_u_acc) + "+-" + str(c3_sd_u_acc), str(c3_mean_n_acc) + "+-" + str(c3_sd_n_acc),
                           str(c3_mean_m_f1s) + "+-" + str(c3_sd_m_f1s), str(c3_mean_auc) + "+-" + str(c3_sd_auc),
                           str(c3_mean_db) + "+-" + str(c3_sd_db)])

    c1_c3_clas_table.add_row(c1_c3_stat)
    c2_c3_clas_table.add_row(c2_c3_stat)

f_name = "Results of hypercube datasets " + model_type + " classifier"
with open(f_name, 'w') as w:
    w.write(d_clas_table.get_string() + "\n")
    w.write(s_clas_table.get_string() + "\n")
    w.write(cc_clas_table.get_string() + "\n")
    w.write(c1_c3_clas_table.get_string() + "\n")
    w.write(c2_c3_clas_table.get_string() + "\n")