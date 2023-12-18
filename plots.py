import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import data_prep
from river import stream


def plot_classifier_boundary(model, x, x_test, y):
    x = np.array(x)
    x_test = np.array(x_test)
    # define bounds of the domain
    min1, max1 = x[:, 0].min() - 5, x[:, 0].max() + 5
    min2, max2 = x[:, 1].min() - 5, x[:, 1].max() + 5
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))
    test_dataset = stream.iter_array(grid)
    y_preds = []
    for g, _ in test_dataset:
        y_pred = model.predict_one(g)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds)
    # reshape the predictions back into a grid
    zz = y_preds.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Greys')
    # create scatter plot for samples from each class
    labels = np.unique(y)
    # colors = [plt.cm.cividis(float(i) / max(labels)) for i in labels]
    colors = [plt.cm.Greys(float(i) / max(labels)) for i in labels]

    for i, label in enumerate(labels):
        xi = [x_test[:, 0][j] for j in range(len(x_test)) if y[j] == label]
        yi = [x_test[:, 1][j] for j in range(len(x_test)) if y[j] == label]
        # plt.scatter(x_test[:, 0], x_test[:, 1], c=y, s=40, edgecolor="k", alpha=1, label=label)
        color = np.array([colors[i]])
        plt.scatter(xi, yi, c=color, s=40, edgecolor="k", alpha=1, label=str(label))
    plt.legend()
    plt.show()


def plot_scatter_data(x, y, plot_title):
    plt.title(plot_title, fontsize="small")
    plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=25, edgecolor="k")
    plt.show()


def plot_scatter_comparison(x, y, y_known, unknown_labels):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    labels = np.unique(y)
    # colors = [plt.cm.cividis(float(i) / max(labels)) for i in labels]
    colors = [plt.cm.Greys(float(i) / max(labels)) for i in labels]

    # ax[0].title("Data with all classes", fontsize="small")
    for i, label in enumerate(labels):
        xi = [x[:, 0][j] for j in range(len(x)) if y[j] == label]
        yi = [x[:, 1][j] for j in range(len(x)) if y[j] == label]
        color = np.array([colors[i]])
        ax[0].scatter(xi, yi, c=color, s=25, edgecolor="k", alpha=1, label=str(label))
    ax[0].legend()

    colors2 = colors
    for n in reversed(unknown_labels):
        colors2.pop(n)

    labels3 = np.unique(y_known)

    for i, label in enumerate(labels3):
        xi = [x[:, 0][j] for j in range(len(x)) if y[j] == label]
        yi = [x[:, 1][j] for j in range(len(x)) if y[j] == label]
        color = np.array([colors[i]])
        ax[1].scatter(xi, yi, c=color, s=25, edgecolor="k", alpha=1, label=str(label))
    ax[1].legend()

    fig.tight_layout()
    plt.show()


def plot_stream_cluster_results(clust_model, x_test, labels):
    x_test = np.array(x_test)
    # labels2 = clust_model.predict_one(x_test)
    # xs = x_test[:, 0]
    # ys = x_test[:, 1]
    # Make a scatter plot of xs and ys, using labels to define the colors
    labels2 = np.unique(labels)
    # colors = [plt.cm.cividis(float(i) / max(labels2)) for i in labels2]
    colors = [plt.cm.Greys(float(i) / max(labels2)) for i in labels2]
    for i, label in enumerate(labels2):
        xi = [x_test[:, 0][j] for j in range(len(x_test)) if labels[j] == label]
        yi = [x_test[:, 1][j] for j in range(len(x_test)) if labels[j] == label]
        # plt.scatter(x_test[:, 0], x_test[:, 1], c=y, s=40, edgecolor="k", alpha=1, label=label)
        color = np.array([colors[i]])
        plt.scatter(xi, yi, c=color, s=40, edgecolor="k", alpha=1, label=str(label))
    plt.legend()
    plt.margins(0.3, 0.3)
    # plt.scatter(xs, ys, c=labels, alpha=0.5)
    # Assign the cluster centers: centroids
    centroids = np.array(data_prep.get_cluster_centers(clust_model))
    # Assign the columns of centroids: centroids_x, centroids_y
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]
    # Make a scatter plot of centroids_x and centroids_y
    plt.scatter(centroids_x, centroids_y, marker='D', s=50)
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc):
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Estimator")
    display.plot()
    plt.show()
