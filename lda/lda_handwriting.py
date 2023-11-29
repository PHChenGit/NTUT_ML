import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA(data, target, n_components=2):
    m, n = data.shape
    clusters = np.unique(target)

    S_W = np.zeros((n, n))
    for i in clusters:
        data_i = data[target == i]
        mean_i = data_i.mean(0)
        S_i = np.mat(data_i - mean_i).T * np.mat(data_i - mean_i)
        S_W += S_i

    S_B = np.zeros((n, n))
    overall_mean = data.mean(0)
    for i in clusters:
        N_i = data[target == i].shape[0]
        mean_i = data[target == i].mean(0)
        S_B_i = N_i * np.mat(mean_i - overall_mean).T * np.mat(mean_i - overall_mean)
        S_B += S_B_i

    S_W += 1e-5 * np.identity(n)
    S_W_inverse = np.linalg.inv(S_W)
    S = S_W_inverse.dot(S_B)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    indices = np.argsort(eigen_values)[::-1]
    # sorted_eigen_values = indices[:, :2]
    w = eigen_vectors[:, indices]
    n_dim_data = np.dot(data, w)

    return n_dim_data[:, :n_components]


def LDA_2(data, target, n_components=2):
    classes, cnts = np.unique(target, return_counts=True)
    n_samples, n_features = data.shape

    _priors = cnts.astype(dtype=np.float64) / float(target.shape[0])
    if np.any(_priors < 0):
        raise ValueError("priors must be non-negative")

    if np.abs(np.sum(_priors) - 1.0) > 1e-5:
        _priors = _priors / np.sum(_priors)

    cov = np.zeros((n_features, n_features))
    for idx, group in enumerate(classes):
        X_group = data[target == group, :]
        cov += _priors[idx] * np.atleast_2d(np.cov(X_group.T))

    S_w = cov
    S_t = np.cov(data.T)
    S_b = S_t - S_w

    S = np.linalg.inv(S_w)
    eigen_values, eigen_vectors = np.linalg.eig(S.dot(S_b))
    sorted_eigen_vectors = eigen_vectors[:, np.argsort(eigen_values)[::-1]]

    new_X = data @ sorted_eigen_vectors

    return new_X[:, :n_components]

df = pd.read_csv("./datasets/digits_data.csv", header=None)

x_data = df.iloc[:, :-1].values.astype(dtype=np.float64)
y_data = df.iloc[:, -1].values.astype(dtype=np.float64)

lda_res = LDA(x_data, y_data)
# lda_res = LDA_2(x_data, y_data, 2)


sklearn_lda = LinearDiscriminantAnalysis(n_components=2)
sklearn_lda_res = sklearn_lda.fit_transform(x_data, y_data)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title("My handwriting LDA")
plt.scatter(lda_res[:, 0], lda_res[:, 1], c=y_data)

plt.subplot(122)
plt.title("sklearn LDA")
plt.scatter(sklearn_lda_res[:, 0], sklearn_lda_res[:, 1], c=y_data)

plt.show()

