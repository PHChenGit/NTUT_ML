import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


df = pd.read_csv('./datasets/digits_data.csv', header=None)

# print(df.head())

x_data = df.iloc[:, :-1].values
y_data = df.iloc[:, -1].values

X = x_data
# print(f"X shape: {X.shape}\n")


"""
Step 1. 取得距離矩陣 D
Step 2. 利用 D^2 計算內積矩陣 K
Step 3. 找出 K 的特徵值分解, 也就是 K 的 eigen value & eigen vector
Step 4. eigen value 按照大到小排序, 求近似保距向量
"""

def MDS(data_mat, k):
    # Step 1.
    def cal_pairwise_dist(x):
        # sum_x = np.sum(np.square(x), 1)
        # dist_mat = sum_x + sum_x + (-2 * np.dot(x, x.T))
        dist_mat = pairwise_distances(x)
        return dist_mat

    def cal_B(D):
        n, m = D.shape
        DD = D @ D
        D_i = np.sum(DD, axis=1) / n
        D_j = np.sum(DD, axis=0) / n
        D_ij = np.sum(DD) / (n**2)

        res = (D_i + D_j - D_ij - DD) / 2
        return res

    def cal_B_2(D):
        n = D.shape[0]
        H = np.eye(n) - (np.ones((n, n))/n)
        # yy = -1 / 2 * H @ D @ D @ H
        yy = -1 / 2 * np.dot(np.dot(H, D), H)
        return yy

    D = cal_pairwise_dist(data_mat)
    B = cal_B_2(D)

    eigen_value, eigen_vector = np.linalg.eig(B)
    indices = np.argsort(-eigen_value)
    sorted_eigen_values = eigen_value[indices]
    sorted_eigen_vectors = eigen_vector[:, indices]
    top_k_eigen_values = np.diag(sorted_eigen_values[:k])
    top_k_eigen_vectors = sorted_eigen_vectors[:, :k]

    Z = np.dot(np.sqrt(top_k_eigen_values), top_k_eigen_vectors.T).T

    return Z

mds_res = MDS(X, 2)

def plot_mds(data, title):
    x = np.array(data)[:, 0]
    y = np.array(data)[:, 1]

    plt.scatter(x, y, c=y_data)

    plt.title(title)
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')

    # for label in np.unique(y_data):
    #     indices = np.where(y_data == label)
    #     plt.scatter(x[indices], y[indices], label=f'Dataset {label}')

    plt.legend()
    plt.show()

plot_mds(mds_res, "Handwritting MDS with MNIST datasets 2")


# from sklearn import manifold
#
# embedding = manifold.MDS(n_components=2, normalized_stress='auto')
# X_reduced = embedding.fit_transform(X)
#
# print('Dimesnion of X after MDS = ', X_reduced.shape)
#
# plot_mds(X_reduced, "MDS (from package) Visualization of Digits Dataset")

# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_data)
# plt.colorbar(label='Digit Label', ticks=range(10))
# plt.title("MDS (from package) Visualization of Digits Dataset")
# plt.xlabel("MDS Dimension 1")
# plt.ylabel("MDS Dimension 2")
# plt.show()