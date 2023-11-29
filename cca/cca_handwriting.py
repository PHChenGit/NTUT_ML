import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def _center_scale_xy(X, Y):
    # center
    X_mean = np.mean(X, axis=0)
    X -= X_mean

    Y_mean = np.mean(Y, axis=0)
    Y -= Y_mean

    x_std = np.std(X, axis=0, ddof=1)
    x_std[x_std == 0.0] = 1.0
    X = np.divide(X, x_std)

    y_std = np.std(Y, axis=0, ddof=1)
    y_std[y_std == 0.0] = 1.0
    Y = np.divide(Y, y_std)

    return X, Y, X_mean, Y_mean, x_std, y_std


def cca(X1, X2, n_components=2):
    N = X1.shape[1]

    X_mean = np.mean(X1, axis=0)
    X1 = X1 - X_mean

    Y_mean = np.mean(X2, axis=0)
    X2 = X2 - Y_mean

    S_x1x1 = np.dot(X1.T, X1)
    S_x2x2 = np.dot(X2.T, X2)
    S_x1x2 = np.dot(X1.T, X2)
    S_x2x1 = np.dot(X2.T, X1)

    # S_x = S_x1x1_inv * S_x1x2 * Sx2x2_inv * S_x2x1
    S_x1x1_inv = np.linalg.pinv(S_x1x1)
    S_x2x2_inv = np.linalg.pinv(S_x2x2)

    cross_cov_1 = np.dot(np.dot(np.dot(S_x1x1_inv, S_x1x2), S_x2x2_inv), S_x2x1)
    cross_cov_2 = np.dot(np.dot(np.dot(S_x2x2_inv, S_x2x1), S_x1x1_inv), S_x1x2)

    w_eigen_values, w_eigen_vectors = np.linalg.eig(cross_cov_1)

    sorted_indices = np.argsort(w_eigen_values)[::-1]
    w_eigen_values = w_eigen_values[sorted_indices]
    w_eigen_vectors = w_eigen_vectors[:, sorted_indices]

    v_eigen_values, v_eigen_vectors = np.linalg.eig(cross_cov_2)
    sorted_indices = np.argsort(v_eigen_values)[::-1]
    v_eigen_values = v_eigen_values[sorted_indices]
    v_eigen_vectors = v_eigen_vectors[:, sorted_indices]

    W = w_eigen_vectors[:, :n_components]
    V = v_eigen_vectors[:, :n_components]

    a = np.dot(X1, W)
    b = np.dot(X2, V)

    return a, b


def CCA2(X, Y, n_comp=2):
    n, p = X.shape
    q = Y.shape[0]

    S = np.cov(X, Y)
    S_xx = S[:n, :n]
    S_xy = S[:n, n:]
    S_yx = S[n:, :n]
    S_yy = S[n:, n:]

    x_max = np.max(np.abs(S_xx))
    y_max = np.max(np.abs(S_yy))

    S_xx /= x_max
    S_yy /= y_max
    S_xy /= np.sqrt(x_max * y_max)
    S_yx /= np.sqrt(y_max * x_max)

    epsilon = 1e-6
    num_x = S_xx.shape[0]
    num_y = S_yy.shape[0]
    S_xx += epsilon * np.eye(num_x)
    S_yy += epsilon * np.eye(num_y)

    inv_S_xx = np.linalg.pinv(S_xx)
    inv_S_yy = np.linalg.pinv(S_yy)

    M1 = np.dot(np.dot(np.dot(inv_S_xx, S_xy), inv_S_yy), S_yx)
    M2 = np.dot(np.dot(np.dot(inv_S_yy, S_yx), inv_S_xx), S_xy)

    u_x, s_x, v_x = np.linalg.svd(M1)
    u_y, s_y, v_y = np.linalg.svd(M2)



df = pd.read_csv("./datasets/digits_data.csv", header=None)

# d= 32 e=1934
x1 = df.iloc[:, :32].values.astype(dtype=np.float64)
x2 = df.iloc[:, 32:-1].values.astype(dtype=np.float64)
y_data = df.iloc[:, -1].values.astype(dtype=np.float64)

w, v = cca(x1, x2)

plt.figure(figsize=(8, 4))

# Plot the canonical variables
plt.subplot(121)
plt.scatter(w[:, 0], w[:, 1], label='Canonical Variable 1')
plt.scatter(v[:, 0], v[:, 1], label='Canonical Variable 2')

plt.xlabel('Canonical Variables for X')
plt.ylabel('Canonical Variables for Y')
plt.title('Handwriting CCA Result')
plt.legend()

from sklearn.cross_decomposition import CCA

cca = CCA(n_components=2)
u, v2 = cca.fit_transform(x1, x2)

print(f"u shape: {u.shape}\n v2 shape: {v2.shape}\n")

# Plot the canonical variables
plt.subplot(122)
plt.scatter(u[:, 0], u[:, 1], label='Canonical Variable 1')
plt.scatter(v2[:, 0], v2[:, 1], label='Canonical Variable 2')

plt.xlabel('Canonical Variables for X')
plt.ylabel('Canonical Variables for Y')
plt.title('Sklearn CCA Result')
plt.legend()
plt.show()