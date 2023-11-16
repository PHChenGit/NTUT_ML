import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset/digits_data.csv", header=None)

x_data = df.iloc[:, :-1].values
y_data = df.iloc[:, -1].values

def zero_mean(data_mat):
    mean_val = np.mean(data_mat, axis=0)
    new_data = data_mat - mean_val
    return new_data, mean_val


def pca(data_mat, k):
    new_data_mat, mean_val = zero_mean(data_mat)
    covar_mat = np.cov(new_data_mat, rowvar=False)
    # 求標準差矩陣的 eigen value & eigen vector
    eigen_val, eigen_vector = np.linalg.eig(np.mat(covar_mat))

    # Sort eigen values in ascending
    sorted_eigen_value_indices = np.argsort(eigen_val)
    top_k_eigen_value = sorted_eigen_value_indices[-1:-(k + 1):-1]
    top_k_eigen_vectors = eigen_vector[:, top_k_eigen_value]

    # print(f"new data: {new_data_mat.shape}, n_eigen_vectors: {top_k_eigen_vectors.shape}")
    # The data in low dimension
    low_dimension_data_mat = np.dot(new_data_mat, top_k_eigen_vectors)
    # Use low dimension data to reconstruct the original data
    reconstruct_mat = np.dot(low_dimension_data_mat, top_k_eigen_vectors.T) + mean_val
    return low_dimension_data_mat, reconstruct_mat

low_dimension_data, new_data = pca(x_data, 2)

x = np.array(low_dimension_data)[:, 0]
y = np.array(low_dimension_data)[:, 1]

plt.scatter(x, y, c=y_data)
# 給圖像加上說明圖示

# Scatter plot with labels
for label in np.unique(y_data):
    indices = np.where(y_data == label)
    plt.scatter(x[indices], y[indices], label=f'Dataset {label}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Plot of Digits Data')

plt.legend()

plt.savefig('./result_img/pca_digits_res.png')
plt.show()
