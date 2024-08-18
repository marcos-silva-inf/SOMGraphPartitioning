import numpy as np
from pathlib import Path
from pandas import read_csv, factorize

ROOT_DIR = Path(__file__).parent

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

np.random.seed(42)

# gaussianian data

mean1 = [2, 2]
mean2 = [-2, 2]
mean3 = [0, -2]
cov = [[1, 0], [0, 1]]

N = 100
X1 = np.random.multivariate_normal(mean1, cov, N)
X2 = np.random.multivariate_normal(mean2, cov, N)
X3 = np.random.multivariate_normal(mean3, cov, N)

gaussian_data = np.vstack((X1, X2, X3))
gaussian_labels = np.concatenate((np.zeros(N), np.ones(N), 2 * np.ones(N)))

# Spiral data

N = 350
theta = np.sqrt(np.random.rand(N)) * 2 * np.pi

r_a = 2 * theta + np.pi
data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
x_a = data_a + np.random.randn(N, 2)

r_b = -2 * theta - np.pi
data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
x_b = data_b + np.random.randn(N, 2)

spiral_data = np.vstack([x_a, x_b])
spiral_data = normalize(spiral_data)
spiral_labels = np.append(np.ones(N), np.zeros(N))

# iris data
iris_df = read_csv(str(ROOT_DIR)+'/iris.csv')
iris_labels = factorize(iris_df['variety'])[0]
iris_data = iris_df.iloc[:, :-1].values


# imports

data = {
    'gaussian': gaussian_data,
    'spiral': spiral_data,
    'iris': iris_data
}

normalized_data = {
    'gaussian': normalize(gaussian_data),
    'spiral': normalize(spiral_data),
    'iris': normalize(iris_data)
}

labels = {
    'gaussian': gaussian_labels,
    'spiral': spiral_labels,
    'iris': iris_labels
}