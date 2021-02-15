import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist_data_path = "./Data/MNIST_100.csv"
housing_data_path = "./Data/housing_training.csv"

mnist_data = pd.read_csv(mnist_data_path)
housing_data = pd.read_csv(housing_data_path)

# ===================================================

# TASK 1: Visualize MNIST data using PCA
# Get column zero for all rows
y = mnist_data.iloc[:, 0]
X = mnist_data.drop('label', axis=1)

print("task1 y: " + str(y.shape))
print("task1 x: " + str(X.shape))

# Project data onto 2 dimensions
pca = PCA(n_components=2)
pca.fit(X)
PCAX = pca.transform(X)

print("task1 pcax: " + str(PCAX.shape))

# Plot data
plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', )
for i in range(len(y)):
    plt.text(PCAX[i:i+1, 0], PCAX[i:i+1, 1], y[i])
# plt.show()

# ===================================================

# TASK 2: Visualize housing data cols K,M,N using violin plot

y = housing_data.iloc[:, 10 + 12:14]
X = housing_data

print(y.shape)
print(X.shape)
