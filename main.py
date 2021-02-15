import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist_data_path = "./Data/MNIST_100.csv"
housing_data_path = "./Data/housing_training.csv"

# ===================================================

# TASK 1: Visualize MNIST data using PCA
# Read MNIST data from csv
mnist_data = pd.read_csv(mnist_data_path)

# Get column zero for all rows
y = mnist_data.iloc[:, 0]
X = mnist_data.drop('label', axis=1)

# Project data onto 2 dimensions
pca = PCA(n_components=2)
pca.fit(X)
PCAX = pca.transform(X)

# Plot data
fig0, ax0 = plt.subplots()
sp = plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', )
for i in range(len(y)):
   plt.text(PCAX[i:i+1, 0], PCAX[i:i+1, 1], y[i])

# Set labels
ax0.set_title('MNIST Data')

# ===================================================

# TASK 2: Visualize housing data cols K, M, N using violin plot

# Read housing data columns K, M, N from csv
housing_data = pd.read_csv(housing_data_path, usecols=[10, 12, 13])

# Create violin plot
fig1, ax1 = plt.subplots()
bp = ax1.violinplot(housing_data)

# Set labels
ax1.set_title('Housing Data: K, M, N Distribution')
ax1.set_xlabel('Columns')
ax1.set_ylabel('Values')

# Set x-axis tick labels
xticklabels = ['K', 'M', 'N']
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(xticklabels)

# ===================================================

# TASK 3: Visualize housing data col A as a histogram

# Read housing data column A from csv
housing_data1 = pd.read_csv(housing_data_path, usecols=[0])

# Create histogram
fig2, ax2 = plt.subplots()
hg = ax2.hist(housing_data1)

# Set labels
ax2.set_title('Housing Data: Column A Distribution')
ax2.set_xlabel('Values')
ax2.set_ylabel('Number of Items')

# ===================================================

# Display all 3 figures
plt.show()

# ===================================================
