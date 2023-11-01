import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset from a CSV file
data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/cumulative.csv')

# Fill missing values with zeros
data.fillna(0, inplace=True)

# Select the relevant columns for clustering
X = data[['Deaths during 2021 (Till 14th July 2021)', 'Total Doses Administered', 'Oxygen allocated (In Metric Ton)',
          'Total Cases']]

# Choose the number of clusters (you can adjust this)
k = 3

# Convert the data to a NumPy array
X = X.to_numpy()

# Define the number of iterations and initialize cluster centroids
max_iterations = 100
n_samples, n_features = X.shape
centroids = X[np.random.choice(n_samples, k, replace=False)]

# Initialize variable to track changes in cluster assignments
old_labels = np.zeros(n_samples)

# Perform K-Means clustering
for _ in range(max_iterations):
    # Assign each point to the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update the centroids based on the mean of the assigned points
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    # Check for convergence
    if np.array_equal(labels, old_labels):
        break

    centroids = new_centroids
    old_labels = labels.copy()

# Print the cluster centers
print("Cluster Centers:")
for i, center in enumerate(centroids):
    print(f"Cluster {i + 1}: {center}")

# Add the cluster labels to your dataset
data['Cluster_Label'] = labels

# Visualize the clusters (you may need to install matplotlib)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i in range(k):
    cluster_data = data[data['Cluster_Label'] == i]
    plt.scatter(cluster_data['Deaths during 2021 (Till 14th July 2021)'], cluster_data['Total Doses Administered'],
                c=colors[i], label=f'Cluster {i + 1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='x', label='Cluster Centers')
plt.xlabel('Deaths during 2021 (Till 14th July 2021)')
plt.ylabel('Total Doses Administered')
plt.legend()
plt.show()
