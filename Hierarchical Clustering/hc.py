import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns

# Load the hospital resources dataset from a CSV file
hospital_data = pd.read_csv('C:/dmpa/healthcare resource allocation/beds.csv')

# Select the relevant attributes for clustering
hospital_attributes = hospital_data[['Total Number of COVID 19 beds', 'ICU beds for COVID', 'Ventilator beds for COVID']]

# Calculate linkage matrix
linkage_matrix = linkage(hospital_attributes, method='ward')

# Set a threshold to determine the number of clusters
threshold = 5  # Adjust this threshold based on your preference

# Assign hospitals to clusters
hospital_data['Cluster'] = fcluster(linkage_matrix, t=threshold, criterion='distance')

# Create a dendrogram and a heatmap side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot the dendrogram on the left subplot
axes[0].set_title("Dendrogram for Hospital Resources Hierarchical Clustering")
dendrogram(linkage_matrix, orientation='top', labels=hospital_data.index, distance_sort='descending', ax=axes[0])
axes[0].set_xlabel("Hospital")
axes[0].set_ylabel("Cluster Distance")
axes[0].set_xticks([])

# Create a heatmap on the right subplot
cluster_data = hospital_data[['Cluster']].sort_index()
sns.heatmap(cluster_data.T, cmap='YlGnBu', annot=True, fmt="d", cbar=False, ax=axes[1])
axes[1].set_title("Cluster Heatmap")

plt.show()

# Print the hospitals in each cluster in list form
clusters = hospital_data['Cluster'].unique()
for cluster in clusters:
    hospitals_in_cluster = hospital_data[hospital_data['Cluster'] == cluster].index.tolist()
    print(f"Cluster {cluster} Hospitals:")
    print(hospitals_in_cluster)
