import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

# Load your dataset
data = pd.read_csv('C:/dmpa/healthcare resource allocation/datasetAll.csv', encoding='latin-1')

# Check for missing values
print(data.isnull().sum())

# Handle missing values (e.g., fill with mean or remove rows with missing values)
data = data.dropna()  # Remove rows with missing values

# Select relevant attributes
attributes = data[['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS']]

# Normalize the data
scaler = StandardScaler()
attributes_scaled = scaler.fit_transform(attributes)

# Perform hierarchical clustering
linkage_matrix = linkage(attributes_scaled, method='ward')

# Set the size of the plot
plt.figure(figsize=(16, 8))

# Plot a dendrogram to visualize the clusters
dendrogram(linkage_matrix, orientation="top", labels=data['Name'].tolist())
plt.show()

# Specify the number of clusters
n_clusters = 7  # Adjust this value as needed

# Perform clustering
cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
data['Cluster'] = cluster_model.fit_predict(attributes_scaled)

# Create a dictionary to store countries in each cluster
cluster_countries = {i: [] for i in range(n_clusters)}

# Collect country names in each cluster
for index, row in data.iterrows():
    cluster = row['Cluster']
    country = row['Name']
    cluster_countries[cluster].append(country)

# Print the countries in each cluster
for cluster, countries in cluster_countries.items():
    print(f"Cluster {cluster + 1} Countries:")
    print(countries)

# Create a heatmap to represent the clusters
cluster_data = data[['Cluster']].sort_index()
plt.figure(figsize=(10, 6))  # Increase the size of the heatmap
sns.heatmap(cluster_data.T, cmap='YlGnBu', annot=True, fmt="d", cbar=False, annot_kws={'size': 12})  # Adjust font size
plt.title("Cluster Heatmap")
plt.show()


#in heatmaps coloumn=s represent diffrent smaples and rows represent diffrent measurement genes
#hc orders the rows and coloumns based on similarity - easy to see corelations in the data
#once we have a sub cluster we have to decide how should it be compared to other rows

#Hierarchical clustering builds a tree-like structure, known as a dendrogram, which allows you
#   to visualize the relationships and similarities between data points or clusters at different levels of granularity.


# This method is particularly useful when you want to explore the inherent structure of your data without specifying the
# number of clusters beforehan

#In agglomerative hierarchical clustering, you start with each data point as its own cluster and then iteratively merge
# the most similar clusters together until all data points belong to a single cluster

#If two branches are merged at a very low height in the dendrogram, it indicates that these clusters were very similar
# or close to each other in terms of the chosen distance metric (e.g., Euclidean distance or Ward linkage).

# The code performs hierarchical agglomerative clustering on the dataset, and the output includes a dendrogram showing the hierarchical structure and a heatmap displaying the cluster assignments
# for each country. These visualizations help you understand the structure and relationships within the data.