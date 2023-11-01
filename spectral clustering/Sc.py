import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

# Load the datasets for Deaths, Doses Administered, and Total Cases
deaths_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/deaths.csv')
doses_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/totalDoses.csv')
cases_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/totalcases.csv')

# Merge the datasets based on a common column (e.g., 'State/UT')
# Make sure you have a common identifier column for merging
merged_data = deaths_data.merge(doses_data, on='State/UT').merge(cases_data, on='State/UT')

# Select the relevant features
X = merged_data[['Deaths during 2021 (Till 14th July 2021)', 'Total Doses Administered', 'Total Cases']]

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you can adjust this)
n_clusters = 5  # Adjust the number of clusters as per your analysis

# Initialize and fit the Spectral Clustering model
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=0)
merged_data['Cluster'] = spectral_clustering.fit_predict(X_scaled)

# Create a dictionary to store states/countries in each cluster
cluster_countries = {i: [] for i in range(n_clusters)}

# Collect states/countries in each cluster
for cluster in range(n_clusters):
    countries_in_cluster = merged_data[merged_data['Cluster'] == cluster]['State/UT'].tolist()
    cluster_countries[cluster] = countries_in_cluster

# Print the countries in each cluster in list form
for cluster, countries in cluster_countries.items():
    print(f"Cluster {cluster} Countries:")
    print(countries)

# Explore and analyze the clusters

# Calculate cluster statistics
cluster_stats = merged_data.groupby('Cluster').agg({
    'Deaths during 2021 (Till 14th July 2021)': 'mean',
    'Total Doses Administered': 'mean',
    'Total Cases': 'mean'
})

# Identify outliers
outliers = merged_data[merged_data['Cluster'] == -1]  # States not assigned to any cluster

# Display cluster statistics and outliers
print("\nCluster Statistics:")
print(cluster_stats)
print("\nOutliers:")
print(outliers)

# Plot the clusters using a scatter plot (2D or 3D, as needed)
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['Total Doses Administered'], merged_data['Deaths during 2021 (Till 14th July 2021)'], c=merged_data['Cluster'], cmap='viridis')
plt.xlabel('Total Doses Administered')
plt.ylabel('Deaths during 2021 (Till 14th July 2021)')
plt.title('Spectral Clustering: Clusters of States/UTs')
plt.colorbar()
plt.show()
