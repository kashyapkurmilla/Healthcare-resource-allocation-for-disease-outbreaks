import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Load the new dataset with the specified encoding
data = pd.read_csv('C:/dmpa/healthcare resource allocation/datasetAll.csv', encoding='latin-1')

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data[['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS']])
data[['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS']] = data_imputed

# Select the relevant features
X = data[['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS']]

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you can adjust this)
n_clusters = 10  # Adjust the number of clusters as per your analysis

# Initialize and fit the Spectral Clustering model
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=0)
data['Cluster'] = spectral_clustering.fit_predict(X_scaled)

# Explore and analyze the clusters

# Cluster Statistics
cluster_stats = data.groupby('Cluster')[['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS']].agg(['mean', 'std'])
print("Cluster Statistics:")
print(cluster_stats)

# Cluster Visualization
sns.pairplot(data, hue='Cluster', vars=['Cases - cumulative total', 'Deaths - cumulative total', 'TOTAL_VACCINATIONS'])
plt.show()

# Print the countries in each cluster in list form
clusters = data['Cluster'].unique()
for cluster in clusters:
    countries_in_cluster = data[data['Cluster'] == cluster]['Name'].tolist()
    print(f"Cluster {cluster} Countries:")
    print(countries_in_cluster)
