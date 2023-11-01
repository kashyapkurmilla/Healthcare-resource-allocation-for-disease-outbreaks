import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the datasets from CSV files
hospital_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/beds.csv')
oxygen_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/oxygen.csv')
deaths_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/deaths.csv')
vaccine_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/totalDoses.csv')
cases_data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/totalcases.csv')

# Select the relevant attributes for clustering
hospital_attributes = hospital_data[['Total Number of COVID 19 beds', 'ICU beds for COVID', 'Ventilator beds for COVID']]
oxygen_attributes = oxygen_data[['Oxygen allocated (In Metric Ton)']]
deaths_attributes = deaths_data[['Deaths during 2021 (Till 14th July 2021)']]
vaccine_attributes = vaccine_data[['Total Doses Administered']]
cases_attributes = cases_data[['Total Cases']]

# Choose the number of clusters (you can adjust this)
k = 3

# Initialize and fit the K-Means models for each dataset
kmeans_hospital = KMeans(n_clusters=k, random_state=0)
kmeans_hospital.fit(hospital_attributes)

kmeans_oxygen = KMeans(n_clusters=k, random_state=0)
kmeans_oxygen.fit(oxygen_attributes)

kmeans_deaths = KMeans(n_clusters=k, random_state=0)
kmeans_deaths.fit(deaths_attributes)

kmeans_vaccine = KMeans(n_clusters=k, random_state=0)
kmeans_vaccine.fit(vaccine_attributes)

kmeans_cases = KMeans(n_clusters=k, random_state=0)
kmeans_cases.fit(cases_attributes)

# Add cluster labels to the datasets
hospital_data['Cluster_Label'] = kmeans_hospital.labels_
oxygen_data['Cluster_Label'] = kmeans_oxygen.labels_
deaths_data['Cluster_Label'] = kmeans_deaths.labels_
vaccine_data['Cluster_Label'] = kmeans_vaccine.labels_
cases_data['Cluster_Label'] = kmeans_cases.labels_

# Visualize the clusters for hospital resources (you may need to install matplotlib)
plt.scatter(hospital_attributes['Total Number of COVID 19 beds'], hospital_attributes['ICU beds for COVID'], c=kmeans_hospital.labels_)
plt.xlabel('Total Number of COVID 19 beds')
plt.ylabel('ICU beds for COVID')
plt.title('Hospital Resources Clustering')
plt.show()

# Repeat the visualization steps for other datasets (oxygen, deaths, vaccine, cases)
# Visualize the clusters for oxygen allocation (you may need to install matplotlib)
plt.scatter(oxygen_attributes['Oxygen allocated (In Metric Ton)'], [0] * len(oxygen_attributes), c=kmeans_oxygen.labels_)
plt.xlabel('Oxygen allocated (In Metric Ton)')
plt.title('Oxygen Allocation Clustering')
plt.show()

# Visualize the clusters for deaths (you may need to install matplotlib)
plt.scatter(deaths_attributes['Deaths during 2021 (Till 14th July 2021)'], [0] * len(deaths_attributes), c=kmeans_deaths.labels_)
plt.xlabel('Deaths during 2021 (Till 14th July 2021)')
plt.title('COVID-19 Deaths Clustering')
plt.show()

# Visualize the clusters for vaccine administration (you may need to install matplotlib)
plt.scatter(vaccine_attributes['Total Doses Administered'], [0] * len(vaccine_attributes), c=kmeans_vaccine.labels_)
plt.xlabel('Total Doses Administered')
plt.title('Vaccine Administration Clustering')
plt.show()

# Visualize the clusters for COVID-19 cases (you may need to install matplotlib)
plt.scatter(cases_attributes['Total Cases'], [0] * len(cases_attributes), c=kmeans_cases.labels_)
plt.xlabel('Total Cases')
plt.title('COVID-19 Cases Clustering')
plt.show()
