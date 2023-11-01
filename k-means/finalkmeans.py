import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from the CSV file into a DataFrame
df = pd.read_csv('datasetAll.csv', encoding='latin-1')

# Select the columns you want to cluster on
X = df[["Cases - cumulative total", "Deaths - cumulative total", "TOTAL_VACCINATIONS"]]

# Define the number of clusters
num_clusters = 7

# Initialize random centroids
np.random.seed(0)  # Set a random seed for reproducibility
centroids = X.sample(n=num_clusters, random_state=0).values

# Maximum number of iterations
max_iterations = 100

for _ in range(max_iterations):
    # Assign each data point to the nearest centroid
    distances = np.sqrt(np.sum((X.values[:, np.newaxis, :] - centroids)**2, axis=2))
    cluster_assignments = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(num_clusters)])

    # Check for convergence
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

# Add cluster labels to the DataFrame
df["Cluster"] = cluster_assignments

# Calculate the sums of cases and vaccinations for each cluster
cluster_sums = df.groupby("Cluster")[["Cases - cumulative total", "TOTAL_VACCINATIONS"]].sum()

# Plot a pie chart
labels = [f"Cluster {i}" for i in range(num_clusters)]
cases_sum = cluster_sums["Cases - cumulative total"].values
vaccinations_sum = cluster_sums["TOTAL_VACCINATIONS"].values
# Print cluster assignments vs. country
print(df[["Name", "Cluster"]])


fig, ax = plt.subplots()
ax.pie(cases_sum, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Distribution of Cases Across Clusters")
plt.show()

fig, ax = plt.subplots()
ax.pie(vaccinations_sum, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Distribution of Vaccinations Across Clusters")
plt.show()