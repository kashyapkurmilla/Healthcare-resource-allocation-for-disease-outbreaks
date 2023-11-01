import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
data = pd.read_csv("C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/datasetAll.csv", encoding='latin-1')

# Handle missing values by filling with the mean
data['Cases - cumulative total'].fillna(data['Cases - cumulative total'].mean(), inplace=True)
data['Deaths - cumulative total'].fillna(data['Deaths - cumulative total'].mean(), inplace=True)
data['TOTAL_VACCINATIONS'].fillna(data['TOTAL_VACCINATIONS'].mean(), inplace=True)

# Select relevant attributes
cases = data['Cases - cumulative total']
deaths = data['Deaths - cumulative total']
vaccinations = data['TOTAL_VACCINATIONS']

# Data exploration and visualization
plt.figure(figsize=(12, 4))

# Plot histograms to visualize data distribution
plt.subplot(131)
plt.hist(cases, bins=20, density=True, alpha=0.6, color='b', label='Cases')
plt.title('Cases - Histogram')
plt.grid(True)

plt.subplot(132)
plt.hist(deaths, bins=20, density=True, alpha=0.6, color='r', label='Deaths')
plt.title('Deaths - Histogram')
plt.grid(True)

plt.subplot(133)
plt.hist(vaccinations, bins=20, density=True, alpha=0.6, color='g', label='Vaccinations')
plt.title('Vaccinations - Histogram')
plt.grid(True)

plt.tight_layout()
plt.show()

# Fit distributions to the data
case_params = norm.fit(cases)
death_params = norm.fit(deaths)
vaccination_params = norm.fit(vaccinations)

# Goodness of fit test
case_ks_stat, case_ks_pval = stats.kstest(cases, 'norm', args=case_params)
death_ks_stat, death_ks_pval = stats.kstest(deaths, 'norm', args=death_params)
vaccination_ks_stat, vaccination_ks_pval = stats.kstest(vaccinations, 'norm', args=vaccination_params)

# Interpretation
print(f"Cases - KS Statistic: {case_ks_stat}, p-value: {case_ks_pval}")
print(f"Deaths - KS Statistic: {death_ks_stat}, p-value: {death_ks_pval}")
print(f"Vaccinations - KS Statistic: {vaccination_ks_stat}, p-value: {vaccination_ks_pval}")

# Visualization of the fitted distributions
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(cases, bins=20, density=True, alpha=0.6, color='b', label='Cases')
plt.plot(np.linspace(min(cases), max(cases), 100), norm.pdf(np.linspace(min(cases), max(cases), 100), *case_params), 'r-')
plt.title('Cases - Fitted Distribution')
plt.grid(True)

plt.subplot(132)
plt.hist(deaths, bins=20, density=True, alpha=0.6, color='r', label='Deaths')
plt.plot(np.linspace(min(deaths), max(deaths), 100), norm.pdf(np.linspace(min(deaths), max(deaths), 100), *death_params), 'b-')
plt.title('Deaths - Fitted Distribution')
plt.grid(True)

plt.subplot(133)
plt.hist(vaccinations, bins=20, density=True, alpha=0.6, color='g', label='Vaccinations')
plt.plot(np.linspace(min(vaccinations), max(vaccinations), 100), norm.pdf(np.linspace(min(vaccinations), max(vaccinations), 100), *vaccination_params), 'b-')
plt.title('Vaccinations - Fitted Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()
