import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data = pd.read_csv('C:/Users/kashy/Desktop/five/lab/dmpa/healthcare resource allocation/cumulative.csv')

# Data Cleaning (if needed)
# Check for missing values and perform data cleaning steps here.
# For example: data.dropna(inplace=True)

# Drop non-numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Data Exploration
# Summary statistics
print("Summary Statistics:")
print(numeric_data.describe())

# Histograms
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(numeric_data['Deaths during 2021 (Till 14th July 2021)'], kde=True)
plt.title('Distribution of Deaths during 2021')

plt.subplot(2, 2, 2)
sns.histplot(numeric_data['Total Doses Administered'], kde=True)
plt.title('Distribution of Total Doses Administered')

plt.subplot(2, 2, 3)
sns.histplot(numeric_data['Oxygen allocated (In Metric Ton)'], kde=True)
plt.title('Distribution of Oxygen allocated')

plt.subplot(2, 2, 4)
sns.histplot(numeric_data['Total Cases'], kde=True)
plt.title('Distribution of Total Cases')

plt.tight_layout()
plt.show()

# Normality Test (Shapiro-Wilk)
print("Shapiro-Wilk Test for Normality:")
print("Deaths during 2021:", stats.shapiro(numeric_data['Deaths during 2021 (Till 14th July 2021)']))
print("Total Doses Administered:", stats.shapiro(numeric_data['Total Doses Administered']))
print("Oxygen allocated:", stats.shapiro(numeric_data['Oxygen allocated (In Metric Ton)']))
print("Total Cases:", stats.shapiro(numeric_data['Total Cases']))

# Correlation Analysis within each group
grouped_data = data.groupby('State/UT')
for state, group in grouped_data:
    numeric_group = group.select_dtypes(include=['int64', 'float64'])
    print(f"State/UT: {state}")
    print("Summary Statistics:")
    print(numeric_group.describe())
    print("Correlation Matrix:")
    print(numeric_group.corr())

# Correlation Analysis
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Statistical Testing (if needed)
# Example t-test:
# t_stat, p_value = stats.ttest_ind(group1['Deaths during 2021 (Till 14th July 2021)'], group2['Deaths during 2021 (Till 14th July 2021)'])

# Data Interpretation and Reporting
# Based on the analysis, provide insights and conclusions.
