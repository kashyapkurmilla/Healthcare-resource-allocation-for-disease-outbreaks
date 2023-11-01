import numpy as np
from sklearn.cluster import KMeans

# Define your dataset as a list of tuples (state, value)
data = [
    ("United States of America", 103436829),
    ("China", 99312876),
    ("India", 44998838),
    ("France", 38997490),
    ("Germany", 38437756),
    ("Brazil", 37721749),
    ("Republic of Korea", 34571873),
    ("Japan", 33803572),
    ("Italy", 26082645),
    ("The United Kingdom", 24743787),
    ("Russian Federation", 23029404),
    ("Türkiye", 17004677),
    ("Spain", 13980340),
    ("Viet Nam", 11623698),
    ("Australia", 11608166),
    ("Argentina", 10054576),
    ("Netherlands", 8618815),
    ("Mexico", 7690908),
    ("Iran (Islamic Republic of)", 7617752),
    ("Indonesia", 6813429),
    ("Poland", 6523335),
    ("Colombia", 6381470),
    ("Austria", 6081287),
    ("Portugal", 5621015),
    ("Ukraine", 5520483),
    ("Greece", 5405742),
    ("Chile", 5288823),
    ("Malaysia", 5129131),
    ("Israel", 4839886),
    ("Belgium", 4817196),
    ("Thailand", 4757473),
    ("Canada", 4712542),
    ("Czechia", 4651538),
    ("Peru", 4520727),
    ("Switzerland", 4413452),
    ("Philippines", 4113434),
    ("South Africa", 4072533),
    ("Romania", 3455207),
    ("Denmark", 3417017),
    ("Sweden", 2717141),
    ("Singapore", 2594809),
    ("Serbia", 2552178),
    ("Iraq", 2465545),
    ("New Zealand", 2385438),
    ("Hungary", 2206311),
    ("Bangladesh", 2045734),
    ("Slovakia", 1867525),
    ("Georgia", 1855289),
    ("Jordan", 1746997),
    ("Ireland", 1721152),
    ("Pakistan", 1580631),
    ("Kazakhstan", 1502857),
    ("Norway", 1489076),
    ("Finland", 1486084),
    ("Slovenia", 1345384),
    ("Lithuania", 1329905),
    ("Bulgaria", 1302188),
    ("Morocco", 1276812),
    ("Croatia", 1275337),
    ("Guatemala", 1271070),
    ("Puerto Rico", 1252713),
    ("Lebanon", 1239904),
    ("Costa Rica", 1238884),
    ("Bolivia (Plurinational State of)", 1208716),
    ("Tunisia", 1153361),
    ("Cuba", 1115103),
    ("Ecuador", 1069135),
    ("United Arab Emirates", 1067030),
    ("Panama", 1047555),
    ("Uruguay", 1039238),
    ("Mongolia", 1011116),
    ("Nepal", 1003441),
    ("Belarus", 994037),
    ("Latvia", 976316),
    ("Saudi Arabia", 841469),
    ("Azerbaijan", 833189),
    ("Paraguay", 735853),
    ("occupied Palestinian territory, including east Jerusalem", 703228),
    ("Bahrain", 696614),
    ("Sri Lanka", 672589),
    ("Dominican Republic", 667073),
    ("Kuwait", 666333),
    ("Cyprus", 660854),
    ("Myanmar", 641280),
    ("Republic of Moldova", 623684),
    ("Estonia", 602187),
    ("Venezuela (Bolivarian Republic of)", 552695),
    ("Egypt", 516023),
    ("Qatar", 514524),
    ("Libya", 507269),
    ("Ethiopia", 501060),
    ("Réunion", 494595),
    ("Honduras", 474587),
    ("Armenia", 449650),
    ("Bosnia and Herzegovina", 403155),
    ("Oman", 399449),
    ("Luxembourg", 384378),
    ("Zambia", 349287)
]

# Extract the values for clustering
values = [value for state, value in data]

# Create a numpy array from the values
X = np.array(values).reshape(-1, 1)

# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=5, random_state=0)

# Set the default value of n_init
KMeans.default_n_init = 10

# Initialize the model without explicitly setting n_init
# kmeans = KMeans()

# Fit the model to the data
kmeans.fit(X)
# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_

# Add the cluster assignments to your data
for i, (state, value) in enumerate(data):
    data[i] = (state, value, cluster_assignments[i])

# Print the results
for state, value, cluster in data:
    print(f"Country: {state}, Value: {value}, Cluster: {cluster}")
