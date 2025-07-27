import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ✅ Step 1: Load data
csv_path = "../data/newdataset.csv"
df = pd.read_csv(csv_path)

# ✅ Step 2: Feature selection
feature_columns = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'IsHTTPS'
]
features = df[feature_columns]

# ✅ Step 3: Standardize features
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# ✅ Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

# ✅ Step 5: Calculate average feature values per cluster
cluster_averages = df.groupby('cluster')[feature_columns].mean().T
cluster_averages.columns = [f"Cluster {i}" for i in cluster_averages.columns]

# ✅ Step 6: Plot cluster centroids (Bar plot)
# The following line is optional and can be removed:
# plt.figure(figsize=(14, 8))  # Not needed since plot() creates a figure

cluster_averages.plot(kind='bar', figsize=(14, 8), colormap='Set2')
plt.title("🔍 Average Feature Values per Cluster (Centroids)")
plt.ylabel("Mean Feature Value")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.legend(title='Cluster', loc='upper right')
plt.show()