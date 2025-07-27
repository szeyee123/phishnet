import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore

# Load dataset
df = pd.read_csv('../data/newdataset.csv')

# Feature columns used for clustering
feature_columns = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'IsHTTPS'
]
X = df[feature_columns]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Distance from each point to its cluster centroid
distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[df['cluster']], axis=1)

# Outlier detection (Z-score threshold)
z_scores = zscore(distances)
df['outlier'] = (np.abs(z_scores) > 2)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Assign PCA results to dataframe
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plotting
plt.figure(figsize=(16, 9))

# Plot clusters
for cluster_label, color in zip([0, 1], ['skyblue', 'orange']):
    cluster_data = df[df['cluster'] == cluster_label]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                label=f'Cluster {cluster_label}', alpha=0.5, s=10, color=color)

# Plot outliers
outliers = df[df['outlier']]
plt.scatter(outliers['PCA1'], outliers['PCA2'], edgecolor='red', facecolor='none', s=60, label='Outliers', linewidth=1.5)

# Styling
plt.title('K-Means Clustering Visualized with PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

