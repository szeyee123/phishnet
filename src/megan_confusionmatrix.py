import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# STEP 1: Load the dataset
csv_path = "../data/newdataset.csv"
df = pd.read_csv(csv_path)

# STEP 2: Select features for clustering
feature_columns = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL', 'IsHTTPS'
]

# STEP 3: Standardize features
features = df[feature_columns]
scaled = StandardScaler().fit_transform(features)

# STEP 4: Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

# STEP 5: Match clusters to actual labels
if 'label' in df.columns:
    cluster_label_counts = df.groupby('cluster')['label'].value_counts().unstack(fill_value=0)
    cluster_to_label = cluster_label_counts.idxmax(axis=1).to_dict()
    df['predicted_label'] = df['cluster'].map(cluster_to_label)
else:
    raise ValueError("Dataset must contain a 'label' column for evaluation.")

# STEP 6: Evaluate performance
y_true = df['label']  # 0 = legitimate, 1 = phishing
y_pred = df['predicted_label']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)

# STEP 7: Output results in a table
results_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Phishing)', 'Recall (Phishing)'],
    'Score (%)': [round(accuracy * 100, 2), round(precision * 100, 2), round(recall * 100, 2)]
})

print("\n🔍 K-Means Clustering Performance Summary:")
print(results_table.to_string(index=False))

# STEP 8: Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate (0)', 'Phishing (1)'])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix - KMeans Clustering")
plt.show()