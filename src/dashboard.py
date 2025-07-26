import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined results
data = pd.read_csv('../data/combined_predictions.csv')

# Plot distribution of predictions
plt.figure(figsize=(10, 6))
sns.countplot(x='RF_Prediction', data=data)
plt.title('Random Forest Prediction Distribution')
plt.xlabel('Prediction (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Count')
plt.savefig('dashboard_rf_distribution.png')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features and Predictions')
plt.savefig('dashboard_correlation_heatmap.png')
plt.close()

print("Dashboard visualizations saved as PNGs for reporting and presentation.")