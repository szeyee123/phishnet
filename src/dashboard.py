import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load combined results
data = pd.read_csv('../data/combined_predictions.csv')

# Plot distribution of RF predictions
plt.figure(figsize=(10, 6))
sns.countplot(x='RF_Prediction', data=data)
plt.title('Random Forest Prediction Distribution')
plt.xlabel('Prediction (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Count')
plt.savefig('dashboard_rf_distribution.png')
plt.close()

# Get true labels and predictions
y_true = data['Label']
y_pred = data['RF_Prediction']

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate (0)', 'Phishing (1)'])
disp.plot(cmap='Reds')
plt.title("Confusion Matrix - Random Forest")
plt.savefig('dashboard_rf_confusion_matrix.png')
plt.close()

# # Correlation Heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap of Features and Predictions')
# plt.savefig('dashboard_correlation_heatmap.png')
# plt.close()


print("Dashboard visualizations saved as PNGs for reporting and presentation.")