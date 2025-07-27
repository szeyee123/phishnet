import pandas as pd
import joblib


# Load preprocessed data
data = pd.read_csv('../data/preprocessed_dataset.csv')
X = data.drop(['label'], axis=1)
y = data['label']

# Load models
rf_model = joblib.load('../models/rf_model.joblib')

# Random Forest Prediction
rf_preds = rf_model.predict(X)
rf_probs = rf_model.predict_proba(X)[:, 1]

# K-Means Prediction (cluster 0 = legitimate, 1 = phishing)


# # Autoencoder Prediction
# y_pred = anomalies  # predicted anomalies from model

# Combine results
combined_df = pd.DataFrame({
    'URL': data.get('URL', pd.Series(['N/A'] * len(data))),
    'RF_Prediction': rf_preds,
    'RF_Probability': rf_probs,
    'Label': y,
    # 'KMeans_Cluster': kmeans_labels,
    # 'AE_Anomaly': (recon_loss > threshold).astype(int)
})

# Save for dashboard
combined_df.to_csv('../data/combined_predictions.csv', index=False)
print(f"Combined predictions saved to data/combined_predictions.csv.")