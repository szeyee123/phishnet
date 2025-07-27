import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('../data/newdataset.csv')

# Identify useful columns
all_columns = list(data.columns)
print("All columns:", all_columns)

# Remove high-cardinality or raw string columns
drop_cols = ['FILENAME', 'URL', 'Domain', 'TLD','Title'] 
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

# If 'label' exists, store it separately for supervised learning
if 'label' in data.columns:
    labels = data['label']
    data = data.drop(columns=['label'])
else:
    labels = None

# Handle missing values
data = data.fillna(0)

# Select only numeric columns to avoid errors (Autoencoder)
numeric_data = data.select_dtypes(include=['number'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data.values)

# # Correlation heatmap (visualization)
# plt.figure(figsize=(12, 10))
# sns.heatmap(data.corr(), cmap='coolwarm')
# plt.title('Feature Correlation Heatmap')
# plt.show()

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save preprocessed data for use in model scripts
processed_df = pd.DataFrame(data_scaled, columns=data.columns)
if labels is not None:
    processed_df['label'] = labels

processed_df.to_csv('../data/preprocessed_dataset.csv', index=False)

print("Preprocessing completed and saved to preprocessed_dataset.csv")
