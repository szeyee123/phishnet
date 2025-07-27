from preprocessing import load_and_preprocess
from autoencoder_model import build_autoencoder
from evaluate import plot_loss, detect_anomalies
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# load and clean the data
# im using my phishing dataset here
data, columns = load_and_preprocess("newdataset.csv")

# build and train the autoencoder model
# it is trained to reconstruct the input
model = build_autoencoder(data.shape[1])
history = model.fit(data, data, epochs=20, batch_size=32, verbose=1)

# visualise the training loss so I can check if it's improving
plot_loss(history)

# after training, detect which samples are considered as anomalies
anomalies, mse = detect_anomalies(model, data)

# store the results for reporting
# im creating a data frame to show the MSE and whether it's an anomaly
import pandas as pd
df_results = pd.DataFrame({'MSE': mse, 'Anomaly': anomalies})

# just printing the first few results for a quick check
print(df_results.head(10))

# saving the full result to a CSV so I can use it in my report or charts
df_results.to_csv("autoencoder_results.csv", index=False)

# to load the dataset again for getting the true labels needed for evaluation
anomalies, mse = detect_anomalies(model, data)

import pandas as pd
df_labels = pd.read_csv("newdataset.csv")
y_test = df_labels['label'].values  # true labels

y_pred = anomalies  # predicted anomalies from model

# now to compute and print metrics!
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
