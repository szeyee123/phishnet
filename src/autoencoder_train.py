# from aufa_preprocessing import load_and_preprocess
from aufa_autoencoder_model import build_autoencoder
from aufa_evaluate import plot_loss, detect_anomalies
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# # load and clean the data
# # im using my phishing dataset here
# data, columns = load_and_preprocess("../data/newdataset.csv")

import pandas as pd
# Load preprocessed dataset
data = pd.read_csv('../data/preprocessed_dataset.csv')

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
df_results.to_csv("../data/autoencoder_results.csv", index=False)

# to load the dataset again for getting the true labels needed for evaluation
anomalies, mse = detect_anomalies(model, data)

import pandas as pd
df_labels = pd.read_csv("../data/newdataset.csv")
y_test = df_labels['label'].values  # true labels

y_pred = anomalies  # predicted anomalies from model

# compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# extract TN, FP, FN, TP
tn, fp, fn, tp = conf_matrix.ravel()

print("Confusion Matrix (Detailed):")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print("\nFull Confusion Matrix:")
print(conf_matrix)

# now to compute and print metrics!
# visualise the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Legit", "Phishing"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Autoencoder")
plt.show()
