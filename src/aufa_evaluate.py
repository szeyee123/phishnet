import matplotlib.pyplot as plt
import numpy as np

# plots the training loss after the model is trained
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# it compares original vs reconstructed to detect anomaly based on MSE
def detect_anomalies(model, data, threshold=0.01):
    # reconstruct the data using the autoencoder
    reconstructions = model.predict(data)

    # calculate the mse for each row
    mse = np.mean((reconstructions - data) ** 2, axis=1)

    # flag if there is anything above the threshold as anomaly
    anomalies = mse > threshold

    return anomalies, mse
