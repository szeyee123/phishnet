from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# builds the autoencoder model structure
def build_autoencoder(input_dim):
    # im using a basic 3-layer autoencoder (can be deeper if needed)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),  # encoding layer
        Dense(8, activation='relu'),                              # bottleneck
        Dense(16, activation='relu'),                             # decoding layer
        Dense(input_dim, activation='sigmoid')                    # output layer
    ])

    # compile the model with mean squared error loss
    model.compile(optimizer='adam', loss='mse')
    return model
