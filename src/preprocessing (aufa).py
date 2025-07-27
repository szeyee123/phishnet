import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # select only numeric columns to avoid errors
    numeric_df = df.select_dtypes(include=['number'])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df.values)

    return scaled_data, numeric_df.columns

