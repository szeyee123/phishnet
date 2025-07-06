import pandas as pd
from urllib.parse import urlparse

def has_ip(url):
    import re
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    return int(bool(ip_pattern.search(url)))

def extract_features(url):
    parsed = urlparse(url)
    features = {
        "url_length": len(url),
        "num_special_chars": sum(url.count(c) for c in ['@', '?', '-', '_', '=', '&']),
        "has_ip": has_ip(url),
        "has_https": int(parsed.scheme == "https")
    }
    return features

# Load dataset
data = pd.read_csv("dataset.csv")

# Extract features
features_list = []
for idx, row in data.iterrows():
    url = row["url"]
    label = row["status"]
    features = extract_features(url)
    features["label"] = label
    features_list.append(features)

features_df = pd.DataFrame(features_list)

# Save extracted features for training
features_df.to_csv("phishnet_features.csv", index=False)

print("Feature extraction completed. Saved as phishnet_features.csv.")
