import numpy as np
import os
from features import extract_features

X = []
y = []

data_path = "../data"

for file in os.listdir(data_path):
    if not file.endswith(".npy"):
        continue

    if file in ["X.npy", "y.npy"]:
        continue

    try:
        data = np.load(os.path.join(data_path, file))
    except:
        print(f"Skipping bad file: {file}")
        continue

    feats = extract_features(data)
    X.append(feats)

    if "signal" in file:
        y.append("signal")
    elif "noise" in file:
        y.append("noise")
    elif "multi" in file:
        y.append("multi")

X = np.array(X)
y = np.array(y)

np.save("../data/X.npy", X)
np.save("../data/y.npy", y)

print("Dataset created:", X.shape)