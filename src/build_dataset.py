"""
build_dataset.py — read .npy files, extract features, save X.npy / y.npy.

No major logic changes here; the improvements come from simulate.py
(harder signals) and features.py (richer feature set).
"""

import numpy as np
import os
from features import extract_features, FEATURE_NAMES

DATA_DIR   = "../data"
SKIP_FILES = {"X.npy", "y.npy"}


def build():
    X, y = [], []
    files_processed = 0

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".npy") or fname in SKIP_FILES:
            continue

        path = os.path.join(DATA_DIR, fname)
        try:
            data = np.load(path)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        feats = extract_features(data)
        X.append(feats)

        if   "signal" in fname: y.append("signal")
        elif "noise"  in fname: y.append("noise")
        elif "multi"  in fname: y.append("multi")
        else:
            print(f"  [SKIP] {fname}: unrecognised label prefix")
            X.pop()
            continue

        files_processed += 1

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)

    print(f"\nDataset created from {files_processed} files.")
    print(f"  X shape : {X.shape}   ({X.shape[1]} features per sample)")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"\nFeatures: {FEATURE_NAMES}")
    return X, y


if __name__ == "__main__":
    build()