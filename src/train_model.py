"""
train.py — train and evaluate the RF signal classifier.

Key improvements:
  1. Stratified 5-fold cross-validation instead of a single random split.
     This gives a reliable accuracy range rather than one lucky/unlucky number.
  2. Confusion matrix to see WHICH classes are being confused.
  3. Feature importance plot — tells you which features actually matter.
  4. StandardScaler — good habit even though RandomForest doesn't need it;
     makes the pipeline easy to swap to SVM or logistic regression later.
  5. Final model trained on all data and saved for inference.
"""

import numpy as np
import os
import joblib

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (accuracy_score, classification_report,
                                       confusion_matrix)
from features import FEATURE_NAMES

DATA_DIR  = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
print(f"Loaded dataset: X={X.shape}, y={y.shape}")
print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}\n")

# ── Pipeline ──────────────────────────────────────────────────────────────────
# StandardScaler first (no information leak — fitted inside each fold by CV)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
                   n_estimators=200,
                   max_depth=10,          # limit depth → less overfitting
                   min_samples_leaf=3,    # require ≥3 samples in leaves
                   random_state=42,
               )),
])

# ── Stratified 5-fold CV ──────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print("5-fold CV accuracy (each fold):", np.round(cv_scores, 4))
print(f"Mean ± std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# Out-of-fold predictions for a confusion matrix on the full dataset
y_pred_oof = cross_val_predict(pipe, X, y, cv=cv)

print("Classification report (out-of-fold):")
print(classification_report(y, y_pred_oof, digits=3))

cm = confusion_matrix(y, y_pred_oof, labels=["signal", "noise", "multi"])
print("Confusion matrix  (rows=true, cols=predicted):")
print("              signal  noise  multi")
for label, row in zip(["signal", "noise ", "multi "], cm):
    print(f"  {label}  ", "  ".join(f"{v:5d}" for v in row))

# ── Feature importance (train once on full data to inspect) ──────────────────
pipe.fit(X, y)
importances = pipe["clf"].feature_importances_

print("\nFeature importances (higher = more useful):")
ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True)
for name, imp in ranked:
    bar = "█" * int(imp * 50)
    print(f"  {name:<22s} {imp:.4f}  {bar}")

# ── Save final model ──────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(pipe, model_path)
print(f"\nFinal model saved to '{model_path}'")

# ── Sanity check: what does accuracy=1.0 look like vs our result? ─────────────
print("\n── Overfitting check ──")
y_train_pred = pipe.predict(X)
train_acc    = accuracy_score(y, y_train_pred)
print(f"Training accuracy (resubstitution): {train_acc:.4f}")
print(f"CV accuracy (generalisation)       : {cv_scores.mean():.4f}")
gap = train_acc - cv_scores.mean()
if gap > 0.05:
    print(f"  ⚠ Gap of {gap:.3f} suggests overfitting — consider reducing max_depth")
else:
    print(f"  ✓ Gap of {gap:.3f} is acceptable")