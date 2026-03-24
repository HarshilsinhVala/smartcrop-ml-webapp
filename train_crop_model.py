"""
============================================================
  SmartCrop — Crop Recommendation Model Retraining Script
  Model: RandomForestClassifier
  Features: N, P, K, temperature, humidity, ph, rainfall
============================================================

HOW TO USE:
  1. Place your Crop_recommendation.csv in the same folder as this script
  2. Run: python train_crop_model.py
  3. New models saved as: crop_model.pkl, label_encoder.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────
CSV_PATH     = "Crop_recommendation.csv"   
OUTPUT_DIR   = "models"                     
TEST_SIZE    = 0.2
RANDOM_STATE = 42
TUNE_PARAMS  = True   # Set False to skip GridSearch and train faster

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COL    = "label"
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  SmartCrop — Crop Model Retraining")
print("=" * 60)

# ─── Step 1: Load Data ────────────────────────────────────
print("\n📂 Loading dataset...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"❌ Could not find '{CSV_PATH}'\n"
        f"   Please place your CSV file in: {os.path.abspath('.')}"
    )

df = pd.read_csv(CSV_PATH)
print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")

# Auto-detect label column if different
if LABEL_COL not in df.columns:
    possible = [c for c in df.columns if c.lower() in ["label","crop","class","target"]]
    if possible:
        LABEL_COL = possible[0]
        print(f"   ⚠️  Using '{LABEL_COL}' as label column")
    else:
        raise ValueError(f"❌ Could not find label column. Columns: {list(df.columns)}")

# Auto-detect feature columns if missing
missing_feats = [f for f in FEATURE_COLS if f not in df.columns]
if missing_feats:
    print(f"   ⚠️  Missing feature columns: {missing_feats}")
    FEATURE_COLS = [c for c in df.columns if c != LABEL_COL]
    print(f"   Using all other columns as features: {FEATURE_COLS}")

print(f"\n📊 Dataset Info:")
print(f"   Classes: {sorted(df[LABEL_COL].unique())}")
print(f"   Total classes: {df[LABEL_COL].nunique()}")
print(f"   Samples per class:\n{df[LABEL_COL].value_counts().to_string()}")

# ─── Step 2: Preprocess ───────────────────────────────────
print("\n🔧 Preprocessing...")
X = df[FEATURE_COLS].values
y_raw = df[LABEL_COL].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"   ✅ Encoded {len(le.classes_)} classes")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ─── Step 3: Baseline Model ───────────────────────────────
print("\n🌲 Training baseline RandomForest...")
baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"   Baseline accuracy: {baseline_acc*100:.2f}%")

# ─── Step 4: Hyperparameter Tuning (Optional) ─────────────
best_model = baseline

if TUNE_PARAMS:
    print("\n🔍 Running GridSearch for best hyperparameters...")
    print("   (This may take 2-5 minutes...)")

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features":  ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    tuned_acc  = accuracy_score(y_test, best_model.predict(X_test))

    print(f"\n   Best params: {grid_search.best_params_}")
    print(f"   Tuned accuracy: {tuned_acc*100:.2f}%")
    print(f"   Improvement: +{(tuned_acc - baseline_acc)*100:.2f}%")

# ─── Step 5: Cross Validation ─────────────────────────────
print("\n📈 Cross-validation (5-fold)...")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
print(f"   CV Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"   Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ─── Step 6: Final Evaluation ─────────────────────────────
print("\n📋 Classification Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

final_acc = accuracy_score(y_test, y_pred)
print(f"   ✅ Final Test Accuracy: {final_acc*100:.2f}%")

# ─── Step 7: Feature Importance ───────────────────────────
print("\n🔑 Feature Importance:")
importances = best_model.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"   {feat:<15} {bar} {imp*100:.1f}%")

# ─── Step 8: Save Models ──────────────────────────────────
print("\n💾 Saving models...")

crop_model_path = os.path.join(OUTPUT_DIR, "crop_model.pkl")
label_enc_path  = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

with open(crop_model_path, "wb") as f:
    pickle.dump(best_model, f)

with open(label_enc_path, "wb") as f:
    pickle.dump(le, f)

print(f"   ✅ crop_model.pkl    → {os.path.abspath(crop_model_path)}")
print(f"   ✅ label_encoder.pkl → {os.path.abspath(label_enc_path)}")

print("\n" + "=" * 60)
print(f"  🎉 Training complete! Accuracy: {final_acc*100:.2f}%")
print("  Copy the files from 'models/' to your project's models/ folder")
print("=" * 60)
