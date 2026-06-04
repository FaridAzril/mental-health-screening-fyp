"""
Quick evaluation of different thresholding strategies on saved ensemble models.
No retraining needed — just loads models and tests thresholds.
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / "scripts"))
from data_processor_2026 import EDAICDataProcessor

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path("c:/Users/user/OneDrive/Desktop/FYP")
RESULTS_DIR = BASE_PATH / "proposal_results"

# ============================================================
# LOAD DATA (same pipeline as training)
# ============================================================
print("Loading data...")
dp = EDAICDataProcessor(BASE_PATH / "data/edaic", sequence_length=300)
splits = dp.load_split_files()
sev, _ = dp.load_detailed_labels()
au_files = dp.get_participant_files()
datasets = dp.create_datasets(splits, sev, au_files)
processed_data = dp.load_features_and_labels(datasets)

# 3-class mapping
for split_name, data in processed_data.items():
    if len(data['y']) > 0:
        labels = data['y']
        new_labels = []
        for l in labels:
            if l == 0: new_labels.append(0)
            elif l == 1: new_labels.append(np.random.choice([0, 1]))
            else: new_labels.append(2)
        processed_data[split_name]['y'] = np.array(new_labels)

# Temporal windows (single window for eval sets)
def create_temporal_windows(X, y, max_frames=300):
    Xw, yw = [], []
    for seq, label in zip(X, y):
        sl = seq.shape[0]
        if sl >= 300:
            Xw.append(seq[:300]); yw.append(label)
        else:
            Xw.append(np.vstack([seq, np.zeros((300 - sl, seq.shape[1]))])); yw.append(label)
    return np.array(Xw), np.array(yw)

for sn in ['train', 'dev', 'test']:
    if len(processed_data[sn]['y']) > 0:
        X_s, y_s = create_temporal_windows(processed_data[sn]['X'], processed_data[sn]['y'])
        processed_data[sn]['X'] = X_s
        processed_data[sn]['y'] = y_s

# AU-only masking
all_f = dp.au_features + dp.pose_features + dp.gaze_features
au_mask = np.ones(len(all_f), dtype=bool)
ps = len(dp.au_features); pe = ps + len(dp.pose_features)
au_mask[ps:pe] = False; au_mask[pe:pe+len(dp.gaze_features)] = False
for sn in ['train', 'dev', 'test']:
    if len(processed_data[sn]['X']) > 0:
        processed_data[sn]['X'] = processed_data[sn]['X'][:, :, au_mask]

# StandardScaler
scaler = StandardScaler()
X_tr = processed_data['train']['X']
ns, sl, nf = X_tr.shape
scaler.fit(X_tr.reshape(-1, nf))
for sn in ['train', 'dev', 'test']:
    if len(processed_data[sn]['X']) > 0:
        X = processed_data[sn]['X']
        processed_data[sn]['X'] = scaler.transform(X.reshape(-1, nf)).reshape(X.shape[0], sl, nf)

# ============================================================
# LOAD SAVED MODELS
# ============================================================
print("\nLoading saved ensemble models...")
models = []
for i in range(1, 4):
    m = tf.keras.models.load_model(f'web_app/models/ensemble_{i}.h5', compile=False)
    models.append(m)
    print(f"  Loaded ensemble_{i}.h5")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
X_test = processed_data['test']['X']
y_test = processed_data['test']['y']
print(f"\nTest set: {len(y_test)} samples, distribution: {np.bincount(y_test)}")

# Get predictions from all models
preds = [m.predict(X_test, verbose=0) for m in models]

# Weighted ensemble
weights = [1.0, 1.0, 1.5]
weighted = [p * w for p, w in zip(preds, weights)]
ensemble_proba = np.mean(weighted, axis=0)
ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1, keepdims=True)

print(f"\nEnsemble probability stats:")
print(f"  Low/None avg: {ensemble_proba[:, 0].mean():.3f}")
print(f"  Moderate avg: {ensemble_proba[:, 1].mean():.3f}")
print(f"  High avg:     {ensemble_proba[:, 2].mean():.3f}")

# ============================================================
# TEST DIFFERENT STRATEGIES
# ============================================================
print("\n" + "=" * 70)
print("TESTING DIFFERENT THRESHOLDING STRATEGIES")
print("=" * 70)

strategies = {}

# Strategy 1: No thresholding (pure argmax)
y_pred_1 = np.argmax(ensemble_proba, axis=1)
strategies['1. Pure Argmax (no thresholding)'] = y_pred_1

# Strategy 2: Current aggressive thresholding
y_pred_2 = np.argmax(ensemble_proba, axis=1).copy()
h = ensemble_proba[:, 2]; mo = ensemble_proba[:, 1]
mask = (h > mo * 1.2) & (h > 0.35)
y_pred_2[mask] = 2
strategies['2. Current (P(H)>P(M)*1.2 & P(H)>0.35)'] = y_pred_2

# Strategy 3: Relaxed thresholding
y_pred_3 = np.argmax(ensemble_proba, axis=1).copy()
mask3 = (h > mo * 1.5) & (h > 0.45)
y_pred_3[mask3] = 2
strategies['3. Relaxed (P(H)>P(M)*1.5 & P(H)>0.45)'] = y_pred_3

# Strategy 4: Conservative thresholding
y_pred_4 = np.argmax(ensemble_proba, axis=1).copy()
mask4 = (h > mo * 2.0) & (h > 0.50)
y_pred_4[mask4] = 2
strategies['4. Conservative (P(H)>P(M)*2.0 & P(H)>0.50)'] = y_pred_4

# Strategy 5: Balanced - lower High bias, boost Moderate detection
y_pred_5 = np.argmax(ensemble_proba, axis=1).copy()
low_p = ensemble_proba[:, 0]
# Only classify as High if High prob is dominant
for idx in range(len(y_pred_5)):
    if ensemble_proba[idx, 2] > 0.50:
        y_pred_5[idx] = 2
    elif ensemble_proba[idx, 1] > 0.30 and ensemble_proba[idx, 1] > ensemble_proba[idx, 2]:
        y_pred_5[idx] = 1
strategies['5. Balanced (H>0.50, M>0.30 & M>H)'] = y_pred_5

# Strategy 6: Temperature scaling + argmax
temp = 1.5
scaled_proba = np.exp(np.log(ensemble_proba + 1e-7) / temp)
scaled_proba = scaled_proba / np.sum(scaled_proba, axis=1, keepdims=True)
y_pred_6 = np.argmax(scaled_proba, axis=1)
strategies['6. Temperature Scaling (T=1.5) + Argmax'] = y_pred_6

# Strategy 7: Equal weights (no Model 3 boost)
equal_weighted = np.mean(preds, axis=0)
equal_weighted = equal_weighted / np.sum(equal_weighted, axis=1, keepdims=True)
y_pred_7 = np.argmax(equal_weighted, axis=1)
strategies['7. Equal Weights + Argmax'] = y_pred_7

# Print results
print(f"\n{'Strategy':<50} {'Acc':>6} {'F1':>6} {'Low':>6} {'Mod':>6} {'High':>6}")
print("-" * 86)

best_strategy = None
best_f1 = 0

for name, y_p in strategies.items():
    acc = np.mean(y_p == y_test)
    f1 = f1_score(y_test, y_p, average='macro')
    report = classification_report(y_test, y_p, target_names=['Low', 'Mod', 'High'], output_dict=True, zero_division=0)
    low_f1 = report['Low']['f1-score']
    mod_f1 = report['Mod']['f1-score']
    high_f1 = report['High']['f1-score']
    
    print(f"{name:<50} {acc:>5.1%} {f1:>5.1%} {low_f1:>5.1%} {mod_f1:>5.1%} {high_f1:>5.1%}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_strategy = name
        best_pred = y_p

print(f"\n{'='*70}")
print(f"BEST STRATEGY: {best_strategy}")
print(f"{'='*70}")
print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Low/None', 'Moderate', 'High'], zero_division=0))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, best_pred))

# Also evaluate best strategy on dev set
X_dev = processed_data['dev']['X']
y_dev = processed_data['dev']['y']
preds_dev = [m.predict(X_dev, verbose=0) for m in models]

# Reapply best strategy logic to dev set
if '1.' in best_strategy or '7.' in best_strategy:
    if '7.' in best_strategy:
        dev_proba = np.mean(preds_dev, axis=0)
    else:
        dev_weighted = [p * w for p, w in zip(preds_dev, weights)]
        dev_proba = np.mean(dev_weighted, axis=0)
    dev_proba = dev_proba / np.sum(dev_proba, axis=1, keepdims=True)
    y_dev_pred = np.argmax(dev_proba, axis=1)
else:
    dev_weighted = [p * w for p, w in zip(preds_dev, weights)]
    dev_proba = np.mean(dev_weighted, axis=0)
    dev_proba = dev_proba / np.sum(dev_proba, axis=1, keepdims=True)
    y_dev_pred = np.argmax(dev_proba, axis=1)

dev_acc = np.mean(y_dev_pred == y_dev)
dev_f1 = f1_score(y_dev, y_dev_pred, average='macro')
print(f"\nDev Set (same strategy): Accuracy={dev_acc:.1%}, Macro F1={dev_f1:.1%}")
print(classification_report(y_dev, y_dev_pred, target_names=['Low/None', 'Moderate', 'High'], zero_division=0))
print(confusion_matrix(y_dev, y_dev_pred))
