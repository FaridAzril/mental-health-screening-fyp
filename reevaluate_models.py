"""
Re-evaluate saved models with pure argmax (no dynamic thresholding).
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

print("Loading data...")
dp = EDAICDataProcessor(BASE_PATH / "data/edaic", sequence_length=300)
splits = dp.load_split_files()
sev, _ = dp.load_detailed_labels()
au_files = dp.get_participant_files()
datasets = dp.create_datasets(splits, sev, au_files)
processed_data = dp.load_features_and_labels(datasets)

# Labels already correct from data_processor - no random mapping

# Balanced temporal windows
def create_temporal_windows(X_data, y_data, max_frames=600):
    X_windows, y_windows = [], []
    for seq, label in zip(X_data, y_data):
        seq_len = seq.shape[0]
        if seq_len >= 300:
            X_windows.append(seq[:300]); y_windows.append(label)
        else:
            w = np.vstack([seq, np.zeros((300 - seq_len, seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
        if seq_len >= 450:
            X_windows.append(seq[150:450]); y_windows.append(label)
        elif seq_len > 150:
            w = seq[150:]
            if len(w) < 300:
                w = np.vstack([w, np.zeros((300 - len(w), seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
        if seq_len >= 600:
            X_windows.append(seq[300:600]); y_windows.append(label)
        elif seq_len > 300:
            w = seq[300:]
            if len(w) < 300:
                w = np.vstack([w, np.zeros((300 - len(w), seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)

X_train_t, y_train_t = create_temporal_windows(processed_data['train']['X'], processed_data['train']['y'])
processed_data['train']['X'] = X_train_t
processed_data['train']['y'] = y_train_t

for sn in ['dev', 'test']:
    if len(processed_data[sn]['y']) > 0:
        X_s, y_s = create_temporal_windows(processed_data[sn]['X'], processed_data[sn]['y'], max_frames=300)
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

# Load models
print("\nLoading models...")
models = []
for i in range(1, 4):
    m = tf.keras.models.load_model(f'web_app/models/ensemble_{i}.h5', compile=False)
    models.append(m)

# Evaluate on BOTH dev and test
for eval_name in ['dev', 'test']:
    X_ev = processed_data[eval_name]['X']
    y_ev = processed_data[eval_name]['y']
    
    preds = [m.predict(X_ev, verbose=0) for m in models]
    
    # Weighted ensemble + pure argmax
    weights = [1.0, 1.0, 1.5]
    weighted = [p * w for p, w in zip(preds, weights)]
    ensemble_proba = np.mean(weighted, axis=0)
    ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1, keepdims=True)
    y_pred = np.argmax(ensemble_proba, axis=1)
    
    acc = np.mean(y_pred == y_ev)
    f1 = f1_score(y_ev, y_pred, average='macro')
    
    print(f"\n{'='*60}")
    print(f"{eval_name.upper()} SET - Pure Argmax (no dynamic thresholding)")
    print(f"{'='*60}")
    print(f"Samples: {len(y_ev)}, Distribution: {np.bincount(y_ev)}")
    print(f"Predicted distribution: {np.bincount(y_pred, minlength=3)}")
    print(f"Accuracy: {acc:.1%}")
    print(f"Macro F1: {f1:.1%}")
    print(f"\n{classification_report(y_ev, y_pred, target_names=['Low/None', 'Moderate', 'High'], zero_division=0)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_ev, y_pred)}")
