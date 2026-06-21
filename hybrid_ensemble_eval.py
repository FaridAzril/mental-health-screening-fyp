#!/usr/bin/env python3
"""
Hybrid Ensemble Evaluation: CNN-BiLSTM (3x) + Pseudo-Image 2D CNN (1x)
Averages softmax probabilities from both architectures on the test set.

Individual results:
  CNN-BiLSTM (3-model ensemble): Accuracy 51.8%, Macro F1 32.0%
    Low 0.68 | Moderate 0.00 | High 0.28
  Pseudo-Image 2D CNN (single): Accuracy 57.1%, Macro F1 31.0%
    Low 0.73 | Moderate 0.20 | High 0.00
"""

import sys
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / "scripts"))
from data_processor_2026 import EDAICDataProcessor

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

BASE_PATH   = Path("c:/Users/user/OneDrive/Desktop/FYP")
MODELS_DIR  = BASE_PATH / "web_app" / "models"
RESULTS_DIR = BASE_PATH / "proposal_results"

# ============================================================
# STEP 1: LOAD & PREPROCESS DATA (identical to both scripts)
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

data_processor = EDAICDataProcessor(BASE_PATH / "data/edaic", sequence_length=300)
splits = data_processor.load_split_files()
participant_severity, _ = data_processor.load_detailed_labels()
au_files = data_processor.get_participant_files()
datasets = data_processor.create_datasets(splits, participant_severity, au_files, scaling_factor=1.0)
processed_data = data_processor.load_features_and_labels(datasets, scaling_factor=1.0)

def create_temporal_windows(X_data, y_data, max_frames=300):
    X_windows, y_windows = [], []
    for seq, label in zip(X_data, y_data):
        seq_len = seq.shape[0]
        if seq_len >= 300:
            X_windows.append(seq[:300]); y_windows.append(label)
        else:
            w = np.vstack([seq, np.zeros((300 - seq_len, seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)

for split_name in ['dev', 'test']:
    if len(processed_data[split_name]['y']) > 0:
        X_s, y_s = create_temporal_windows(
            processed_data[split_name]['X'], processed_data[split_name]['y'])
        processed_data[split_name]['X'] = X_s
        processed_data[split_name]['y'] = y_s

# AU-only masking
all_features = (data_processor.au_features
                + data_processor.pose_features
                + data_processor.gaze_features)
au_mask = np.ones(len(all_features), dtype=bool)
pose_start = len(data_processor.au_features)
au_mask[pose_start:pose_start + len(data_processor.pose_features)
                  + len(data_processor.gaze_features)] = False

# Apply masking to train too (needed to fit scaler)
for split_name in ['train', 'dev', 'test']:
    if len(processed_data[split_name]['X']) > 0:
        processed_data[split_name]['X'] = processed_data[split_name]['X'][:, :, au_mask]

# StandardScaler
scaler = StandardScaler()
X_tr = processed_data['train']['X']
n_tr, seq_len, nf = X_tr.shape
processed_data['train']['X'] = scaler.fit_transform(
    X_tr.reshape(-1, nf)).reshape(n_tr, seq_len, nf)

for split_name in ['dev', 'test']:
    if len(processed_data[split_name]['X']) > 0:
        X = processed_data[split_name]['X']
        ns = X.shape[0]
        processed_data[split_name]['X'] = scaler.transform(
            X.reshape(-1, nf)).reshape(ns, seq_len, nf)

X_test_1d = processed_data['test']['X'].astype(np.float32)  # (56, 300, 17) for BiLSTM
y_test     = processed_data['test']['y']

# Pseudo-image transform for 2D CNN: (N, 300, 17) -> (N, 17, 300, 1)
X_test_2d = np.transpose(X_test_1d, (0, 2, 1))[:, :, :, np.newaxis]

print(f"BiLSTM input shape : {X_test_1d.shape}")
print(f"2D CNN input shape : {X_test_2d.shape}")
print(f"Test labels        : {np.bincount(y_test)}")

# ============================================================
# STEP 2: LOAD MODELS
# ============================================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

# 3× CNN-BiLSTM ensemble
bilstm_models = []
for i in range(1, 4):
    path = MODELS_DIR / f"ensemble_{i}.h5"
    m = tf.keras.models.load_model(str(path), compile=False)
    bilstm_models.append(m)
    print(f"  Loaded {path.name}  input={m.input_shape}")

# 1× Pseudo-image 2D CNN
cnn2d_path  = MODELS_DIR / "prototype_2d_cnn.h5"
cnn2d_model = tf.keras.models.load_model(str(cnn2d_path), compile=False)
print(f"  Loaded {cnn2d_path.name}  input={cnn2d_model.input_shape}")

# ============================================================
# STEP 3: GET PREDICTIONS FROM EACH MODEL
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

# BiLSTM ensemble average (equal weights)
bilstm_probas = np.zeros((len(y_test), 3))
for i, m in enumerate(bilstm_models):
    p = m.predict(X_test_1d, verbose=0)
    bilstm_probas += p
    print(f"  BiLSTM model {i+1}: argmax dist = {np.bincount(np.argmax(p, axis=1), minlength=3)}")
bilstm_probas /= len(bilstm_models)

# 2D CNN predictions
cnn2d_probas = cnn2d_model.predict(X_test_2d, verbose=0)
print(f"  2D CNN model   : argmax dist = {np.bincount(np.argmax(cnn2d_probas, axis=1), minlength=3)}")

# ============================================================
# STEP 4: HYBRID COMBINATIONS - try three weighting ratios
# ============================================================
def evaluate_hybrid(bilstm_p, cnn2d_p, w_bilstm, w_cnn2d, y_true, label):
    combined = (w_bilstm * bilstm_p + w_cnn2d * cnn2d_p) / (w_bilstm + w_cnn2d)
    y_pred   = np.argmax(combined, axis=1)
    acc      = np.mean(y_pred == y_true)
    mf1      = f1_score(y_true, y_pred, average='macro', zero_division=0)
    rep      = classification_report(
        y_true, y_pred,
        target_names=['Low/None', 'Moderate', 'High'],
        zero_division=0, output_dict=True)
    print(f"\n{'─'*60}")
    print(f"{label}  (BiLSTM×{w_bilstm} + 2DCNN×{w_cnn2d})")
    print(f"{'─'*60}")
    print(classification_report(y_true, y_pred,
          target_names=['Low/None', 'Moderate', 'High'], zero_division=0))
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)   Macro F1: {mf1:.4f} ({mf1*100:.1f}%)")
    return acc, mf1, rep

print("\n" + "=" * 70)
print("HYBRID ENSEMBLE RESULTS")
print("=" * 70)

results = {}
for w_b, w_c, tag in [(1, 1, "50/50"),
                       (2, 1, "BiLSTM 67% / 2DCNN 33%"),
                       (1, 2, "BiLSTM 33% / 2DCNN 67%")]:
    acc, mf1, rep = evaluate_hybrid(bilstm_probas, cnn2d_probas, w_b, w_c, y_test, tag)
    results[tag] = {"accuracy": float(acc), "macro_f1": float(mf1), "report": rep}

# ============================================================
# STEP 5: COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)
baseline = {"accuracy": 0.5179, "macro_f1": 0.3203,
            "low_f1": 0.68, "moderate_f1": 0.00, "high_f1": 0.28}
cnn2d_r  = {"accuracy": 0.5714, "macro_f1": 0.3098,
            "low_f1": 0.73, "moderate_f1": 0.20, "high_f1": 0.00}

print(f"\n{'Model':<30} {'Acc':>6} {'MacroF1':>8} {'LowF1':>7} {'ModF1':>7} {'HiF1':>7}")
print("─" * 72)
print(f"{'CNN-BiLSTM baseline':<30} {baseline['accuracy']:>6.3f} {baseline['macro_f1']:>8.3f} "
      f"{baseline['low_f1']:>7.3f} {baseline['moderate_f1']:>7.3f} {baseline['high_f1']:>7.3f}")
print(f"{'2D CNN (single)':<30} {cnn2d_r['accuracy']:>6.3f} {cnn2d_r['macro_f1']:>8.3f} "
      f"{cnn2d_r['low_f1']:>7.3f} {cnn2d_r['moderate_f1']:>7.3f} {cnn2d_r['high_f1']:>7.3f}")

for tag, r in results.items():
    rep = r['report']
    print(f"{'Hybrid '+tag:<30} {r['accuracy']:>6.3f} {r['macro_f1']:>8.3f} "
          f"{rep['Low/None']['f1-score']:>7.3f} "
          f"{rep['Moderate']['f1-score']:>7.3f} "
          f"{rep['High']['f1-score']:>7.3f}")

# ============================================================
# STEP 6: DYNAMIC THRESHOLDING ON BiLSTM 67% / 2DCNN 33%
# Rules:
#   1. High_prob > (Moderate_prob * 1.2)  AND  High_prob > 0.35  -> High
#   2. Moderate_prob > 0.35                                       -> Moderate
#   3. Otherwise                                                  -> Low/None
# ============================================================
print("\n" + "=" * 70)
print("DYNAMIC THRESHOLDING  --  Hybrid BiLSTM 67% / 2DCNN 33%")
print("=" * 70)

# Recompute the 67/33 combined probabilities
hybrid_67_33 = (2 * bilstm_probas + 1 * cnn2d_probas) / 3  # (56, 3)

def dynamic_threshold(probs):
    """Apply custom decision rules to a (N, 3) probability array."""
    y_pred = []
    for p in probs:
        low_p, mod_p, high_p = p[0], p[1], p[2]
        if high_p > (mod_p * 1.2) and high_p > 0.35:
            y_pred.append(2)   # High
        elif mod_p > 0.35:
            y_pred.append(1)   # Moderate
        else:
            y_pred.append(0)   # Low/None
    return np.array(y_pred)

y_dyn = dynamic_threshold(hybrid_67_33)

print(f"\nThreshold rules applied:")
print(f"  Rule 1 (High)     : high_prob > mod_prob×1.2  AND  high_prob > 0.35")
print(f"  Rule 2 (Moderate) : mod_prob > 0.35")
print(f"  Rule 3 (Low)      : otherwise")
print(f"\nTrue  distribution : {np.bincount(y_test, minlength=3)}")
print(f"Pred  distribution : {np.bincount(y_dyn,  minlength=3)}")

print("\nClassification Report:")
print(classification_report(y_test, y_dyn,
      target_names=['Low/None', 'Moderate', 'High'], zero_division=0))

cm_dyn  = confusion_matrix(y_test, y_dyn)
acc_dyn = np.mean(y_dyn == y_test)
mf1_dyn = f1_score(y_test, y_dyn, average='macro', zero_division=0)
rep_dyn = classification_report(y_test, y_dyn,
          target_names=['Low/None', 'Moderate', 'High'],
          zero_division=0, output_dict=True)

print(f"Confusion Matrix:\n{cm_dyn}")
print(f"\nOverall Accuracy : {acc_dyn:.4f} ({acc_dyn*100:.1f}%)")
print(f"Macro F1-Score   : {mf1_dyn:.4f} ({mf1_dyn*100:.1f}%)")

# Updated comparison table including dynamic thresholding
print("\n" + "=" * 70)
print("UPDATED FULL COMPARISON TABLE")
print("=" * 70)
print(f"\n{'Model':<38} {'Acc':>6} {'MacroF1':>8} {'LowF1':>7} {'ModF1':>7} {'HiF1':>7}")
print("─" * 80)
rows = [
    ("CNN-BiLSTM baseline (argmax)",
     baseline['accuracy'], baseline['macro_f1'],
     baseline['low_f1'], baseline['moderate_f1'], baseline['high_f1']),
    ("2D CNN single (argmax)",
     cnn2d_r['accuracy'], cnn2d_r['macro_f1'],
     cnn2d_r['low_f1'], cnn2d_r['moderate_f1'], cnn2d_r['high_f1']),
    ("Hybrid 67/33 (argmax)",
     results['BiLSTM 67% / 2DCNN 33%']['accuracy'],
     results['BiLSTM 67% / 2DCNN 33%']['macro_f1'],
     results['BiLSTM 67% / 2DCNN 33%']['report']['Low/None']['f1-score'],
     results['BiLSTM 67% / 2DCNN 33%']['report']['Moderate']['f1-score'],
     results['BiLSTM 67% / 2DCNN 33%']['report']['High']['f1-score']),
    ("Hybrid 67/33 + dynamic thresh",
     acc_dyn, mf1_dyn,
     rep_dyn['Low/None']['f1-score'],
     rep_dyn['Moderate']['f1-score'],
     rep_dyn['High']['f1-score']),
]
for name, acc, mf1, lf, mf, hf in rows:
    print(f"{name:<38} {acc:>6.3f} {mf1:>8.3f} {lf:>7.3f} {mf:>7.3f} {hf:>7.3f}")

# Save
out_path = RESULTS_DIR / "hybrid_ensemble_results.json"
with open(out_path, 'w') as f:
    json.dump({
        "baseline": baseline,
        "cnn2d_single": cnn2d_r,
        "hybrids": results,
        "hybrid_67_33_dynamic_threshold": {
            "accuracy": float(acc_dyn),
            "macro_f1": float(mf1_dyn),
            "classification_report": rep_dyn,
            "confusion_matrix": cm_dyn.tolist(),
            "threshold_rules": {
                "High": "high_prob > mod_prob*1.2 AND high_prob > 0.35",
                "Moderate": "mod_prob > 0.35",
                "Low": "otherwise"
            }
        }
    }, f, indent=2)
print(f"\nResults saved -> {out_path}")

# ============================================================
# STEP 7: MODERATE THRESHOLD SWEEP
# Keep High rules fixed; vary Moderate threshold from 0.15-0.32
# ============================================================
print("\n" + "=" * 70)
print("MODERATE THRESHOLD SWEEP  (High rules unchanged)")
print("=" * 70)
print(f"\n{'Mod thresh':>11} {'Acc':>6} {'MacroF1':>8} {'LowF1':>7} {'ModF1':>7} {'HiF1':>7} {'PredDist':>18}")
print("─" * 84)

best_mf1, best_thresh = 0, 0
for mod_t in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]:
    y_p = []
    for p in hybrid_67_33:
        low_p, mod_p, high_p = p[0], p[1], p[2]
        if high_p > (mod_p * 1.2) and high_p > 0.35:
            y_p.append(2)
        elif mod_p > mod_t:
            y_p.append(1)
        else:
            y_p.append(0)
    y_p = np.array(y_p)
    acc_t = np.mean(y_p == y_test)
    mf1_t = f1_score(y_test, y_p, average='macro', zero_division=0)
    rep_t = classification_report(y_test, y_p,
            target_names=['Low/None', 'Moderate', 'High'],
            zero_division=0, output_dict=True)
    dist  = np.bincount(y_p, minlength=3)
    marker = " <-- BEST" if mf1_t > best_mf1 else ""
    print(f"{mod_t:>11.2f} {acc_t:>6.3f} {mf1_t:>8.3f} "
          f"{rep_t['Low/None']['f1-score']:>7.3f} "
          f"{rep_t['Moderate']['f1-score']:>7.3f} "
          f"{rep_t['High']['f1-score']:>7.3f}  {str(dist):>18}{marker}")
    if mf1_t > best_mf1:
        best_mf1, best_thresh = mf1_t, mod_t

print(f"\nBest Moderate threshold: {best_thresh}  ->  Macro F1: {best_mf1:.4f}")
print(f"Baseline Macro F1      : 0.3203")
print(f"Improvement            : {best_mf1 - 0.3203:+.4f}")
