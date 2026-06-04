"""
Optimise dynamic thresholding with temperature-scaled probability calibration.
Uses the dev set to find the best temperature + threshold parameters,
then evaluates on the test set.  No retraining required.
"""
import sys, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent / "scripts"))
from data_processor_2026 import EDAICDataProcessor

BASE_PATH = Path("c:/Users/user/OneDrive/Desktop/FYP")

# ── 1. Load & prepare data (identical pipeline to training) ─────────────────
print("Loading data...")
dp = EDAICDataProcessor(BASE_PATH / "data/edaic", sequence_length=300)
splits        = dp.load_split_files()
sev, _        = dp.load_detailed_labels()
au_files      = dp.get_participant_files()
datasets      = dp.create_datasets(splits, sev, au_files)
processed_data = dp.load_features_and_labels(datasets)

def create_temporal_windows(X_data, y_data, max_frames=300):
    Xw, yw = [], []
    for seq, label in zip(X_data, y_data):
        sl = seq.shape[0]
        if sl >= 300:
            Xw.append(seq[:300]); yw.append(label)
        else:
            Xw.append(np.vstack([seq, np.zeros((300-sl, seq.shape[1]))])); yw.append(label)
        if max_frames >= 450:
            if sl >= 450:
                Xw.append(seq[150:450]); yw.append(label)
            elif sl > 150:
                w = seq[150:]
                if len(w) < 300:
                    w = np.vstack([w, np.zeros((300-len(w), seq.shape[1]))])
                Xw.append(w); yw.append(label)
        if max_frames >= 600:
            if sl >= 600:
                Xw.append(seq[300:600]); yw.append(label)
            elif sl > 300:
                w = seq[300:]
                if len(w) < 300:
                    w = np.vstack([w, np.zeros((300-len(w), seq.shape[1]))])
                Xw.append(w); yw.append(label)
    return np.array(Xw), np.array(yw)

# Augment train, single window for dev/test
X_train_t, y_train_t = create_temporal_windows(processed_data['train']['X'], processed_data['train']['y'], max_frames=600)
processed_data['train']['X'] = X_train_t
processed_data['train']['y'] = y_train_t
for sn in ['dev', 'test']:
    if len(processed_data[sn]['y']) > 0:
        X_s, y_s = create_temporal_windows(processed_data[sn]['X'], processed_data[sn]['y'], max_frames=300)
        processed_data[sn]['X'] = X_s
        processed_data[sn]['y'] = y_s

# AU-only mask
all_f   = dp.au_features + dp.pose_features + dp.gaze_features
au_mask = np.ones(len(all_f), dtype=bool)
ps = len(dp.au_features); pe = ps + len(dp.pose_features)
au_mask[ps:pe] = False; au_mask[pe:] = False
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

# ── 2. Load saved ensemble models ───────────────────────────────────────────
print("Loading saved models...")
models = [tf.keras.models.load_model(f'web_app/models/ensemble_{i}.h5', compile=False) for i in range(1, 4)]

def get_raw_ensemble_proba(X, weights=[1.0, 1.0, 1.5]):
    preds = [m.predict(X, verbose=0) for m in models]
    wp    = [p * w for p, w in zip(preds, weights)]
    ens   = np.mean(wp, axis=0)
    return ens / np.sum(ens, axis=1, keepdims=True)

# ── 3. Temperature scaling (calibrate on dev set) ───────────────────────────
# Temperature T > 1 → flattens probabilities (less confident)
# We pick the T that maximises macro-F1 on the dev set

X_dev = processed_data['dev']['X']
y_dev = processed_data['dev']['y']
X_test = processed_data['test']['X']
y_test = processed_data['test']['y']

raw_dev  = get_raw_ensemble_proba(X_dev)
raw_test = get_raw_ensemble_proba(X_test)

def apply_temperature(proba, T):
    log_p = np.log(proba + 1e-9) / T
    exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
    return exp_p / exp_p.sum(axis=1, keepdims=True)

def dynamic_predict(proba, ratio, conf):
    y_pred = np.argmax(proba, axis=1).copy()
    h = proba[:, 2]; mo = proba[:, 1]
    mask = (h > mo * ratio) & (h > conf)
    y_pred[mask] = 2
    return y_pred, np.sum(mask)

# ── 4. Grid search on dev set ───────────────────────────────────────────────
print("\nSearching best Temperature + Threshold on DEV set...")
print(f"{'Temp':>6} {'Ratio':>6} {'Conf':>6} {'Acc':>6} {'F1':>6} {'High→':>7}")
print("-" * 50)

best_f1      = 0
best_params  = None
results_grid = []

for T in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    cal_dev = apply_temperature(raw_dev, T)
    for ratio in [1.2, 1.5, 2.0, 2.5, 3.0]:
        for conf in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            y_p, n_high = dynamic_predict(cal_dev, ratio, conf)
            acc = np.mean(y_p == y_dev)
            f1  = f1_score(y_dev, y_p, average='macro', zero_division=0)
            results_grid.append((T, ratio, conf, acc, f1, n_high))
            if f1 > best_f1:
                best_f1    = f1
                best_params = (T, ratio, conf)

# Print top 10
top10 = sorted(results_grid, key=lambda x: -x[4])[:10]
for T, ratio, conf, acc, f1, nh in top10:
    print(f"{T:>6.1f} {ratio:>6.1f} {conf:>6.2f} {acc:>5.1%} {f1:>5.1%} {nh:>6d}")

# ── 5. Evaluate best params on test set ─────────────────────────────────────
T_best, ratio_best, conf_best = best_params
print(f"\nBest params (T={T_best}, ratio>{ratio_best}, conf>{conf_best}) → Dev Macro F1: {best_f1:.1%}")

cal_test = apply_temperature(raw_test, T_best)
y_pred_test, n_high = dynamic_predict(cal_test, ratio_best, conf_best)

test_acc = np.mean(y_pred_test == y_test)
test_f1  = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
cm       = confusion_matrix(y_test, y_pred_test)

print(f"\n{'='*60}")
print(f"TEST SET  (Temperature={T_best}, P(H)>P(M)*{ratio_best} & P(H)>{conf_best})")
print(f"{'='*60}")
print(f"Samples: {len(y_test)}, Distribution: {np.bincount(y_test)}")
print(f"Predicted: {np.bincount(y_pred_test, minlength=3)}  (reclassified as High: {n_high})")
print(f"Accuracy:  {test_acc:.1%}")
print(f"Macro F1:  {test_f1:.1%}")
print(f"\n{classification_report(y_test, y_pred_test, target_names=['Low/None','Moderate','High'], zero_division=0)}")
print(f"Confusion Matrix:\n{cm}")

diag = np.trace(cm) / cm.sum()
avg_d  = np.mean([cm[i,i] for i in range(3)])
avg_od = np.mean([cm[i,j] for i in range(3) for j in range(3) if i!=j])
print(f"Diagonal Dominance: {avg_d/avg_od:.2f}")
