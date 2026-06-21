#!/usr/bin/env python3
"""
Prototype: Pseudo-Image 2D CNN for AU Depression Screening
Transforms (300, 17) AU sequences -> (17, 300, 1) pseudo-images.
Rows = 17 AUs (spatial/concurrent activation axis)
Cols = 300 frames (temporal axis)
Uses Conv2D to capture concurrent AU activations across time.

Baseline to beat:
  CNN-BiLSTM ensemble -> Accuracy: 51.8%, Macro F1: 32.0%
  Low F1: 0.68 | Moderate F1: 0.00 | High F1: 0.28
"""

import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / "scripts"))
from data_processor_2026 import EDAICDataProcessor

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path("c:/Users/user/OneDrive/Desktop/FYP")
RESULTS_DIR = BASE_PATH / "proposal_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING  (identical to baseline pipeline)
# ============================================================
print("=" * 70)
print("LOADING DATA & PREPROCESSING")
print("=" * 70)

data_processor = EDAICDataProcessor(BASE_PATH / "data/edaic", sequence_length=300)
splits = data_processor.load_split_files()
participant_severity, labels_df = data_processor.load_detailed_labels()
au_files = data_processor.get_participant_files()

datasets = data_processor.create_datasets(splits, participant_severity, au_files, scaling_factor=1.0)
processed_data = data_processor.load_features_and_labels(datasets, scaling_factor=1.0)

for split_name, data in processed_data.items():
    if len(data['y']) > 0:
        print(f"  {split_name.upper()}: {np.bincount(data['y'])}")

# ============================================================
# STEP 2: TEMPORAL WINDOWING  (identical to baseline)
# ============================================================
def create_temporal_windows(X_data, y_data, max_frames=600):
    X_windows, y_windows = [], []
    for seq, label in zip(X_data, y_data):
        seq_len = seq.shape[0]
        if seq_len >= 300:
            X_windows.append(seq[:300]); y_windows.append(label)
        else:
            w = np.vstack([seq, np.zeros((300 - seq_len, seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
        if max_frames >= 450:
            if seq_len >= 450:
                X_windows.append(seq[150:450]); y_windows.append(label)
            elif seq_len > 150:
                w = seq[150:]
                if len(w) < 300:
                    w = np.vstack([w, np.zeros((300 - len(w), seq.shape[1]))])
                X_windows.append(w); y_windows.append(label)
        if max_frames >= 600:
            if seq_len >= 600:
                X_windows.append(seq[300:600]); y_windows.append(label)
            elif seq_len > 300:
                w = seq[300:]
                if len(w) < 300:
                    w = np.vstack([w, np.zeros((300 - len(w), seq.shape[1]))])
                X_windows.append(w); y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)

X_train_t, y_train_t = create_temporal_windows(
    processed_data['train']['X'], processed_data['train']['y'])
processed_data['train']['X'] = X_train_t
processed_data['train']['y'] = y_train_t

for split_name in ['dev', 'test']:
    if len(processed_data[split_name]['y']) > 0:
        X_s, y_s = create_temporal_windows(
            processed_data[split_name]['X'], processed_data[split_name]['y'], max_frames=300)
        processed_data[split_name]['X'] = X_s
        processed_data[split_name]['y'] = y_s

# ============================================================
# STEP 3: AU-ONLY MASKING  (identical to baseline)
# ============================================================
all_features = (data_processor.au_features
                + data_processor.pose_features
                + data_processor.gaze_features)
au_mask = np.ones(len(all_features), dtype=bool)
pose_start = len(data_processor.au_features)
pose_end   = pose_start + len(data_processor.pose_features)
au_mask[pose_start:pose_end + len(data_processor.gaze_features)] = False

for split_name in ['train', 'dev', 'test']:
    if len(processed_data[split_name]['X']) > 0:
        processed_data[split_name]['X'] = processed_data[split_name]['X'][:, :, au_mask]

n_au = np.sum(au_mask)  # 17
print(f"AU features: {n_au}")

# ============================================================
# STEP 4: STANDARDSCALER  (identical to baseline)
# ============================================================
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

# ============================================================
# STEP 5: PSEUDO-IMAGE TRANSFORM  (300, 17) -> (17, 300, 1)
# AUs become the row axis (concurrent activations detectable
# by tall Conv2D kernels), frames become the column axis.
# ============================================================
def to_pseudo_image(X):
    """(N, 300, 17) -> (N, 17, 300, 1)"""
    X_t = np.transpose(X, (0, 2, 1))          # (N, 17, 300)
    return X_t[:, :, :, np.newaxis].astype(np.float32)  # (N, 17, 300, 1)

X_train_img = to_pseudo_image(processed_data['train']['X'])
X_val_img   = to_pseudo_image(processed_data['dev']['X'])
X_test_img  = to_pseudo_image(processed_data['test']['X'])
y_train     = processed_data['train']['y']
y_val       = processed_data['dev']['y']
y_test      = processed_data['test']['y']

print(f"\nPseudo-image shapes:")
print(f"  Train : {X_train_img.shape}")
print(f"  Val   : {X_val_img.shape}")
print(f"  Test  : {X_test_img.shape}")

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=3)

# ============================================================
# STEP 6: 2D CNN ARCHITECTURE
# Kernel strategy:
#   (3, 7)  -> few AUs wide, captures short temporal bursts
#   (3, 5)  -> narrower temporal window in deeper layers
#   Final (n_au, 1) -> integrates across ALL 17 AUs simultaneously
# ============================================================
def build_2d_cnn(n_au=17, n_frames=300):
    inp = layers.Input(shape=(n_au, n_frames, 1), name='pseudo_image_input')

    # Block 1 - detect short-duration AU co-activations
    x = layers.Conv2D(32, (3, 7), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)     # (17, 150, 32)
    x = layers.SpatialDropout2D(0.2)(x)

    # Block 2 - detect medium-duration patterns
    x = layers.Conv2D(64, (3, 5), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)     # (17, 75, 64)
    x = layers.SpatialDropout2D(0.3)(x)

    # Block 3 - detect sustained activation patterns
    x = layers.Conv2D(128, (3, 5), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2))(x)     # (17, 37, 128)

    # Integrate across ALL AU rows simultaneously
    x = layers.Conv2D(128, (n_au, 1), padding='valid', activation='relu')(x)
    # Shape: (1, 37, 128) - full AU integration

    x = layers.GlobalAveragePooling2D()(x)  # (128,)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(3, activation='softmax', name='severity_output')(x)

    return models.Model(inputs=inp, outputs=out, name='pseudo_image_2d_cnn')

# ============================================================
# STEP 7: TRAINING  (single model - lightweight prototype)
# ============================================================
print("\n" + "=" * 70)
print("TRAINING PSEUDO-IMAGE 2D CNN")
print("=" * 70)

# Same class weights as baseline
class_weights = {0: 1.0, 1: 1.5, 2: 2.0}

model = build_2d_cnn(n_au=n_au, n_frames=seq_len)
model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),  # no label smoothing
    metrics=['accuracy']
)

class MacroF1Callback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_v, self.y_v = validation_data
        self.best_macro_f1 = 0.0
    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_v, verbose=0), axis=1)
        macro_f1 = f1_score(self.y_v, y_pred, average='macro', zero_division=0)
        logs['val_macro_f1'] = macro_f1
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1

early_stop  = callbacks.EarlyStopping(
    monitor='val_macro_f1', patience=12, restore_best_weights=True, mode='max')
reduce_lr   = callbacks.ReduceLROnPlateau(
    monitor='val_macro_f1', factor=0.5, patience=5, min_lr=1e-6, mode='max')
macro_f1_cb = MacroF1Callback((X_val_img, y_val))

history = model.fit(
    X_train_img, y_train_cat,
    validation_data=(X_val_img, y_val_cat),
    epochs=60,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, macro_f1_cb],
    verbose=1
)

# ============================================================
# STEP 8: EVALUATION  (pure argmax - same as baseline)
# ============================================================
print("\n" + "=" * 70)
print("PROTOTYPE EVALUATION - PURE ARGMAX")
print("=" * 70)

y_pred_proba = model.predict(X_test_img, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

print(f"Test label distribution : {np.bincount(y_test)}")
print(f"Prediction distribution : {np.bincount(y_pred, minlength=3)}")

report = classification_report(
    y_test, y_pred,
    target_names=['Low/None', 'Moderate', 'High'],
    zero_division=0
)
print("\nClassification Report:")
print(report)

cm         = confusion_matrix(y_test, y_pred)
accuracy   = np.sum(y_pred == y_test) / len(y_test)
macro_f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Confusion Matrix:\n{cm}")
print(f"\nOverall Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"Macro F1-Score   : {macro_f1:.4f} ({macro_f1*100:.1f}%)")

# ============================================================
# STEP 9: COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON: 2D CNN vs CNN-BiLSTM BASELINE")
print("=" * 70)
baseline = {"accuracy": 0.5179, "macro_f1": 0.3203,
            "low_f1": 0.68, "moderate_f1": 0.00, "high_f1": 0.28}

report_dict = classification_report(
    y_test, y_pred,
    target_names=['Low/None', 'Moderate', 'High'],
    zero_division=0, output_dict=True)

print(f"{'Metric':<20} {'Baseline':>12} {'2D CNN':>12} {'Delta':>10}")
print("-" * 56)
for metric, b_val, cnn_val in [
    ("Accuracy",       baseline["accuracy"],      accuracy),
    ("Macro F1",       baseline["macro_f1"],       macro_f1),
    ("Low/None F1",    baseline["low_f1"],         report_dict['Low/None']['f1-score']),
    ("Moderate F1",    baseline["moderate_f1"],    report_dict['Moderate']['f1-score']),
    ("High F1",        baseline["high_f1"],        report_dict['High']['f1-score']),
]:
    delta = cnn_val - b_val
    sign  = "+" if delta >= 0 else ""
    print(f"{metric:<20} {b_val:>12.4f} {cnn_val:>12.4f} {sign+f'{delta:.4f}':>10}")

# Save results
results = {
    "model_type"   : "Prototype - Pseudo-Image 2D CNN",
    "input_shape"  : [17, 300, 1],
    "loss"         : "CategoricalCrossentropy (no label smoothing)",
    "accuracy"     : float(accuracy),
    "macro_f1"     : float(macro_f1),
    "classification_report": report_dict,
    "confusion_matrix"     : cm.tolist(),
    "baseline_macro_f1"    : baseline["macro_f1"],
    "improvement_macro_f1" : float(macro_f1 - baseline["macro_f1"]),
}
out_path = RESULTS_DIR / "prototype_2d_cnn_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved -> {out_path}")

# Save model for hybrid ensemble evaluation
model_save_path = BASE_PATH / "web_app" / "models" / "prototype_2d_cnn.h5"
model.save(str(model_save_path))
print(f"2D CNN model saved -> {model_save_path}")
