#!/usr/bin/env python3
"""
Generate Training History Graphs (Accuracy vs Epoch, Loss vs Epoch)
Re-runs the ensemble training pipeline and saves actual training curves.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from data_processor_2026 import EDAICDataProcessor

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

BASE_PATH = Path("c:/Users/user/OneDrive/Desktop/FYP")
RESULTS_DIR = BASE_PATH / "proposal_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1: DATA LOADING & PREPROCESSING (same as proposal_flow_2026.py)
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

# Labels are already correctly mapped by data_processor (0-9→0, 10-14→1, 15+→2)
# No random re-mapping needed
for split_name, data in processed_data.items():
    if len(data['y']) > 0:
        print(f"  {split_name.upper()}: {np.bincount(data['y'])}")

# Balanced Temporal Augmentation - same windowing for ALL classes
def create_temporal_windows(X_data, y_data, max_frames=600):
    X_windows, y_windows = [], []
    for seq, label in zip(X_data, y_data):
        seq_len = seq.shape[0]
        # Window 1: frames 0-300 (always included)
        if seq_len >= 300:
            X_windows.append(seq[:300]); y_windows.append(label)
        else:
            w = np.vstack([seq, np.zeros((300 - seq_len, seq.shape[1]))])
            X_windows.append(w); y_windows.append(label)
        # Window 2: frames 150-450 (only if max_frames allows)
        if max_frames >= 450:
            if seq_len >= 450:
                X_windows.append(seq[150:450]); y_windows.append(label)
            elif seq_len > 150:
                w = seq[150:]
                if len(w) < 300:
                    w = np.vstack([w, np.zeros((300 - len(w), seq.shape[1]))])
                X_windows.append(w); y_windows.append(label)
        # Window 3: frames 300-600 (only if max_frames allows)
        if max_frames >= 600:
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

for split_name in ['dev', 'test']:
    if len(processed_data[split_name]['y']) > 0:
        X_s, y_s = create_temporal_windows(processed_data[split_name]['X'], processed_data[split_name]['y'], max_frames=300)
        processed_data[split_name]['X'] = X_s
        processed_data[split_name]['y'] = y_s

# No extra augmentation - synthetic noise copies hurt performance on this small dataset

# Feature Masking - AU only
all_features = data_processor.au_features + data_processor.pose_features + data_processor.gaze_features
au_mask = np.ones(len(all_features), dtype=bool)
pose_start = len(data_processor.au_features)
pose_end = pose_start + len(data_processor.pose_features)
gaze_start = pose_end
gaze_end = gaze_start + len(data_processor.gaze_features)
au_mask[pose_start:pose_end] = False
au_mask[gaze_start:gaze_end] = False

for split_name in ['train', 'dev', 'test']:
    if len(processed_data[split_name]['X']) > 0:
        processed_data[split_name]['X'] = processed_data[split_name]['X'][:, :, au_mask]

n_features = np.sum(au_mask)
print(f"Features: {n_features} (AU-only)")

# StandardScaler normalization
scaler = StandardScaler()
X_train = processed_data['train']['X']
n_samples, seq_len, nf = X_train.shape
X_train_flat = X_train.reshape(-1, nf)
X_train_scaled = scaler.fit_transform(X_train_flat)
processed_data['train']['X'] = X_train_scaled.reshape(n_samples, seq_len, nf)

for split_name in ['dev', 'test']:
    if len(processed_data[split_name]['X']) > 0:
        X = processed_data[split_name]['X']
        ns = X.shape[0]
        X_flat = X.reshape(-1, nf)
        X_scaled = scaler.transform(X_flat)
        processed_data[split_name]['X'] = X_scaled.reshape(ns, seq_len, nf)

# ============================================================
# STEP 2: BUILD & TRAIN ENSEMBLE MODELS
# ============================================================
print("\n" + "=" * 70)
print("BUILDING & TRAINING ENSEMBLE MODELS")
print("=" * 70)

def build_model(n_features, model_index):
    inputs = layers.Input(shape=(300, n_features), name='input_layer')
    noise_input = layers.GaussianNoise(0.05, name='gaussian_noise')(inputs)
    cnn = layers.Conv1D(64, 3, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(0.001), name='cnn_scan')(noise_input)
    cnn = layers.SpatialDropout1D(0.3, name='spatial_dropout')(cnn)
    cnn = layers.LayerNormalization(name='cnn_layer_norm')(cnn)
    lstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), name='lstm_temporal'),
        name='bidirectional_lstm'
    )(cnn)
    lstm = layers.LayerNormalization(name='lstm_layer_norm')(lstm)
    pooled = layers.GlobalAveragePooling1D(name='temporal_summary')(lstm)
    dense = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_layer')(pooled)
    dense = layers.Dropout(0.4, name='final_dropout')(dense)
    outputs = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.001), name='three_class_softmax')(dense)
    model = models.Model(inputs=inputs, outputs=outputs, name=f'ensemble_{model_index}')
    return model

# Focal loss
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = tf.one_hot(tf.cast(y_true[:, 0], tf.int32), depth=3)
        cross_entropy = -y_true * tf.math.log(y_pred)
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha_weights = tf.constant(alpha, dtype=tf.float32)
            else:
                alpha_weights = alpha
            focal_loss = alpha_weights * tf.pow(1 - y_pred, gamma) * cross_entropy
        else:
            focal_loss = tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    return loss

# Prepare data
X_train = processed_data['train']['X']
y_train = processed_data['train']['y']
X_val = processed_data['dev']['X']
y_val = processed_data['dev']['y']

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=3)
y_train_smooth = y_train_cat * (1 - 0.1) + 0.1 / 3
y_val_smooth = y_val_cat * (1 - 0.1) + 0.1 / 3

manual_alpha_weights = np.array([0.2, 0.4, 0.8])
final_class_weights = np.array([1.0, 1.5, 2.0])

all_histories = []
trained_models = []

for i in range(3):
    print(f"\n--- Training Model {i+1}/3 ---")
    model = build_model(n_features, i+1)

    fresh_optimizer = optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=fresh_optimizer,
        loss=categorical_focal_loss(gamma=2.0, alpha=manual_alpha_weights),
        metrics=['accuracy']
    )

    class WarmupCallback(callbacks.Callback):
        def __init__(self, warmup_epochs=5, target_lr=1e-3):
            super().__init__()
            self.warmup_epochs = warmup_epochs
            self.target_lr = target_lr
        def on_epoch_end(self, epoch, logs=None):
            if epoch == self.warmup_epochs - 1:
                self.model.optimizer.learning_rate.assign(self.target_lr)
                print(f"Warmup complete! LR -> {self.target_lr}")

    class MacroF1Callback(callbacks.Callback):
        def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data
            self.best_macro_f1 = 0.0
        def on_epoch_end(self, epoch, logs=None):
            X_v, y_v = self.validation_data
            y_pred = np.argmax(self.model.predict(X_v, verbose=0), axis=1)
            macro_f1 = f1_score(y_v, y_pred, average='macro')
            logs['val_macro_f1'] = macro_f1
            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = macro_f1

    macro_f1_cb = MacroF1Callback((X_val, y_val))
    warmup_cb = WarmupCallback(warmup_epochs=5, target_lr=1e-3)

    early_stop = callbacks.EarlyStopping(
        monitor='val_macro_f1', patience=10, restore_best_weights=True, mode='max'
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_macro_f1', factor=0.5, patience=5, min_lr=1e-6, mode='max'
    )

    history = model.fit(
        X_train, y_train_smooth,
        validation_data=(X_val, y_val_smooth),
        epochs=50,
        batch_size=32,
        class_weight=dict(enumerate(final_class_weights)),
        callbacks=[early_stop, reduce_lr, macro_f1_cb, warmup_cb],
        verbose=1
    )

    all_histories.append(history.history)
    trained_models.append(model)
    print(f"Model {i+1} done! Epochs trained: {len(history.history['accuracy'])}")

# Save ensemble models to web_app/models/
import os
models_dir = os.path.join(os.path.dirname(__file__), 'web_app', 'models')
os.makedirs(models_dir, exist_ok=True)
for i, m in enumerate(trained_models):
    model_path = os.path.join(models_dir, f'ensemble_{i+1}.h5')
    m.save(model_path)
    print(f"Saved Model {i+1} to: {model_path}")

# ============================================================
# STEP 3: ENSEMBLE EVALUATION WITH PURE ARGMAX
# ============================================================
print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION WITH PURE ARGMAX")
print("=" * 70)

# Use test set if available, otherwise dev set
if len(processed_data['test']['y']) > 0:
    X_eval = processed_data['test']['X']
    y_eval = processed_data['test']['y']
    eval_set_name = "Test"
else:
    X_eval = X_val
    y_eval = y_val
    eval_set_name = "Dev (Validation)"

print(f"Evaluation set: {eval_set_name}")
print(f"Evaluation data shape: {X_eval.shape}")
print(f"Evaluation labels distribution: {np.bincount(y_eval)}")

# Get predictions from all models
all_predictions = []
for i, m in enumerate(trained_models):
    pred_proba = m.predict(X_eval, verbose=0)
    all_predictions.append(pred_proba)
    print(f"Model {i+1} predictions shape: {pred_proba.shape}")

# Confidence-Weighted Voting
model_weights = [1.0, 1.0, 1.5]
weighted_predictions = [pred * w for pred, w in zip(all_predictions, model_weights)]
ensemble_proba = np.mean(weighted_predictions, axis=0)
ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1, keepdims=True)

# Pure Argmax - select class with highest ensemble probability
y_pred = np.argmax(ensemble_proba, axis=1)
print(f"Prediction distribution: {np.bincount(y_pred, minlength=3)}")

# Classification Report
from sklearn.metrics import classification_report, confusion_matrix
report = classification_report(y_eval, y_pred, target_names=['Low/None', 'Moderate', 'High'], output_dict=True)
print("\nClassification Report:")
print(classification_report(y_eval, y_pred, target_names=['Low/None', 'Moderate', 'High']))

# Confusion Matrix
cm = confusion_matrix(y_eval, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Diagonal analysis
diagonal_correct = np.trace(cm)
total_samples = np.sum(cm)
diagonal_accuracy = diagonal_correct / total_samples
diagonal_elements = [cm[i, i] for i in range(3)]
off_diagonal_elements = [cm[i, j] for i in range(3) for j in range(3) if i != j]
avg_diagonal = np.mean(diagonal_elements)
avg_off_diagonal = np.mean(off_diagonal_elements)
dominance_ratio = avg_diagonal / avg_off_diagonal if avg_off_diagonal > 0 else 0

overall_accuracy = report['accuracy']
macro_f1 = report['macro avg']['f1-score']

print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.1%})")
print(f"Macro F1-Score: {macro_f1:.4f} ({macro_f1:.1%})")
print(f"Diagonal Dominance Ratio: {dominance_ratio:.2f}")

# Save confusion matrix plot
try:
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low/None', 'Moderate', 'High'],
                yticklabels=['Low/None', 'Moderate', 'High'])
    plt.title('2026 Ensemble - 3x3 Confusion Matrix\n'
              'CNN-BiLSTM Ensemble Voting + Pure Argmax')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = RESULTS_DIR / "ensemble_2026_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
except ImportError:
    print("seaborn not installed, skipping confusion matrix heatmap")

# Save results JSON
results = {
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'macro_f1_score': macro_f1,
    'overall_accuracy': overall_accuracy,
    'target_achieved': macro_f1 >= 0.60,
    'model_type': '2026 Ensemble - Pure Argmax',
    'classes': ['Low/None', 'Moderate', 'High'],
    'ensemble_size': len(trained_models),
    'dynamic_thresholding': False,
    'diagonal_accuracy': diagonal_accuracy,
    'diagonal_dominance_ratio': dominance_ratio,
    'evaluation_set': eval_set_name
}

results_file = RESULTS_DIR / "ensemble_2026_results.json"
pd.DataFrame([results]).to_json(results_file, indent=2)
print(f"Results saved to: {results_file}")

# ============================================================
# STEP 4: SAVE TRAINING HISTORY
# ============================================================
print("\n" + "=" * 70)
print("SAVING TRAINING HISTORY")
print("=" * 70)

history_data = {}
for i, h in enumerate(all_histories):
    history_data[f'model_{i+1}'] = {
        'accuracy': [float(v) for v in h['accuracy']],
        'val_accuracy': [float(v) for v in h['val_accuracy']],
        'loss': [float(v) for v in h['loss']],
        'val_loss': [float(v) for v in h['val_loss']],
    }

history_file = RESULTS_DIR / "training_history.json"
with open(history_file, 'w') as f:
    json.dump(history_data, f, indent=2)
print(f"Training history saved to: {history_file}")

# ============================================================
# STEP 4: GENERATE GRAPHS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING ACCURACY & LOSS GRAPHS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('CNN-BiLSTM Ensemble Training History', fontsize=16, fontweight='bold')

for i, h in enumerate(all_histories):
    epochs = range(1, len(h['accuracy']) + 1)

    # Accuracy plot
    axes[0, i].plot(epochs, h['accuracy'], 'b-', label='Training Accuracy', linewidth=1.5)
    axes[0, i].plot(epochs, h['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=1.5)
    axes[0, i].set_title(f'Model {i+1} - Accuracy vs Epoch', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('Epoch')
    axes[0, i].set_ylabel('Accuracy')
    axes[0, i].legend(loc='lower right')
    axes[0, i].grid(True, alpha=0.3)

    # Loss plot
    axes[1, i].plot(epochs, h['loss'], 'b-', label='Training Loss', linewidth=1.5)
    axes[1, i].plot(epochs, h['val_loss'], 'r-', label='Validation Loss', linewidth=1.5)
    axes[1, i].set_title(f'Model {i+1} - Loss vs Epoch', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel('Epoch')
    axes[1, i].set_ylabel('Loss')
    axes[1, i].legend(loc='upper right')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
graph_path = RESULTS_DIR / "ensemble_training_history.png"
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"Combined graph saved to: {graph_path}")
plt.show()

# Individual model graphs (larger, one per model)
for i, h in enumerate(all_histories):
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f'Ensemble Model {i+1} - Training History', fontsize=14, fontweight='bold')
    epochs = range(1, len(h['accuracy']) + 1)

    ax1.plot(epochs, h['accuracy'], 'b-o', label='Training Accuracy', markersize=3, linewidth=1.5)
    ax1.plot(epochs, h['val_accuracy'], 'r-o', label='Validation Accuracy', markersize=3, linewidth=1.5)
    ax1.set_title('Accuracy vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, h['loss'], 'b-o', label='Training Loss', markersize=3, linewidth=1.5)
    ax2.plot(epochs, h['val_loss'], 'r-o', label='Validation Loss', markersize=3, linewidth=1.5)
    ax2.set_title('Loss vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    individual_path = RESULTS_DIR / f"model_{i+1}_training_history.png"
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    print(f"Model {i+1} graph saved to: {individual_path}")
    plt.show()

# ============================================================
# STEP 5: PRINT TRAINING DATASET SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DATASET SUMMARY FOR REPORT")
print("=" * 70)
print(f"Training samples: {len(y_train)}")
print(f"  Low/None: {np.sum(y_train == 0)}")
print(f"  Moderate: {np.sum(y_train == 1)}")
print(f"  High:     {np.sum(y_train == 2)}")
print(f"Validation samples: {len(y_val)}")
print(f"  Low/None: {np.sum(y_val == 0)}")
print(f"  Moderate: {np.sum(y_val == 1)}")
print(f"  High:     {np.sum(y_val == 2)}")
if len(processed_data['test']['y']) > 0:
    y_test = processed_data['test']['y']
    print(f"Test samples: {len(y_test)}")
    print(f"  Low/None: {np.sum(y_test == 0)}")
    print(f"  Moderate: {np.sum(y_test == 1)}")
    print(f"  High:     {np.sum(y_test == 2)}")
else:
    print("Test set: Empty (using dev set for evaluation)")

for i, h in enumerate(all_histories):
    best_epoch = np.argmax(h['val_accuracy']) + 1
    best_val_acc = max(h['val_accuracy'])
    final_train_acc = h['accuracy'][-1]
    print(f"\nModel {i+1}:")
    print(f"  Epochs trained: {len(h['accuracy'])}")
    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Final train accuracy: {final_train_acc:.4f}")

print("\n" + "=" * 70)
print("ALL DONE! Check proposal_results/ for graphs and history.")
print("=" * 70)
