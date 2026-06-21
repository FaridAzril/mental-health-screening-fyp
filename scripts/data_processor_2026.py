"""
E-DAIC Data Processor for 2026 Ensemble Training
Handles loading AU/Pose/Gaze features, PHQ-8 labels, and train/dev/test splits.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class EDAICDataProcessor:
    def __init__(self, data_path, sequence_length=300):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        
        # Feature column names from OpenFace output
        self.au_features = [
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
            'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        ]
        
        self.pose_features = [
            'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz'
        ]
        
        self.gaze_features = [
            'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
            'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
            'gaze_angle_x', 'gaze_angle_y'
        ]
    
    def load_split_files(self):
        """Load train/dev/test split participant IDs"""
        splits = {}
        for split_name in ['train', 'dev', 'test']:
            split_file = self.data_path / f"{split_name}_split.csv"
            if split_file.exists():
                df = pd.read_csv(split_file)
                splits[split_name] = df.iloc[:, 0].astype(str).tolist()
                print(f"  {split_name}: {len(splits[split_name])} participants")
            else:
                splits[split_name] = []
                print(f"  {split_name}: split file not found")
        return splits
    
    def load_detailed_labels(self):
        """Load PHQ-8 labels and return participant severity mapping"""
        # Try both possible label files
        labels_file = self.data_path / "detailed_lables.csv"
        if not labels_file.exists():
            labels_file = self.data_path / "Detailed_PHQ8_Labels.csv"
        
        df = pd.read_csv(labels_file)
        
        # Identify columns
        id_col = None
        score_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'participant' in col_lower or col_lower == 'id':
                id_col = col
            if 'total' in col_lower or col_lower == 'phq_score' or col_lower == 'phq8_score':
                score_col = col
        
        if id_col is None:
            id_col = df.columns[0]
        
        if score_col is None:
            # Try to find PHQ8 total or sum individual items
            phq_item_cols = [c for c in df.columns if 'phq' in c.lower() and c != id_col]
            if len(phq_item_cols) >= 8:
                df['_phq8_total'] = df[phq_item_cols[:8]].sum(axis=1)
                score_col = '_phq8_total'
            else:
                score_col = df.columns[-1]
        
        # Build severity mapping
        participant_severity = {}
        for _, row in df.iterrows():
            try:
                pid = str(int(row[id_col]))
            except (ValueError, TypeError):
                continue
            
            score = row[score_col]
            if pd.isna(score):
                continue
            
            score = int(score)
            # 3-class mapping
            if score <= 9:
                severity = 0  # Low/None
            elif score <= 14:
                severity = 1  # Moderate
            else:
                severity = 2  # High
            
            participant_severity[pid] = severity
        
        print(f"  Loaded {len(participant_severity)} participant labels")
        
        # Count distribution
        counts = [0, 0, 0]
        for s in participant_severity.values():
            counts[s] += 1
        print(f"  Distribution: Low={counts[0]}, Moderate={counts[1]}, High={counts[2]}")
        
        return participant_severity, df
    
    def get_participant_files(self):
        """Find OpenFace AU CSV files for each participant"""
        au_data_dir = self.data_path / "extracted_au_data"
        
        au_files = {}
        if au_data_dir.exists():
            for f in au_data_dir.glob("*.csv"):
                # Pattern: "300_OpenFace2.1.0_Pose_gaze_AUs.csv" -> "300"
                first_part = f.stem.split('_')[0]
                pid = ''.join(filter(str.isdigit, first_part))
                if pid:
                    au_files[pid] = f
        
        print(f"  Found {len(au_files)} AU data files")
        return au_files
    
    def _load_sequence(self, filepath):
        """Load features from a single CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Get all feature columns (AU + Pose + Gaze)
            all_features = self.au_features + self.pose_features + self.gaze_features
            
            # Use only columns that exist in the file
            available_cols = [c for c in all_features if c in df.columns]
            
            if not available_cols:
                return None
            
            features = df[available_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)
            
            return features
            
        except Exception as e:
            print(f"    Error loading {filepath}: {e}")
            return None
    
    def create_datasets(self, splits, participant_severity, au_files, scaling_factor=1.0):
        """Create train/dev/test datasets from splits, labels, and AU files"""
        datasets = {}
        
        for split_name, pids in splits.items():
            X_list = []
            y_list = []
            loaded = 0
            skipped = 0
            
            for pid in pids:
                if pid not in participant_severity or pid not in au_files:
                    skipped += 1
                    continue
                
                features = self._load_sequence(au_files[pid])
                if features is None:
                    skipped += 1
                    continue
                
                # Apply scaling factor
                if scaling_factor != 1.0:
                    features = features * scaling_factor
                
                X_list.append(features)
                y_list.append(participant_severity[pid])
                loaded += 1
            
            datasets[split_name] = {
                'X': X_list,
                'y': np.array(y_list) if y_list else np.array([])
            }
            
            print(f"  {split_name.upper()}: loaded={loaded}, skipped={skipped}")
        
        return datasets
    
    def load_features_and_labels(self, datasets, scaling_factor=1.0):
        """Pad/truncate sequences to fixed length and return processed data"""
        processed = {}
        
        for split_name, data in datasets.items():
            if len(data['y']) == 0:
                processed[split_name] = {'X': np.array([]), 'y': np.array([])}
                continue
            
            X_padded = []
            for seq in data['X']:
                seq_len = len(seq)
                n_features = seq.shape[1] if seq.ndim > 1 else 1
                
                if seq_len >= self.sequence_length:
                    # Uniform temporal sampling across entire interview
                    indices = np.linspace(0, seq_len - 1, self.sequence_length, dtype=int)
                    padded = seq[indices]
                else:
                    # Pad with zeros at the beginning
                    padding = np.zeros((self.sequence_length - seq_len, n_features))
                    padded = np.vstack([padding, seq])
                
                X_padded.append(padded)
            
            processed[split_name] = {
                'X': np.array(X_padded),
                'y': data['y']
            }
            
            print(f"  {split_name.upper()}: shape={processed[split_name]['X'].shape}, labels={np.bincount(data['y'])}")
        
        return processed
