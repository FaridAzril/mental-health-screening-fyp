import pandas as pd
import numpy as np

df = pd.read_csv('data/edaic/detailed_lables.csv')

# Replicate data_processor_2026.py logic
id_col = None
score_col = None

for col in df.columns:
    col_lower = col.lower().strip()
    if 'participant' in col_lower or col_lower == 'id':
        id_col = col
    if 'total' in col_lower or col_lower == 'phq_score' or col_lower == 'phq8_score':
        score_col = col

print(f"id_col: {id_col}")
print(f"score_col: {score_col}")

if score_col is None:
    phq_item_cols = [c for c in df.columns if 'phq' in c.lower() and c != id_col]
    print(f"PHQ item cols: {phq_item_cols}")
    if len(phq_item_cols) >= 8:
        df['_phq8_total'] = df[phq_item_cols[:8]].sum(axis=1)
        score_col = '_phq8_total'
        print(f"Computed PHQ8 total, using score_col = {score_col}")

# Now replicate the severity mapping
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
    if score <= 9:
        severity = 0
    elif score <= 14:
        severity = 1
    else:
        severity = 2
    
    participant_severity[pid] = severity

print(f"\nTotal loaded: {len(participant_severity)}")
print(f"Total rows in CSV: {len(df)}")

# Check which are missing
all_pids = set(str(int(row[id_col])) for _, row in df.iterrows())
loaded_pids = set(participant_severity.keys())
missing = all_pids - loaded_pids
print(f"Missing PIDs: {len(missing)}")
if missing:
    print(f"Sample missing: {list(missing)[:10]}")

# Check test specifically
test_df = pd.read_csv('data/edaic/test_split.csv')
test_pids = [str(int(x)) for x in test_df.iloc[:, 0]]
test_in_sev = [p for p in test_pids if p in participant_severity]
test_not_in_sev = [p for p in test_pids if p not in participant_severity]
print(f"\nTest PIDs in severity: {len(test_in_sev)}/56")
print(f"Test PIDs NOT in severity: {len(test_not_in_sev)}")
if test_not_in_sev:
    print(f"Missing test PIDs: {test_not_in_sev[:10]}")
