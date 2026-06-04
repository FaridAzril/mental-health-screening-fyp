import sys
sys.path.append('scripts')
from data_processor_2026 import EDAICDataProcessor

dp = EDAICDataProcessor('data/edaic')
splits = dp.load_split_files()
sev, _ = dp.load_detailed_labels()
au = dp.get_participant_files()

test_pids = splits['test'][:5]
print(f"\nTest PIDs sample: {test_pids}")
print(f"Types: {[type(p) for p in test_pids]}")

print(f"\nSeverity keys sample: {list(sev.keys())[:5]}")
print(f"AU keys sample: {list(au.keys())[:5]}")

for pid in test_pids:
    in_sev = pid in sev
    in_au = pid in au
    print(f"  PID '{pid}': in severity={in_sev}, in au_files={in_au}")
