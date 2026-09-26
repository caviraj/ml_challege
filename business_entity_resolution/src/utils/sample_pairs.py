import pandas as pd
from pathlib import Path
import random

random.seed(42)

train_dir = Path('student_resource/dataset/train')
print("Loading datasets...")
df_s1 = pd.read_csv(train_dir / 'train_source1.tsv', sep='\t')
df_s2 = pd.read_csv(train_dir / 'train_source2.tsv', sep='\t')
df_s3 = pd.read_csv(train_dir / 'train_source3.tsv', sep='\t')
gt = pd.read_csv(train_dir / 'train_ground_truth.tsv', sep='\t')

valid_gt = gt.dropna(subset=['matched_entity_ids']).copy()
sample_gt = valid_gt.sample(n=15, random_state=42)

s1_lookup = df_s1.set_index('entity_id')
s2_lookup = df_s2.set_index('entity_id')
s3_lookup = df_s3.set_index('entity_id')

print("=== 15 Matched Pairs Sample ===")
for idx, (_, row) in enumerate(sample_gt.iterrows(), 1):
    s1_id = row['source1_entity_id']
    matches = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
    target_id = random.choice(matches)
    
    s1_data = s1_lookup.loc[s1_id]
    if target_id.startswith('S2-'):
        target_data = s2_lookup.loc[target_id]
        src_label = 'Source 2'
    else:
        target_data = s3_lookup.loc[target_id]
        src_label = 'Source 3'
        
    print(f"Pair {idx}:")
    print(f"  [S1: {s1_id}] ({s1_data['country']})")
    print(f"    Name   : {s1_data['business_name']}")
    print(f"    Address: {s1_data['business_address']}")
    print(f"  [{src_label}: {target_id}] ({target_data['country']})")
    print(f"    Name   : {target_data['business_name']}")
    print(f"    Address: {target_data['business_address']}")
    print("-" * 70)
