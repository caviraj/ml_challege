import pandas as pd
import random

train_dir = 'student_resource/dataset/train'

print('--- Source 1 ---')
s1 = pd.read_csv(f'{train_dir}/train_source1.tsv', sep='\t', dtype=str)
print('Shape:', s1.shape)
print('Nulls:\n', s1.isnull().sum())
print('Prefix check (all S1-?):', s1['entity_id'].str.startswith('S1-').all())

print('--- Ground Truth ---')
gt = pd.read_csv(f'{train_dir}/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
print('Shape:', gt.shape)
print('Nulls:\n', gt.isnull().sum())
print('Prefix check (all S1-?):', gt['source1_entity_id'].str.startswith('S1-').all())

random.seed(42)
non_empty = gt[gt['matched_entity_ids'] != ''].sample(15, random_state=42)

s1_indexed = s1.set_index('entity_id')

sampled_pairs = []
s2_needed = set()
s3_needed = set()

for _, row in non_empty.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = row['matched_entity_ids'].split(',')
    chosen_m = random.choice(m_ids)
    sampled_pairs.append((s1_id, chosen_m))
    if chosen_m.startswith('S2-'):
        s2_needed.add(chosen_m)
    elif chosen_m.startswith('S3-'):
        s3_needed.add(chosen_m)

print(f'Fetching {len(s2_needed)} matches from S2, {len(s3_needed)} from S3...')

s2_matches = {}
for chunk in pd.read_csv(f'{train_dir}/train_source2.tsv', sep='\t', dtype=str, chunksize=500000):
    subset = chunk[chunk['entity_id'].isin(s2_needed)]
    for _, r in subset.iterrows():
        s2_matches[r['entity_id']] = r.to_dict()
    if len(s2_matches) == len(s2_needed):
        break

s3_matches = {}
for chunk in pd.read_csv(f'{train_dir}/train_source3.tsv', sep='\t', dtype=str, chunksize=500000):
    subset = chunk[chunk['entity_id'].isin(s3_needed)]
    for _, r in subset.iterrows():
        s3_matches[r['entity_id']] = r.to_dict()
    if len(s3_matches) == len(s3_needed):
        break

import json

output_pairs = []
with open('sampled_pairs.txt', 'w', encoding='utf-8') as f:
    for i, (s1_id, m_id) in enumerate(sampled_pairs, 1):
        s1_row = s1_indexed.loc[s1_id]
        m_row = s2_matches.get(m_id) or s3_matches.get(m_id, {})
        entry = {
            "pair_idx": i,
            "s1_id": s1_id,
            "s1_country": s1_row['country'],
            "s1_name": s1_row['business_name'],
            "s1_address": s1_row['business_address'],
            "match_id": m_id,
            "match_country": m_row.get('country'),
            "match_name": m_row.get('business_name'),
            "match_address": m_row.get('business_address')
        }
        output_pairs.append(entry)
        line = f"""Pair {i}:
  S1 ({s1_id}) [Country: {s1_row['country']}]:
    Name:    {s1_row['business_name']}
    Address: {s1_row['business_address']}
  Match ({m_id}) [Country: {m_row.get('country')}]:
    Name:    {m_row.get('business_name')}
    Address: {m_row.get('business_address')}
------------------------------------------------------------
"""
        f.write(line)
        try:
            print(line.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

with open('sampled_pairs.json', 'w', encoding='utf-8') as jf:
    json.dump(output_pairs, jf, indent=2, ensure_ascii=False)
print("Saved 15 sampled pairs to sampled_pairs.txt and sampled_pairs.json")
