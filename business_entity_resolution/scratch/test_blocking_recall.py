import sys
import polars as pl
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

s1 = pl.read_parquet('data/processed/sample/train_source1.parquet')
s2 = pl.read_parquet('data/processed/sample/train_source2.parquet')
s3 = pl.read_parquet('data/processed/sample/train_source3.parquet')
gt = pl.read_parquet('data/processed/sample/train_ground_truth.parquet')

s1_map = {row['entity_id']: row for row in s1.iter_rows(named=True)}
s2_map = {row['entity_id']: row for row in s2.iter_rows(named=True)}
s3_map = {row['entity_id']: row for row in s3.iter_rows(named=True)}

eval_pairs = []
for row in gt.filter(~pl.col('is_singleton')).iter_rows(named=True):
    s1_id = row['source1_entity_id']
    if s1_id not in s1_map: continue
    for tid in row['matched_entity_ids'].split(','):
        tid = tid.strip()
        if tid in s2_map:
            eval_pairs.append((s1_map[s1_id], s2_map[tid]))
        elif tid in s3_map:
            eval_pairs.append((s1_map[s1_id], s3_map[tid]))

total = len(eval_pairs)
print(f"Total true match pairs in sample: {total}")

STOPWORDS = {
    'private', 'limited', 'ltd', 'pvt', 'inc', 'incorporated', 'llc', 'corp', 'corporation',
    'company', 'co', 'services', 'enterprises', 'solutions', 'the', 'and', 'of', 'in', 'group',
    'holdings', 'technologies', 'india', 'international', 'global', 'trading', 'management'
}

def extract_blocking_keys(row):
    keys = set()
    country = row.get('country_norm') or ''
    if not country:
        return keys

    name = row.get('name_norm') or ''
    addr = row.get('clean_address') or ''
    city = row.get('city') or ''
    postal = row.get('postal_code') or ''

    # 1. Compact name prefix (strips spaces to bridge "high energy" and "highenergy")
    compact_name = name.replace(' ', '')
    if len(compact_name) >= 3:
        keys.add(f"cn_p3:{country}:{compact_name[:3]}")
    if len(compact_name) >= 4:
        keys.add(f"cn_p4:{country}:{compact_name[:4]}")

    # 2. Significant name tokens (ALL tokens >= 4 chars, or first 3 tokens >= 3 chars, not stopwords)
    tokens = [t for t in name.split() if len(t) >= 3 and t not in STOPWORDS]
    for t in tokens[:4]:
        keys.add(f"tok:{country}:{t}")

    # 3. Exact Postal Code + street number or postal + name prefix
    if postal and postal != 'None' and len(postal) >= 3:
        if len(compact_name) >= 2:
            keys.add(f"post_p2:{country}:{postal}:{compact_name[:2]}")

    # 4. City + Name prefix 2
    if city and city != 'None' and len(city) >= 3 and len(compact_name) >= 2:
        keys.add(f"city_p2:{country}:{city}:{compact_name[:2]}")

    # 5. Address tokens: rare words in address paired with country and city/postal
    # Find words in clean_address that are >= 5 chars (e.g. 'edisto', 'chanakya', 'dioro', 'rabindra')
    addr_words = [w for w in addr.split() if len(w) >= 5 and w not in STOPWORDS and not w.isdigit()]
    for aw in addr_words[:3]:
        keys.add(f"addr_word:{country}:{aw}")


    return keys

# Now let's test recall on all eval_pairs!
covered = 0
missed_pairs = []
for s, t in eval_pairs:
    s_keys = extract_blocking_keys(s)
    t_keys = extract_blocking_keys(t)
    overlap = s_keys & t_keys
    if overlap:
        covered += 1
    else:
        missed_pairs.append((s, t))

print(f"Covered: {covered}/{total} ({covered/total*100:.2f}%)")
print(f"Missed: {len(missed_pairs)}/{total} ({len(missed_pairs)/total*100:.2f}%)")

if missed_pairs:
    print("\n--- Sample Misses ---")
    for i, (s, t) in enumerate(missed_pairs[:10]):
        print(f"[{i+1}]")
        print(f"  S1: '{s['name_norm']}' | Addr: '{s['clean_address']}' | City: '{s['city']}' | Post: '{s['postal_code']}'")
        print(f"  Tg: '{t['name_norm']}' | Addr: '{t['clean_address']}' | City: '{t['city']}' | Post: '{t['postal_code']}'")
