import sys
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

# Load sample data
s1 = pl.read_parquet('data/processed/sample/train_source1.parquet')
s2 = pl.read_parquet('data/processed/sample/train_source2.parquet')
s3 = pl.read_parquet('data/processed/sample/train_source3.parquet')
gt = pl.read_parquet('data/processed/sample/train_ground_truth.parquet')

s1_map = {row['entity_id']: row for row in s1.iter_rows(named=True)}
s2_map = {row['entity_id']: row for row in s2.iter_rows(named=True)}
s3_map = {row['entity_id']: row for row in s3.iter_rows(named=True)}

# Find all true pairs where BOTH s1 and target are present in our sample
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

total_eval = len(eval_pairs)
print(f"Total true pairs in sample: {total_eval}")

STOPWORDS = {
    'private', 'limited', 'ltd', 'pvt', 'inc', 'incorporated', 'llc', 'corp', 'corporation',
    'company', 'co', 'services', 'enterprises', 'solutions', 'the', 'and', 'of', 'in', 'group',
    'holdings', 'technologies', 'india', 'international', 'global', 'trading', 'management',
    'enterprise', 'technology', 'associates', 'consulting', 'industries', 'ventures',
    'llp', 'center', 'partners', 'care', 'systems',
    'लिमिटेड', 'प्राइवेट', 'कंपनी', 'सेवाएं', 'इंडिया'
}

ADDR_STOP = {
    'street', 'road', 'floor', 'suite', 'block', 'near', 'opp', 'opposite', 'avenue',
    'nagar', 'delhi', 'texas', 'mumbai', 'california', 'door', 'number', 'plot',
    'flat', 'shop', 'room', 'building', 'phase', 'sector', 'layout', 'lane', 'cross',
    'main', 'state', 'india', 'bengal', 'gujarat', 'maharashtra', 'null'
}

def extract_blocking_keys(row):
    keys = set()
    country = row.get('country_norm') or ''
    if not country: return keys

    name = row.get('name_norm') or ''
    addr = row.get('clean_address') or ''
    city = row.get('city') or ''
    postal = row.get('postal_code') or ''

    # Clean leading prefixes
    clean_name = name
    for pfx in ['the ', 'a ', 'an ', 'shri ', 'smt ', 'dr ', 'm/s ']:
        if clean_name.startswith(pfx):
            clean_name = clean_name[len(pfx):]
            break

    # 1. Compact name prefix: 3-char and 4-char
    compact_name = clean_name.replace(' ', '')
    if len(compact_name) >= 3:
        keys.add(f"cn_p3:{country}:{compact_name[:3]}")
    if len(compact_name) >= 4:
        keys.add(f"cn_p4:{country}:{compact_name[:4]}")

    # 2. Significant name tokens (tokens >= 3 chars, not in STOPWORDS)
    tokens = [t for t in name.split() if len(t) >= 3 and t not in STOPWORDS]
    for t in tokens[:4]:
        keys.add(f"tok:{country}:{t}")

    if clean_name != name:
        for t in [t for t in clean_name.split() if len(t) >= 3 and t not in STOPWORDS][:4]:
            keys.add(f"tok:{country}:{t}")

    # 3. Address tokens >= 4 chars, not numbers, not generic
    addr_tokens = [t for t in addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOP]
    for at in addr_tokens[:4]:
        keys.add(f"addr_tok:{country}:{at}")

    # 4. Exact Postal Code + 2-char Name Prefix
    if postal and postal != 'None' and len(postal) >= 3:
        if len(compact_name) >= 2:
            keys.add(f"post_p2:{country}:{postal}:{compact_name[:2]}")

    # 5. City + 2-char Name Prefix
    if city and city != 'None' and len(city) >= 3 and len(compact_name) >= 2:
        keys.add(f"city_p2:{country}:{city}:{compact_name[:2]}")

    # 6. City + Street Number
    nums = [t for t in addr.split() if t.isdigit() and 2 <= len(t) <= 4]
    if city and city != 'None' and len(city) >= 3 and nums:
        for n in nums[:2]:
            keys.add(f"city_num:{country}:{city}:{n}")

    return keys

# Build inverted index
targets = []
for row in s2.iter_rows(named=True):
    targets.append((row['entity_id'], 2, row['name_norm'], row['clean_address'], row['city'], row['postal_code'], row['country_norm']))
for row in s3.iter_rows(named=True):
    targets.append((row['entity_id'], 3, row['name_norm'], row['clean_address'], row['city'], row['postal_code'], row['country_norm']))

inv_index = defaultdict(list)
for idx, t in enumerate(targets):
    r = {'country_norm': t[6], 'name_norm': t[2], 'clean_address': t[3], 'city': t[4], 'postal_code': t[5]}
    for k in extract_blocking_keys(r):
        inv_index[k].append(idx)

print(f"Indexed {len(targets)} targets across {len(inv_index)} keys.")
top_keys = sorted(inv_index.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("Top 10 most frequent keys:")
for k, v in top_keys:
    print(f"  {k}: {len(v)} targets")

# Evaluate recall on eval_pairs
MAX_POSTLIST_SIZE = 500  # Cap key postlist size

k_values = [5, 10, 15, 20, 25, 30]
hits_at_k = {k: 0 for k in k_values}
cand_counts = []
raw_covered = 0

s1_eval_map = defaultdict(set)
for s, t in eval_pairs:
    s1_eval_map[s['entity_id']].add(t['entity_id'])

for s1_id, true_targets in s1_eval_map.items():
    s1_row = s1_map[s1_id]
    s1_keys = extract_blocking_keys(s1_row)
    
    cand_indices = set()
    for k in s1_keys:
        posting = inv_index.get(k)
        if posting:
            if len(posting) <= MAX_POSTLIST_SIZE:
                cand_indices.update(posting)
            else:
                cand_indices.update(posting[:100])
                
    cand_counts.append(len(cand_indices))
    
    # Check raw coverage
    retrieved_tids = {targets[i][0] for i in cand_indices}
    raw_hits = len(true_targets & retrieved_tids)
    raw_covered += raw_hits
    
    # Score & rank candidates
    s1_name = s1_row['name_norm'] or ''
    s1_addr = s1_row['clean_address'] or ''
    
    scored_cands = []
    for c_idx in cand_indices:
        t = targets[c_idx]
        t_id = t[0]
        t_name = t[2] or ''
        t_addr = t[3] or ''
        
        # Name similarities
        name_sim = fuzz.ratio(s1_name, t_name)
        ts_sim = fuzz.token_sort_ratio(s1_name, t_name)
        n_score = max(name_sim, ts_sim)
        
        # Address token set similarity (handles out-of-order and partial address matches)
        addr_sim = fuzz.token_set_ratio(s1_addr, t_addr) if s1_addr and t_addr else 0
        
        # Combined ranking score: 70% name + 30% address, or 100% address if name score very low (cross-script)
        combined_score = max(0.70 * n_score + 0.30 * addr_sim, 0.50 * addr_sim + 0.50 * n_score)
        if n_score < 30 and addr_sim > 60:
            combined_score = addr_sim  # boost cross-script address matches
            
        scored_cands.append((t_id, combined_score))
        
    scored_cands.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [c[0] for c in scored_cands]
    
    for k in k_values:
        top_k = set(ranked_ids[:k])
        hits_at_k[k] += len(true_targets & top_k)

print(f"\nRaw candidate coverage: {raw_covered}/{total_eval} ({raw_covered/total_eval*100:.2f}%)")
print("\n--- Recall at K ---")
for k in k_values:
    rec = hits_at_k[k] / total_eval * 100
    print(f"Top-{k:2d}: {hits_at_k[k]}/{total_eval} hits ({rec:.2f}%)")

print(f"\nAverage candidates per S1 entity: {sum(cand_counts)/len(cand_counts):.1f}")
print(f"Median candidates: {sorted(cand_counts)[len(cand_counts)//2]}")
print(f"Max candidates: {max(cand_counts)}")
