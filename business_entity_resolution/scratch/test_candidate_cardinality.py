import sys
import time
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

# Load sample data
s1 = pl.read_parquet('data/processed/sample/train_source1.parquet')
s2 = pl.read_parquet('data/processed/sample/train_source2.parquet')
s3 = pl.read_parquet('data/processed/sample/train_source3.parquet')
gt = pl.read_parquet('data/processed/sample/train_ground_truth.parquet')

STOPWORDS = {
    # English corporate
    'private', 'limited', 'ltd', 'pvt', 'inc', 'incorporated', 'llc', 'corp', 'corporation',
    'company', 'co', 'services', 'enterprises', 'solutions', 'the', 'and', 'of', 'in', 'group',
    'holdings', 'technologies', 'india', 'international', 'global', 'trading', 'management',
    'enterprise', 'technology', 'associates', 'consulting', 'industries', 'ventures',
    # Address stopwords
    'street', 'drive', 'avenue', 'court', 'number', 'floor', 'road', 'suite', 'lane',
    'building', 'block', 'phase', 'sector', 'nagar', 'cross', 'main', 'near', 'opp',
    'bazaar', 'marg', 'post', 'dist', 'state', 'null',
    # Indic script legal terms
    'प्राइवेट', 'लिमिटेड', 'कंपनी', 'इण्डिया', 'इंडिया'
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
    addr_words = [w for w in addr.split() if len(w) >= 5 and w not in STOPWORDS and not w.isdigit()]
    for aw in addr_words[:3]:
        keys.add(f"addr_word:{country}:{aw}")

    return keys

print("Building inverted index for S2 and S3...")
t0 = time.time()
# Inverted index: key -> list of (target_id, source_type, name_norm, clean_address, city, postal_code)
# Or compact index storing row indices into an array
targets = []
# target tuple: (entity_id, source_type, name_norm, clean_address, city, postal_code)
for row in s2.iter_rows(named=True):
    targets.append((row['entity_id'], 2, row['name_norm'], row['clean_address'], row['city'], row['postal_code'], row['country_norm']))
for row in s3.iter_rows(named=True):
    targets.append((row['entity_id'], 3, row['name_norm'], row['clean_address'], row['city'], row['postal_code'], row['country_norm']))

inv_index = defaultdict(list)
for idx, t in enumerate(targets):
    # build row dict for extract_blocking_keys
    r = {'country_norm': t[6], 'name_norm': t[2], 'clean_address': t[3], 'city': t[4], 'postal_code': t[5]}
    for k in extract_blocking_keys(r):
        inv_index[k].append(idx)

t1 = time.time()
print(f"Indexed {len(targets)} targets across {len(inv_index)} keys in {t1-t0:.2f}s")

# Let's inspect posting list lengths
lengths = [len(v) for v in inv_index.values()]
print(f"Posting list lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
# Find top 10 most common keys
top_keys = sorted(inv_index.items(), key=lambda x: len(x[1]), reverse=True)[:10]
print("Top 10 most frequent keys:")
for k, v in top_keys:
    print(f"  {k}: {len(v)} targets")

# Now let's test candidate retrieval and recall at top-K for true ground truth pairs
gt_active = gt.filter(~pl.col('is_singleton'))
s1_map = {row['entity_id']: row for row in s1.iter_rows(named=True)}

# Map s1_id -> set of true target IDs
s1_to_true_targets = defaultdict(set)
total_pairs = 0
for row in gt_active.iter_rows(named=True):
    s1_id = row['source1_entity_id']
    if s1_id not in s1_map: continue
    for tid in row['matched_entity_ids'].split(','):
        tid = tid.strip()
        s1_to_true_targets[s1_id].add(tid)
        total_pairs += 1

print(f"Evaluating on {len(s1_to_true_targets)} non-singleton S1 entities ({total_pairs} true pairs)...")

# Cap postlist size if key is too frequent (e.g. > 500)
MAX_POSTLIST_SIZE = 1000

k_values = [5, 10, 15, 20, 25, 30, 50, 100]
hits_at_k = {k: 0 for k in k_values}
total_candidates_retrieved = []

for s1_id, true_targets in s1_to_true_targets.items():
    s1_row = s1_map[s1_id]
    s1_keys = extract_blocking_keys(s1_row)
    
    # Collect candidate indices
    cand_indices = set()
    for k in s1_keys:
        posting = inv_index.get(k)
        if posting:
            if len(posting) <= MAX_POSTLIST_SIZE:
                cand_indices.update(posting)
            else:
                # Still take a sample or skip if too generic
                pass
                
    total_candidates_retrieved.append(len(cand_indices))
    
    # Rank candidates by fast similarity
    s1_name = s1_row['name_norm'] or ''
    s1_addr = s1_row['clean_address'] or ''
    
    scored_cands = []
    for c_idx in cand_indices:
        t = targets[c_idx]
        t_id = t[0]
        t_name = t[2] or ''
        # RapidFuzz ratio is fast
        score = fuzz.ratio(s1_name, t_name)
        # If score is very low, check address similarity as secondary
        if score < 30 and s1_addr and t[3]:
            addr_score = fuzz.ratio(s1_addr, t[3])
            score = max(score, addr_score * 0.8)
        scored_cands.append((t_id, score))
    
    scored_cands.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [c[0] for c in scored_cands]
    
    for k in k_values:
        top_k = set(ranked_ids[:k])
        hits = len(true_targets & top_k)
        hits_at_k[k] += hits

print("\n--- Recall at K ---")
for k in k_values:
    rec = hits_at_k[k] / total_pairs * 100
    print(f"Top-{k:3d}: {hits_at_k[k]}/{total_pairs} hits ({rec:.2f}%)")

avg_cand = sum(total_candidates_retrieved) / len(total_candidates_retrieved)
print(f"\nAverage candidates per S1 entity before top-K: {avg_cand:.1f}")
print(f"Median candidates: {sorted(total_candidates_retrieved)[len(total_candidates_retrieved)//2]}")
print(f"Max candidates: {max(total_candidates_retrieved)}")
