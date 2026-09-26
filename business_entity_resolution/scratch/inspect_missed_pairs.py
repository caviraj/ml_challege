import sys
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz

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

STOPWORDS = {
    'private', 'limited', 'ltd', 'pvt', 'inc', 'incorporated', 'llc', 'corp', 'corporation',
    'company', 'co', 'services', 'enterprises', 'solutions', 'the', 'and', 'of', 'in', 'group',
    'holdings', 'technologies', 'india', 'international', 'global', 'trading', 'management',
    'enterprise', 'technology', 'associates', 'consulting', 'industries', 'ventures',
    'llp', 'center', 'partners', 'care', 'systems',
    'लिमिटेड', 'प्राइवेट', 'कंपनी', 'सेवाएं', 'इंडिया'
}

def extract_blocking_keys(row):
    keys = set()
    country = row.get('country_norm') or ''
    if not country: return keys

    name = row.get('name_norm') or ''
    addr = row.get('clean_address') or ''
    city = row.get('city') or ''
    postal = row.get('postal_code') or ''

    # Remove leading stopwords like "the ", "shri ", "dr "
    clean_name = name
    for pfx in ['the ', 'a ', 'an ', 'shri ', 'smt ', 'dr ', 'm/s ']:
        if clean_name.startswith(pfx):
            clean_name = clean_name[len(pfx):]
            break

    compact_name = clean_name.replace(' ', '')
    if len(compact_name) >= 3:
        keys.add(f"cn_p3:{country}:{compact_name[:3]}")
    if len(compact_name) >= 4:
        keys.add(f"cn_p4:{country}:{compact_name[:4]}")

    tokens = [t for t in name.split() if len(t) >= 3 and t not in STOPWORDS]
    for t in tokens[:5]:
        keys.add(f"tok:{country}:{t}")

    # Also extract tokens from clean_name if different
    if clean_name != name:
        for t in [t for t in clean_name.split() if len(t) >= 3 and t not in STOPWORDS][:5]:
            keys.add(f"tok:{country}:{t}")

    # Address tokens >= 4 chars, not numbers, not generic
    addr_stop = {
        'street', 'road', 'floor', 'suite', 'block', 'near', 'opp', 'opposite', 'avenue',
        'nagar', 'delhi', 'texas', 'mumbai', 'california', 'door', 'number', 'plot',
        'flat', 'shop', 'room', 'building', 'phase', 'sector', 'layout', 'lane', 'cross',
        'main', 'state', 'india', 'bengal', 'gujarat', 'maharashtra', 'null'
    }
    addr_tokens = [t for t in addr.split() if len(t) >= 4 and not t.isdigit() and t not in addr_stop]
    for at in addr_tokens[:4]:
        keys.add(f"addr_tok:{country}:{at}")

    # Exact Postal Code + 2-char Name Prefix
    if postal and postal != 'None' and len(postal) >= 3:
        keys.add(f"post:{country}:{postal}")
        if len(compact_name) >= 2:
            keys.add(f"post_p2:{country}:{postal}:{compact_name[:2]}")

    # City + 2-char Name Prefix
    if city and city != 'None' and len(city) >= 3 and len(compact_name) >= 2:
        keys.add(f"city_p2:{country}:{city}:{compact_name[:2]}")

    # City + Street Number (if any 2-4 digit number in address)
    nums = [t for t in addr.split() if t.isdigit() and 2 <= len(t) <= 4]
    if city and city != 'None' and len(city) >= 3 and nums:
        for n in nums[:2]:
            keys.add(f"city_num:{country}:{city}:{n}")

    return keys

# Check key intersection for each eval pair
missed = []
hit_count = 0
for s, t in eval_pairs:
    sk = extract_blocking_keys(s)
    tk = extract_blocking_keys(t)
    inter = sk & tk
    if inter:
        hit_count += 1
    else:
        missed.append((s, t))

print(f"Key overlap hits: {hit_count}/{len(eval_pairs)} ({hit_count/len(eval_pairs)*100:.2f}%)")
print(f"Missed: {len(missed)}")
print("\nFirst 10 missed pairs:")
for s, t in missed[:10]:
    print(f"S1: name='{s['name_norm']}' | addr='{s['clean_address']}' | city='{s['city']}' | postal='{s['postal_code']}'")
    print(f"T : name='{t['name_norm']}' | addr='{t['clean_address']}' | city='{t['city']}' | postal='{t['postal_code']}'")
    print("-" * 50)
