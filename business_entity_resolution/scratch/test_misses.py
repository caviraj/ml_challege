import sys
import polars as pl
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

print(f'Total eval pairs: {len(eval_pairs)}')
misses = []
for s, t in eval_pairs:
    s_toks = set(s['name_norm'].split())
    t_toks = set(t['name_norm'].split())
    if not (s_toks & t_toks):
        misses.append((s, t))

print(f'Pairs without any common name token: {len(misses)} ({len(misses)/len(eval_pairs)*100:.2f}%)')
for i, (s, t) in enumerate(misses[:20]):
    print(f"[{i+1}]")
    print(f"  S1: '{s['name_norm']}' | Addr: '{s['clean_address']}' | City: '{s['city']}' | Post: '{s['postal_code']}'")
    print(f"  Tg: '{t['name_norm']}' | Addr: '{t['clean_address']}' | City: '{t['city']}' | Post: '{t['postal_code']}'")
