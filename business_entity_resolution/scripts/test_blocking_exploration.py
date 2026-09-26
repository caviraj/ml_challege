"""Exploration script to analyze true match patterns and evaluate blocking keys."""

import polars as pl
import re

def main():
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
        if s1_id not in s1_map:
            continue
        for tid in row['matched_entity_ids'].split(','):
            tid = tid.strip()
            if tid in s2_map:
                eval_pairs.append((s1_map[s1_id], s2_map[tid]))
            elif tid in s3_map:
                eval_pairs.append((s1_map[s1_id], s3_map[tid]))

    print(f"Total evaluatable true match pairs in sample: {len(eval_pairs)}")

    # Inspect first 10 pairs
    for i, (src, tgt) in enumerate(eval_pairs[:10]):
        print(f"\n--- Pair {i+1} ---")
        print(f"  S1 [{src['source']}]: name='{src['name_norm']}' | addr='{src['clean_address']}' | city='{src['city']}' | post='{src['postal_code']}' | ctry='{src['country_norm']}'")
        print(f"  Tg [{tgt['source']}]: name='{tgt['name_norm']}' | addr='{tgt['clean_address']}' | city='{tgt['city']}' | post='{tgt['postal_code']}' | ctry='{tgt['country_norm']}'")

    # Analyze attribute agreement rates
    same_country = sum(1 for s, t in eval_pairs if s['country_norm'] == t['country_norm'])
    same_name_exact = sum(1 for s, t in eval_pairs if s['name_norm'] and s['name_norm'] == t['name_norm'])
    prefix3_match = sum(1 for s, t in eval_pairs if s['name_norm'][:3] and s['name_norm'][:3] == t['name_norm'][:3])
    prefix4_match = sum(1 for s, t in eval_pairs if s['name_norm'][:4] and s['name_norm'][:4] == t['name_norm'][:4])
    same_post = sum(1 for s, t in eval_pairs if s['postal_code'] and s['postal_code'] == t['postal_code'])
    same_city = sum(1 for s, t in eval_pairs if s['city'] and s['city'] == t['city'])
    
    # First token match
    def first_tok(n):
        parts = n.split()
        return parts[0] if parts else ""
    
    first_tok_match = sum(1 for s, t in eval_pairs if first_tok(s['name_norm']) and first_tok(s['name_norm']) == first_tok(t['name_norm']))

    # Any token overlap in name
    def token_overlap(s_name, t_name):
        s_toks = set(s_name.split())
        t_toks = set(t_name.split())
        return bool(s_toks & t_toks)

    name_tok_overlap = sum(1 for s, t in eval_pairs if token_overlap(s['name_norm'], t['name_norm']))

    n = len(eval_pairs)
    print("\n=== Agreement Statistics on Ground Truth Matches ===")
    print(f"Same Country:         {same_country}/{n} ({same_country/n*100:.2f}%)")
    print(f"Exact Name Match:     {same_name_exact}/{n} ({same_name_exact/n*100:.2f}%)")
    print(f"Name Prefix 3 Match:  {prefix3_match}/{n} ({prefix3_match/n*100:.2f}%)")
    print(f"Name Prefix 4 Match:  {prefix4_match}/{n} ({prefix4_match/n*100:.2f}%)")
    print(f"Name First Token:     {first_tok_match}/{n} ({first_tok_match/n*100:.2f}%)")
    print(f"Name Token Overlap:   {name_tok_overlap}/{n} ({name_tok_overlap/n*100:.2f}%)")
    print(f"Exact Postal Match:   {same_post}/{n} ({same_post/n*100:.2f}%)")
    print(f"Exact City Match:     {same_city}/{n} ({same_city/n*100:.2f}%)")

if __name__ == '__main__':
    main()
