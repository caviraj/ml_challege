"""
Inference pipeline for business entity resolution.
Generates candidate pairs from test parquets, computes pairwise features,
evaluates LightGBM classification probabilities, applies the optimal threshold,
and writes the required submission files:
- output/matching_results.tsv
- output/candidate_pairs.tsv
"""

import os
import gc
import logging
import argparse
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl
import joblib
from tqdm import tqdm

from src.blocking.indexer import CountryPartitionedIndex
from src.blocking.candidate_generator import CandidateGenerator
from src.features.pairwise_features import FEATURE_NAMES, extract_pair_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("predict")


def run_inference(
    model_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_dir: str,
    threshold_override: float = None,
    max_candidates: int = 15,
    min_score: float = 35.0,
    chunk_size: int = 5_000
):
    """
    Runs country-partitioned inference over test sources, producing submission TSVs.
    """
    logger.info(f"Loading model artifact from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact['model']
    best_threshold = artifact.get('best_threshold', 0.70)
    decision_threshold = threshold_override if threshold_override is not None else best_threshold

    logger.info(f"Using decision threshold: {decision_threshold:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Read Source 1 test records
    logger.info(f"Loading Source 1 test data from {s1_path}...")
    s1_df = pl.read_parquet(s1_path)
    total_s1 = len(s1_df)
    logger.info(f"Total Source 1 entities to predict: {total_s1:,}")

    # Prepare output TSV files and write headers
    with open(matching_path, "w", encoding="utf-8") as f_m, open(candidate_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    # Get country breakdown to process country by country (optimizes RAM)
    countries = s1_df['country_norm'].fill_null('unknown').unique().to_list()
    logger.info(f"Processing across {len(countries)} unique countries...")

    total_matches_found = 0
    total_singletons = 0
    total_candidates_generated = 0

    for country in countries:
        s1_c_df = s1_df.filter(pl.col('country_norm').fill_null('unknown') == country)
        n_c = len(s1_c_df)
        if n_c == 0:
            continue

        logger.info(f"--- Country: {country} ({n_c:,} Source 1 records) ---")

        # Build index for this country only
        index = CountryPartitionedIndex(max_postlist_size=300)
        c_filter = None if country == 'unknown' else country
        n_s2 = index.add_records_from_parquet(s2_path, source_id=2, filter_country=c_filter)
        n_s3 = index.add_records_from_parquet(s3_path, source_id=3, filter_country=c_filter)

        logger.info(f"Targets indexed for {country}: S2={n_s2:,}, S3={n_s3:,}")

        generator = CandidateGenerator(index)
        s1_rows = s1_c_df.to_dicts()

        # If no targets exist for this country, all are singletons
        if n_s2 + n_s3 == 0:
            with open(matching_path, "a", encoding="utf-8") as f_m, open(candidate_path, "a", encoding="utf-8") as f_c:
                for row in s1_rows:
                    sid = row['entity_id']
                    f_m.write(f"{sid}\t\n")
                    f_c.write(f"{sid}\t\n")
                    total_singletons += 1
            del index, generator
            gc.collect()
            continue

        # Process in batches
        matching_lines = []
        candidate_lines = []

        for i in range(0, len(s1_rows), chunk_size):
            chunk = s1_rows[i:i + chunk_size]
            pairs_to_score: List[Dict[str, Any]] = []
            pair_meta: List[Tuple[str, str]] = []  # (s1_id, target_id)
            entity_cand_map: Dict[str, List[str]] = {r['entity_id']: [] for r in chunk}

            for row in chunk:
                s1_id = row['entity_id']
                cands = generator.generate_candidates_for_record(
                    row,
                    max_candidates=max_candidates,
                    min_score=min_score
                )

                for t_id, s_id, score, t_rec in cands:
                    entity_cand_map[s1_id].append(t_id)
                    pair_meta.append((s1_id, t_id))
                    pair_dict = {
                        's1_name': row.get('name_norm') or '',
                        's1_address': row.get('clean_address') or '',
                        's1_city': row.get('city') or '',
                        's1_postal': row.get('postal_code') or '',
                        's1_country': row.get('country_norm') or '',
                        'target_name': t_rec[2] or '',
                        'target_address': t_rec[3] or '',
                        'target_city': t_rec[4] or '',
                        'target_postal': t_rec[5] or '',
                        'target_country': t_rec[6] or '',
                        'source_id': s_id,
                        'blocking_score': score
                    }
                    pairs_to_score.append(pair_dict)

            # Score pairs if any candidates generated
            matched_pairs_by_s1: Dict[str, List[str]] = {r['entity_id']: [] for r in chunk}

            if pairs_to_score:
                X_chunk = np.empty((len(pairs_to_score), len(FEATURE_NAMES)), dtype=np.float32)
                for p_idx, pair in enumerate(pairs_to_score):
                    X_chunk[p_idx, :] = extract_pair_features(pair)

                probs = model.predict_proba(X_chunk)[:, 1]

                for p_idx, (s1_id, t_id) in enumerate(pair_meta):
                    if probs[p_idx] >= decision_threshold:
                        matched_pairs_by_s1[s1_id].append(t_id)

            # Format output lines
            for row in chunk:
                sid = row['entity_id']
                cand_ids = entity_cand_map.get(sid, [])
                match_ids = matched_pairs_by_s1.get(sid, [])

                # Deduplicate while preserving order
                seen_c = set()
                uniq_c = [c for c in cand_ids if not (c in seen_c or seen_c.add(c))]
                seen_m = set()
                uniq_m = [m for m in match_ids if not (m in seen_m or seen_m.add(m))]

                cand_str = ",".join(uniq_c)
                match_str = ",".join(uniq_m)

                candidate_lines.append(f"{sid}\t{cand_str}\n")
                matching_lines.append(f"{sid}\t{match_str}\n")

                if uniq_m:
                    total_matches_found += 1
                else:
                    total_singletons += 1
                total_candidates_generated += len(uniq_c)

            del pairs_to_score, pair_meta, entity_cand_map, matched_pairs_by_s1
            gc.collect()

        # Write buffered lines to disk
        with open(matching_path, "a", encoding="utf-8") as f_m, open(candidate_path, "a", encoding="utf-8") as f_c:
            f_m.writelines(matching_lines)
            f_c.writelines(candidate_lines)

        del index, generator, s1_rows, matching_lines, candidate_lines
        gc.collect()

    logger.info("=" * 60)
    logger.info(f"Inference complete!")
    logger.info(f"Total Source 1 entities processed: {total_s1:,}")
    logger.info(f"Entities with predicted matches: {total_matches_found:,} ({(total_matches_found / max(1, total_s1)) * 100:.1f}%)")
    logger.info(f"Entities predicted as singletons: {total_singletons:,} ({(total_singletons / max(1, total_s1)) * 100:.1f}%)")
    logger.info(f"Output files written:")
    logger.info(f"  - {matching_path}")
    logger.info(f"  - {candidate_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate Submission TSVs for Entity Resolution")
    parser.add_argument("--model-path", default="models/lgb_model.joblib", help="Path to trained model artifact")
    parser.add_argument("--test-s1", default="data/processed/test_source1.parquet", help="Test Source 1 parquet")
    parser.add_argument("--test-s2", default="data/processed/test_source2.parquet", help="Test Source 2 parquet")
    parser.add_argument("--test-s3", default="data/processed/test_source3.parquet", help="Test Source 3 parquet")
    parser.add_argument("--output-dir", default="output", help="Directory for submission TSVs")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override (default: optimal threshold from model)")
    parser.add_argument("--max-candidates", type=int, default=15, help="Max candidates per entity")
    parser.add_argument("--min-score", type=float, default=35.0, help="Min blocking score")
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        s1_path=args.test_s1,
        s2_path=args.test_s2,
        s3_path=args.test_s3,
        output_dir=args.output_dir,
        threshold_override=args.threshold,
        max_candidates=args.max_candidates,
        min_score=args.min_score
    )


if __name__ == "__main__":
    main()
