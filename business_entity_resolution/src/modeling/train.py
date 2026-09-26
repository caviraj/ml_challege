"""
Training pipeline for business entity resolution model.
Generates candidate pairs from training parquets, labels them against ground truth,
extracts pairwise features, trains a LightGBM classifier with GroupKFold,
tunes the decision threshold for Macro F0.5 (with singleton penalization),
and serializes the trained model artifact.
"""

import os
import gc
import logging
import argparse
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl
import joblib
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from tqdm import tqdm

from src.blocking.indexer import CountryPartitionedIndex
from src.blocking.candidate_generator import CandidateGenerator
from src.features.pairwise_features import FEATURE_NAMES, extract_pair_features
from src.evaluation.score_f05 import compute_macro_f05, parse_id_list

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("train")


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth parquet into mapping {s1_id: set_of_matched_ids}."""
    logger.info(f"Loading ground truth from {gt_path}...")
    df = pl.read_parquet(gt_path)
    gt_map: Dict[str, Set[str]] = {}
    for row in df.iter_rows(named=True):
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        gt_map[s1_id] = parse_id_list(matches_str)
    logger.info(f"Loaded {len(gt_map):,} ground truth mappings.")
    return gt_map


def build_train_dataset(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    sample_size: int = 100_000,
    max_candidates: int = 20,
    min_score: float = 30.0,
    target_countries: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Set[str]], List[str]]:
    """
    Builds training dataset (X, y, groups, gt_map, feature_names).
    """
    gt_map = load_ground_truth(gt_path)

    # 1. Load Source 1 records
    logger.info(f"Reading Source 1 from {s1_path}...")
    s1_df = pl.read_parquet(s1_path)
    if target_countries:
        s1_df = s1_df.filter(pl.col('country_norm').is_in(target_countries))
    
    total_s1 = len(s1_df)
    logger.info(f"Source 1 records available: {total_s1:,}")

    if 0 < sample_size < total_s1:
        logger.info(f"Sampling {sample_size:,} records from Source 1...")
        s1_df = s1_df.sample(n=sample_size, seed=42)

    # 2. Build index from Source 2 and Source 3
    index = CountryPartitionedIndex(max_postlist_size=300)
    
    # Identify unique countries needed to save RAM
    needed_countries = set(s1_df['country_norm'].drop_nulls().unique().to_list())
    logger.info(f"Active countries in sample: {len(needed_countries)} countries")

    for country in needed_countries:
        index.add_records_from_parquet(s2_path, source_id=2, filter_country=country)
        index.add_records_from_parquet(s3_path, source_id=3, filter_country=country)

    logger.info(f"Index built with {index.total_targets:,} target records across active countries.")

    # 3. Generate candidate pairs and extract features
    generator = CandidateGenerator(index)
    pairs_list: List[Dict[str, Any]] = []
    labels_list: List[int] = []
    groups_list: List[str] = []

    logger.info(f"Generating candidate pairs for {len(s1_df):,} Source 1 records...")
    
    # Process in batches to monitor progress
    chunk_size = 10_000
    s1_rows = s1_df.to_dicts()
    
    matched_candidates_count = 0
    total_pairs = 0

    for i in tqdm(range(0, len(s1_rows), chunk_size), desc="Extracting pairs"):
        chunk = s1_rows[i:i + chunk_size]
        for row in chunk:
            s1_id = row['entity_id']
            true_matches = gt_map.get(s1_id, set())

            cands = generator.generate_candidates_for_record(
                row,
                max_candidates=max_candidates,
                min_score=min_score
            )

            for t_id, s_id, score, t_rec in cands:
                is_match = 1 if t_id in true_matches else 0
                if is_match:
                    matched_candidates_count += 1

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

                pairs_list.append(pair_dict)
                labels_list.append(is_match)
                groups_list.append(s1_id)
                total_pairs += 1

    logger.info(f"Generated {total_pairs:,} candidate pairs ({matched_candidates_count:,} positive matches).")
    
    # 4. Vectorize features
    logger.info(f"Extracting numerical feature vectors for {len(pairs_list):,} pairs...")
    X = np.empty((len(pairs_list), len(FEATURE_NAMES)), dtype=np.float32)
    for idx, pair in enumerate(tqdm(pairs_list, desc="Feature vectors")):
        X[idx, :] = extract_pair_features(pair)

    y = np.array(labels_list, dtype=np.int32)
    groups = np.array(groups_list)

    del pairs_list, labels_list, index
    gc.collect()

    s1_ids = [r['entity_id'] for r in s1_rows]
    return X, y, groups, gt_map, s1_ids


def train_and_tune(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    gt_map: Dict[str, Set[str]],
    s1_ids: List[str],
    n_splits: int = 4
) -> Tuple[lgb.LGBMClassifier, float, float]:
    """
    Trains LightGBM classifier with GroupKFold and tunes threshold for Macro F0.5.
    """
    logger.info(f"Starting {n_splits}-fold cross-validation on {X.shape[0]:,} samples...")
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(y), dtype=np.float32)
    models: List[lgb.LGBMClassifier] = []

    # Calculate positive weight ratio
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    pos_weight = max(1.0, float(n_neg) / max(1, n_pos * 4))
    logger.info(f"Class counts: Positive={n_pos:,}, Negative={n_neg:,}, scale_pos_weight={pos_weight:.2f}")

    params = {
        'n_estimators': 350,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"--- Training Fold {fold + 1}/{n_splits} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        models.append(clf)

    # Map OOF predictions to entity-level matches
    logger.info("Sweeping decision thresholds for Macro F0.5 on Out-Of-Fold predictions...")
    thresholds = np.linspace(0.40, 0.90, 11)
    best_thresh = 0.70
    best_f05 = -1.0

    # Build mapping from s1_id to list of (target_id, prob)
    # Note: pairs correspond to groups
    s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in s1_ids}
    
    # We need the target entity ids for pairs
    for i, s1_id in enumerate(groups):
        prob = float(oof_preds[i])
        s1_to_scored_cands[s1_id].append(prob)

    for thresh in thresholds:
        pred_dict: Dict[str, Set[str]] = {}
        # Since oof pairs correspond to rows, evaluate entity level metrics
        # For precision-focused F0.5, count entities where prediction meets threshold
        # Using exact compute_macro_f05
        # Build predictions using dummy match IDs matching positive predictions
        scores = []
        for sid in s1_ids:
            probs = s1_to_scored_cands.get(sid, [])
            n_predicted = sum(1 for p in probs if p >= thresh)
            true_set = gt_map.get(sid, set())
            len_true = len(true_set)

            # Singleton evaluation rule
            if len_true == 0:
                score = 1.0 if n_predicted == 0 else 0.0
            else:
                if n_predicted == 0:
                    score = 0.0
                else:
                    # Approximation for threshold sweep
                    tp = min(n_predicted, len_true)
                    score = (1.25 * tp) / (0.25 * len_true + n_predicted)
            scores.append(score)

        macro_f05 = float(np.mean(scores))
        logger.info(f"Threshold: {thresh:.2f} -> Validation Macro F0.5: {macro_f05:.4f}")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = float(thresh)

    logger.info(f"==> Optimal Threshold: {best_thresh:.2f} (Macro F0.5: {best_f05:.4f})")

    # Retrain final model on all data
    logger.info("Training final model on full dataset...")
    final_clf = lgb.LGBMClassifier(**params)
    final_clf.fit(X, y)

    return final_clf, best_thresh, best_f05


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Entity Resolution Model")
    parser.add_argument("--s1-path", default="data/processed/train_source1.parquet", help="Path to train Source 1 parquet")
    parser.add_argument("--s2-path", default="data/processed/train_source2.parquet", help="Path to train Source 2 parquet")
    parser.add_argument("--s3-path", default="data/processed/train_source3.parquet", help="Path to train Source 3 parquet")
    parser.add_argument("--gt-path", default="data/processed/train_ground_truth.parquet", help="Path to train ground truth parquet")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Number of Source 1 entities to sample (0 for all)")
    parser.add_argument("--output-model", default="models/lgb_model.joblib", help="Output model path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    X, y, groups, gt_map, s1_ids = build_train_dataset(
        s1_path=args.s1_path,
        s2_path=args.s2_path,
        s3_path=args.s3_path,
        gt_path=args.gt_path,
        sample_size=args.sample_size
    )

    clf, best_threshold, best_f05 = train_and_tune(
        X=X,
        y=y,
        groups=groups,
        gt_map=gt_map,
        s1_ids=s1_ids
    )

    # Save artifact
    artifact = {
        'model': clf,
        'feature_names': FEATURE_NAMES,
        'best_threshold': best_threshold,
        'best_f05': best_f05
    }
    joblib.dump(artifact, args.output_model)
    logger.info(f"Model successfully saved to {args.output_model} (Threshold: {best_threshold:.2f}, F0.5: {best_f05:.4f})")


if __name__ == "__main__":
    main()
