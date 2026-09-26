"""Official Macro F0.5 Metric implementation with Singleton Rule handling.

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
      = (1.25 * TP) / (0.25 * |True| + |Pred|)

Singletons:
- If |True| == 0:
    - If |Pred| == 0: F0.5 = 1.0 (Correctly identified singleton)
    - If |Pred| > 0:  F0.5 = 0.0 (False merge penalty)
- If |True| > 0:
    - If |Pred| == 0: F0.5 = 0.0 (Missed matches)
    - If TP == 0:     F0.5 = 0.0
"""

from typing import Dict, Iterable, List, Set, Tuple, Union
import numpy as np
import pandas as pd


def parse_id_list(val: Union[str, Iterable[str], None]) -> Set[str]:
    """Parse comma-separated or iterable ID string into a set of clean IDs."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, (set, frozenset)):
        return set(val)
    if isinstance(val, (list, tuple)):
        return {str(x).strip() for x in val if str(x).strip()}
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return set()
        return {x.strip() for x in val.split(",") if x.strip()}
    return set()


def compute_entity_f05(
    y_true: Union[Set[str], List[str], str, None],
    y_pred: Union[Set[str], List[str], str, None],
) -> float:
    """Compute F0.5 score for a single Source 1 entity.

    Parameters
    ----------
    y_true : Set[str], List[str], or comma-separated str of ground-truth IDs.
    y_pred : Set[str], List[str], or comma-separated str of predicted IDs.

    Returns
    -------
    float : F0.5 score in range [0.0, 1.0].
    """
    true_set = parse_id_list(y_true)
    pred_set = parse_id_list(y_pred)

    len_true = len(true_set)
    len_pred = len(pred_set)

    # Singleton case
    if len_true == 0:
        return 1.0 if len_pred == 0 else 0.0

    # Non-singleton but no prediction
    if len_pred == 0:
        return 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0

    # Formula: (1.25 * tp) / (0.25 * len_true + len_pred)
    return (1.25 * tp) / (0.25 * len_true + len_pred)


def to_id_dict(
    data: Union[Dict[str, Union[Set[str], List[str], str]], pd.DataFrame],
    id_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
) -> Dict[str, Set[str]]:
    """Convert input dictionary or DataFrame to standard {entity_id: set_of_matches}."""
    if isinstance(data, dict):
        return {k: parse_id_list(v) for k, v in data.items()}
    elif isinstance(data, pd.DataFrame):
        if id_col not in data.columns or match_col not in data.columns:
            raise ValueError(f"DataFrame must contain columns '{id_col}' and '{match_col}'")
        res = {}
        for s1_id, matches in zip(data[id_col], data[match_col]):
            res[str(s1_id).strip()] = parse_id_list(matches)
        return res
    else:
        raise TypeError(f"Unsupported data type for ground truth/predictions: {type(data)}")


def compute_macro_f05(
    ground_truth: Union[Dict[str, Union[Set[str], List[str], str]], pd.DataFrame],
    predictions: Union[Dict[str, Union[Set[str], List[str], str]], pd.DataFrame],
    all_s1_ids: Union[Iterable[str], None] = None,
) -> float:
    """Compute Macro F0.5 score across all Source 1 entities.

    Parameters
    ----------
    ground_truth : Ground truth mapping or DataFrame.
    predictions : Predicted mapping or DataFrame.
    all_s1_ids : Optional explicit list/set of all S1 IDs to evaluate over.
                 If None, uses keys of ground_truth.

    Returns
    -------
    float : Macro-averaged F0.5 score.
    """
    gt_dict = to_id_dict(ground_truth)
    pred_dict = to_id_dict(predictions)

    if all_s1_ids is None:
        eval_ids = list(gt_dict.keys())
    else:
        eval_ids = [str(x).strip() for x in all_s1_ids]

    if not eval_ids:
        return 0.0

    scores = []
    for s1_id in eval_ids:
        true_matches = gt_dict.get(s1_id, set())
        pred_matches = pred_dict.get(s1_id, set())
        scores.append(compute_entity_f05(true_matches, pred_matches))

    return float(np.mean(scores))


def detailed_evaluation_report(
    ground_truth: Union[Dict[str, Union[Set[str], List[str], str]], pd.DataFrame],
    predictions: Union[Dict[str, Union[Set[str], List[str], str]], pd.DataFrame],
    all_s1_ids: Union[Iterable[str], None] = None,
) -> Dict[str, Union[float, int]]:
    """Produce detailed diagnostic evaluation report."""
    gt_dict = to_id_dict(ground_truth)
    pred_dict = to_id_dict(predictions)

    if all_s1_ids is None:
        eval_ids = list(gt_dict.keys())
    else:
        eval_ids = [str(x).strip() for x in all_s1_ids]

    n_entities = len(eval_ids)
    if n_entities == 0:
        return {"macro_f05": 0.0, "total_entities": 0}

    scores = []
    singleton_scores = []
    non_singleton_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_matches = 0
    total_pred_matches = 0

    for s1_id in eval_ids:
        true_matches = gt_dict.get(s1_id, set())
        pred_matches = pred_dict.get(s1_id, set())

        score = compute_entity_f05(true_matches, pred_matches)
        scores.append(score)

        is_singleton = len(true_matches) == 0
        if is_singleton:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

        tp = len(true_matches & pred_matches)
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true_matches += len(true_matches)
        total_pred_matches += len(pred_matches)

    macro_f05 = float(np.mean(scores))
    singleton_f05 = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    precision_micro = total_tp / total_pred_matches if total_pred_matches > 0 else 0.0
    recall_micro = total_tp / total_true_matches if total_true_matches > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "total_entities": n_entities,
        "singleton_count": len(singleton_scores),
        "singleton_f05": singleton_f05,
        "non_singleton_count": len(non_singleton_scores),
        "non_singleton_f05": non_singleton_f05,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
    }
