"""Unit tests for the official Macro F0.5 evaluation metric."""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.score_f05 import (
    compute_entity_f05,
    compute_macro_f05,
    detailed_evaluation_report,
    parse_id_list,
)


def test_parse_id_list():
    assert parse_id_list("") == set()
    assert parse_id_list("   ") == set()
    assert parse_id_list(None) == set()
    assert parse_id_list(np.nan) == set()
    assert parse_id_list("S2-001, S3-002, S2-001") == {"S2-001", "S3-002"}
    assert parse_id_list(["S2-001", "S3-002"]) == {"S2-001", "S3-002"}
    assert parse_id_list({"S2-001", "S3-002"}) == {"S2-001", "S3-002"}


def test_readme_example():
    """Verify the exact example provided in student_resource/README.md."""
    # True = [S2-00047, S3-00812], Pred = [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 1.0
    # F_0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = 2.5 / 3.5 = 5/7 ≈ 0.7142857
    y_true = {"S2-00047", "S3-00812"}
    y_pred = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(y_true, y_pred)
    assert score == pytest.approx(5.0 / 7.0, abs=1e-5)


def test_singleton_cases():
    # True singleton with empty prediction -> 1.0
    assert compute_entity_f05(set(), set()) == 1.0
    assert compute_entity_f05("", "") == 1.0
    assert compute_entity_f05(None, None) == 1.0

    # True singleton with false positive prediction -> 0.0
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0
    assert compute_entity_f05("", "S2-00001") == 0.0


def test_non_singleton_cases():
    # Perfect match -> 1.0
    assert compute_entity_f05({"S2-00001"}, {"S2-00001"}) == 1.0

    # Completely wrong match -> 0.0
    assert compute_entity_f05({"S2-00001"}, {"S2-00002"}) == 0.0

    # Missed match -> 0.0
    assert compute_entity_f05({"S2-00001"}, set()) == 0.0
    assert compute_entity_f05({"S2-00001"}, "") == 0.0


def test_compute_macro_f05():
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-3", "S3-3"},
    }
    # Prediction:
    # S1-1: {"S2-1"} -> 1.0
    # S1-2: set() -> 1.0
    # S1-3: {"S2-3"} -> TP=1, |T|=2, |P|=1 -> (1.25*1)/(0.25*2 + 1) = 1.25/1.5 = 5/6 ≈ 0.8333
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": {"S2-3"},
    }
    macro = compute_macro_f05(gt, preds)
    expected = (1.0 + 1.0 + (5.0 / 6.0)) / 3.0
    assert macro == pytest.approx(expected, abs=1e-5)


def test_dataframe_compatibility():
    gt_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-10", ""],
    })
    pred_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-10", "S2-99"],  # second is FP on singleton
    })
    report = detailed_evaluation_report(gt_df, pred_df)
    assert report["total_entities"] == 2
    assert report["singleton_count"] == 1
    assert report["singleton_f05"] == 0.0  # FP on singleton
    assert report["non_singleton_count"] == 1
    assert report["non_singleton_f05"] == 1.0
    assert report["macro_f05"] == 0.5
