"""
Unit tests for pairwise feature extraction.
"""

import numpy as np
import polars as pl
import pytest
from src.features.pairwise_features import (
    FEATURE_NAMES,
    extract_pair_features,
    extract_feature_matrix,
    extract_features_df,
)


def test_exact_match_features():
    """Verify that identical entities produce perfect 1.0 similarity scores."""
    pair = {
        's1_entity_id': 's1_001',
        'target_entity_id': 's2_001',
        'source_id': 2,
        's1_name': 'acme technology solutions',
        'target_name': 'acme technology solutions',
        's1_address': '123 main street suite 400',
        'target_address': '123 main street suite 400',
        's1_city': 'seattle',
        'target_city': 'seattle',
        's1_postal': '98101',
        'target_postal': '98101',
        's1_country': 'US',
        'target_country': 'US',
        'blocking_score': 95.0,
    }
    feats = extract_pair_features(pair)
    assert len(feats) == len(FEATURE_NAMES)
    fmap = dict(zip(FEATURE_NAMES, feats))

    assert fmap['name_ratio'] == pytest.approx(1.0)
    assert fmap['name_exact_match'] == 1.0
    assert fmap['name_len_diff'] == 0.0
    assert fmap['name_jaccard'] == pytest.approx(1.0)
    assert fmap['addr_both_present'] == 1.0
    assert fmap['addr_ratio'] == pytest.approx(1.0)
    assert fmap['addr_digits_match'] == 1.0
    assert fmap['city_exact_match'] == 1.0
    assert fmap['postal_exact_match'] == 1.0
    assert fmap['country_is_us'] == 1.0
    assert fmap['country_is_in'] == 0.0
    assert fmap['is_source3'] == 0.0
    assert fmap['exact_name_and_city'] == 1.0
    assert fmap['exact_name_and_postal'] == 1.0


def test_partial_match_features():
    """Verify feature behavior on partial name and address matches."""
    pair = {
        's1_entity_id': 's1_002',
        'target_entity_id': 's3_002',
        'source_id': 3,
        's1_name': 'acme tech solutions',
        'target_name': 'acme technologies corp',
        's1_address': '123 main st',
        'target_address': '456 oak ave',
        's1_city': 'mumbai',
        'target_city': 'mumbai',
        's1_postal': '400001',
        'target_postal': '400002',
        's1_country': 'India',
        'target_country': 'India',
        'blocking_score': 60.0,
    }
    feats = extract_pair_features(pair)
    fmap = dict(zip(FEATURE_NAMES, feats))

    assert 0.0 < fmap['name_ratio'] < 1.0
    assert fmap['name_exact_match'] == 0.0
    assert fmap['is_source3'] == 1.0
    assert fmap['country_is_in'] == 1.0
    assert fmap['country_is_us'] == 0.0
    assert fmap['city_exact_match'] == 1.0
    assert fmap['postal_exact_match'] == 0.0
    assert fmap['postal_p2_match'] == 1.0  # Both start with '40'
    assert fmap['addr_digits_match'] == 0.0  # '123' vs '456'


def test_missing_attributes_graceful():
    """Verify feature extraction is robust to missing or empty attributes."""
    pair = {
        's1_entity_id': 's1_003',
        'target_entity_id': 's2_003',
        'source_id': 2,
        's1_name': 'sole proprietorship',
        'target_name': 'sole proprietorship',
        's1_address': '',
        'target_address': None,
        's1_city': None,
        'target_city': '',
        's1_postal': None,
        'target_postal': None,
        's1_country': 'France',
        'target_country': 'France',
        'blocking_score': 0.0,
    }
    feats = extract_pair_features(pair)
    assert len(feats) == len(FEATURE_NAMES)
    assert not any(np.isnan(f) or np.isinf(f) for f in feats)
    fmap = dict(zip(FEATURE_NAMES, feats))

    assert fmap['addr_s1_missing'] == 1.0
    assert fmap['addr_t_missing'] == 1.0
    assert fmap['addr_both_present'] == 0.0
    assert fmap['city_both_present'] == 0.0
    assert fmap['postal_both_present'] == 0.0
    assert fmap['country_is_fr'] == 1.0


def test_extract_feature_matrix_and_df():
    """Verify batch matrix and Polars DataFrame extraction."""
    pairs = [
        {
            's1_entity_id': 's1_a',
            'target_entity_id': 's2_a',
            'source_id': 2,
            's1_name': 'alpha corp',
            'target_name': 'alpha corp',
            's1_country': 'US',
        },
        {
            's1_entity_id': 's1_b',
            'target_entity_id': 's3_b',
            'source_id': 3,
            's1_name': 'beta llc',
            'target_name': 'beta inc',
            's1_country': 'India',
        },
    ]

    # Matrix extraction
    X, feat_names = extract_feature_matrix(pairs)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert X.dtype == np.float32
    assert feat_names == FEATURE_NAMES

    # Empty handling
    X_empty, _ = extract_feature_matrix([])
    assert X_empty.shape == (0, len(FEATURE_NAMES))

    # DataFrame extraction
    df = extract_features_df(pairs)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert 's1_entity_id' in df.columns
    assert 'target_entity_id' in df.columns
    assert 'source_id' in df.columns
    for fname in FEATURE_NAMES:
        assert fname in df.columns

    # Empty DataFrame
    df_empty = extract_features_df([])
    assert df_empty.height == 0
    assert 's1_entity_id' in df_empty.columns
