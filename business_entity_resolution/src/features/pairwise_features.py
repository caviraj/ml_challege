"""
Pairwise feature extraction for business entity resolution.
Computes string, token, phonetic, and geospatial similarity features
between Source 1 entities and candidate target entities from Source 2 and Source 3.
"""

from typing import Dict, List, Tuple, Any, Optional
import re
import numpy as np
import polars as pl
from rapidfuzz import fuzz

FEATURE_NAMES = [
    # Name features
    'name_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_len_diff',
    'name_len_ratio',
    'name_exact_match',
    'name_prefix_match',
    'name_jaccard',
    
    # Address features
    'addr_s1_missing',
    'addr_t_missing',
    'addr_both_present',
    'addr_ratio',
    'addr_token_set_ratio',
    'addr_jaccard',
    'addr_digits_match',
    'addr_digits_jaccard',
    
    # City features
    'city_both_present',
    'city_exact_match',
    'city_ratio',
    
    # Postal features
    'postal_both_present',
    'postal_exact_match',
    'postal_p2_match',
    
    # Metadata & Source
    'is_source3',
    'blocking_score',
    
    # Country indicators
    'country_is_us',
    'country_is_in',
    'country_is_fr',
    
    # Interaction features
    'name_x_addr',
    'exact_name_and_city',
    'exact_name_and_postal'
]

_DIGIT_RE = re.compile(r'\d+')


def extract_pair_features(pair: Dict[str, Any]) -> List[float]:
    """
    Extracts numerical feature vector for a single candidate pair dictionary.
    Order matches FEATURE_NAMES exactly.
    """
    s1_name: str = pair.get('s1_name') or ''
    t_name: str = pair.get('target_name') or ''
    s1_addr: str = pair.get('s1_address') or ''
    t_addr: str = pair.get('target_address') or ''
    s1_city: str = pair.get('s1_city') or ''
    t_city: str = pair.get('target_city') or ''
    s1_post: str = pair.get('s1_postal') or ''
    t_post: str = pair.get('target_postal') or ''
    s1_country: str = pair.get('s1_country') or ''
    t_country: str = pair.get('target_country') or ''
    source_id: int = int(pair.get('source_id') or 2)
    blocking_score: float = float(pair.get('blocking_score') or 0.0)

    # 1. Name features
    len1 = len(s1_name)
    len2 = len(t_name)
    max_len = max(len1, len2, 1)
    min_len = min(len1, len2)

    if len1 > 0 and len2 > 0:
        name_ratio = fuzz.ratio(s1_name, t_name) / 100.0
        name_partial_ratio = fuzz.partial_ratio(s1_name, t_name) / 100.0
        name_token_sort_ratio = fuzz.token_sort_ratio(s1_name, t_name) / 100.0
        name_token_set_ratio = fuzz.token_set_ratio(s1_name, t_name) / 100.0
        name_len_diff = float(abs(len1 - len2))
        name_len_ratio = float(min_len / max_len)
        name_exact_match = 1.0 if s1_name == t_name else 0.0
        name_prefix_match = 1.0 if (s1_name.startswith(t_name) or t_name.startswith(s1_name)) else 0.0

        s1_words = set(s1_name.split())
        t_words = set(t_name.split())
        if s1_words and t_words:
            name_jaccard = float(len(s1_words & t_words) / len(s1_words | t_words))
        else:
            name_jaccard = 0.0
    else:
        name_ratio = 0.0
        name_partial_ratio = 0.0
        name_token_sort_ratio = 0.0
        name_token_set_ratio = 0.0
        name_len_diff = float(max(len1, len2))
        name_len_ratio = 0.0
        name_exact_match = 0.0
        name_prefix_match = 0.0
        name_jaccard = 0.0

    # 2. Address features
    addr_s1_missing = 1.0 if not s1_addr else 0.0
    addr_t_missing = 1.0 if not t_addr else 0.0
    addr_both_present = 1.0 if (s1_addr and t_addr) else 0.0

    if addr_both_present > 0.5:
        addr_ratio = fuzz.ratio(s1_addr, t_addr) / 100.0
        addr_token_set_ratio = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0
        s1_awords = set(s1_addr.split())
        t_awords = set(t_addr.split())
        if s1_awords and t_awords:
            addr_jaccard = float(len(s1_awords & t_awords) / len(s1_awords | t_awords))
        else:
            addr_jaccard = 0.0

        # Number extraction
        s1_nums = set(_DIGIT_RE.findall(s1_addr))
        t_nums = set(_DIGIT_RE.findall(t_addr))
        if s1_nums and t_nums:
            addr_digits_match = 1.0 if s1_nums == t_nums else 0.0
            addr_digits_jaccard = float(len(s1_nums & t_nums) / len(s1_nums | t_nums))
        elif not s1_nums and not t_nums:
            addr_digits_match = 0.5
            addr_digits_jaccard = 0.5
        else:
            addr_digits_match = 0.0
            addr_digits_jaccard = 0.0
    else:
        addr_ratio = 0.0
        addr_token_set_ratio = 0.0
        addr_jaccard = 0.0
        addr_digits_match = 0.5
        addr_digits_jaccard = 0.5

    # 3. City features
    city_both_present = 1.0 if (s1_city and t_city) else 0.0
    if city_both_present > 0.5:
        city_exact_match = 1.0 if s1_city == t_city else 0.0
        city_ratio = fuzz.ratio(s1_city, t_city) / 100.0
    else:
        city_exact_match = 0.0
        city_ratio = 0.0

    # 4. Postal features
    postal_both_present = 1.0 if (s1_post and t_post) else 0.0
    if postal_both_present > 0.5:
        postal_exact_match = 1.0 if s1_post == t_post else 0.0
        postal_p2_match = 1.0 if (len(s1_post) >= 2 and len(t_post) >= 2 and s1_post[:2] == t_post[:2]) else 0.0
    else:
        postal_exact_match = 0.0
        postal_p2_match = 0.0

    # 5. Metadata & Source
    is_source3 = 1.0 if source_id == 3 else 0.0
    norm_blocking_score = blocking_score / 100.0 if blocking_score > 1.0 else blocking_score

    # 6. Country indicators
    c_str = str(s1_country or t_country or '').strip().lower()
    country_is_us = 1.0 if c_str in ('us', 'usa', 'united states') else 0.0
    country_is_in = 1.0 if c_str in ('india', 'in', 'ind') else 0.0
    country_is_fr = 1.0 if c_str in ('france', 'fr', 'fra') else 0.0

    # 7. Interaction features
    if addr_both_present > 0.5:
        name_x_addr = name_token_set_ratio * addr_token_set_ratio
    else:
        name_x_addr = name_token_set_ratio * 0.5

    exact_name_and_city = 1.0 if (name_exact_match > 0.5 and city_exact_match > 0.5) else 0.0
    exact_name_and_postal = 1.0 if (name_exact_match > 0.5 and postal_exact_match > 0.5) else 0.0

    return [
        name_ratio,
        name_partial_ratio,
        name_token_sort_ratio,
        name_token_set_ratio,
        name_len_diff,
        name_len_ratio,
        name_exact_match,
        name_prefix_match,
        name_jaccard,
        addr_s1_missing,
        addr_t_missing,
        addr_both_present,
        addr_ratio,
        addr_token_set_ratio,
        addr_jaccard,
        addr_digits_match,
        addr_digits_jaccard,
        city_both_present,
        city_exact_match,
        city_ratio,
        postal_both_present,
        postal_exact_match,
        postal_p2_match,
        is_source3,
        norm_blocking_score,
        country_is_us,
        country_is_in,
        country_is_fr,
        name_x_addr,
        exact_name_and_city,
        exact_name_and_postal
    ]


def extract_feature_matrix(pairs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts a 2D NumPy array of shape (N, num_features) from a list of candidate pair dicts.
    Returns (X, FEATURE_NAMES).
    """
    n = len(pairs)
    n_feats = len(FEATURE_NAMES)
    if n == 0:
        return np.empty((0, n_feats), dtype=np.float32), FEATURE_NAMES

    X = np.empty((n, n_feats), dtype=np.float32)
    for i, pair in enumerate(pairs):
        X[i, :] = extract_pair_features(pair)

    return X, FEATURE_NAMES


def extract_features_df(pairs: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Extracts candidate pair features into a Polars DataFrame including identifier columns.
    Useful for training data preparation and inspection.
    """
    if not pairs:
        schema = {
            's1_entity_id': pl.Utf8,
            'target_entity_id': pl.Utf8,
            'source_id': pl.Int32,
            **{fname: pl.Float32 for fname in FEATURE_NAMES}
        }
        return pl.DataFrame(schema=schema)

    X, _ = extract_feature_matrix(pairs)
    data = {
        's1_entity_id': [p['s1_entity_id'] for p in pairs],
        'target_entity_id': [p['target_entity_id'] for p in pairs],
        'source_id': [int(p.get('source_id') or 2) for p in pairs],
    }
    for idx, fname in enumerate(FEATURE_NAMES):
        data[fname] = X[:, idx]

    return pl.DataFrame(data)
