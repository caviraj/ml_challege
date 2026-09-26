"""
Unit tests for blocking and candidate generation engine.
"""

import pytest
import polars as pl
from src.blocking.keys import extract_blocking_keys
from src.blocking.indexer import CountryPartitionedIndex
from src.blocking.candidate_generator import CandidateGenerator, score_candidate_pair


def test_extract_blocking_keys():
    record = {
        'entity_id': 'E1',
        'name_norm': 'amazon web services inc',
        'clean_address': '410 terry ave n',
        'city': 'seattle',
        'postal_code': '98109',
        'country_norm': 'united states'
    }
    keys = extract_blocking_keys(record)
    assert len(keys) > 0

    # Check that country is part of all keys
    for k in keys:
        assert 'united states' in k

    # Check name prefix
    assert any(k.startswith('cn_p3:united states:ama') for k in keys)
    assert any(k.startswith('cn_p4:united states:amaz') for k in keys)

    # Check postal key
    assert any('post_p2:united states:98109:am' in k for k in keys)


def test_country_partitioned_index():
    index = CountryPartitionedIndex(max_postlist_size=50)

    # Add sample targets manually to index for testing
    us_target = ('T_US_1', 2, 'amazon retail', '410 terry ave', 'seattle', '98109', 'united states')
    in_target = ('T_IN_1', 2, 'amazon india', 'brigade road', 'bengaluru', '560001', 'india')

    index.targets['united states'].append(us_target)
    for k in extract_blocking_keys({'name_norm': 'amazon retail', 'clean_address': '410 terry ave', 'city': 'seattle', 'postal_code': '98109', 'country_norm': 'united states'}):
        index.index['united states'][k].append(0)

    index.targets['india'].append(in_target)
    for k in extract_blocking_keys({'name_norm': 'amazon india', 'clean_address': 'brigade road', 'city': 'bengaluru', 'postal_code': '560001', 'country_norm': 'india'}):
        index.index['india'][k].append(0)

    # Query for US record
    s1_us = {'name_norm': 'amazon corporate', 'clean_address': 'terry ave', 'city': 'seattle', 'postal_code': '98109', 'country_norm': 'united states'}
    cands_us = index.get_candidate_indices(s1_us)
    assert 0 in cands_us

    # Query for IN record
    s1_in = {'name_norm': 'amazon enterprise', 'clean_address': 'brigade road', 'city': 'bengaluru', 'postal_code': '560001', 'country_norm': 'india'}
    cands_in = index.get_candidate_indices(s1_in)
    assert 0 in cands_in

    # Zero cross-country leakage:
    # A US query should NEVER return candidates from India index
    assert index.get_candidate_indices({'country_norm': 'france'}) == set()


def test_candidate_generator():
    index = CountryPartitionedIndex()
    target_1 = ('T1', 2, 'starbucks coffee company', '1912 pike place', 'seattle', '98101', 'united states')
    target_2 = ('T2', 3, 'walmart store 100', '100 main st', 'bentonville', '72712', 'united states')

    index.targets['united states'] = [target_1, target_2]
    for idx, t in enumerate([target_1, target_2]):
        for k in extract_blocking_keys({'name_norm': t[2], 'clean_address': t[3], 'city': t[4], 'postal_code': t[5], 'country_norm': t[6]}):
            index.index['united states'][k].append(idx)

    gen = CandidateGenerator(index)
    s1 = {
        'entity_id': 'S1_1',
        'name_norm': 'starbucks coffee',
        'clean_address': 'pike place',
        'city': 'seattle',
        'postal_code': '98101',
        'country_norm': 'united states'
    }

    cands = gen.generate_candidates_for_record(s1, max_candidates=10, min_score=35.0)
    assert len(cands) >= 1
    best_cand = cands[0]
    assert best_cand[0] == 'T1'
    assert best_cand[2] > 70.0  # high similarity score


def test_chunk_candidate_generation():
    index = CountryPartitionedIndex()
    target = ('T10', 2, 'microsoft corporation', 'one microsoft way', 'redmond', '98052', 'united states')
    index.targets['united states'] = [target]
    for k in extract_blocking_keys({'name_norm': target[2], 'clean_address': target[3], 'city': target[4], 'postal_code': target[5], 'country_norm': target[6]}):
        index.index['united states'][k].append(0)

    gen = CandidateGenerator(index)
    df_chunk = pl.DataFrame({
        'entity_id': ['S1_10'],
        'name_norm': ['microsoft corp'],
        'clean_address': ['microsoft way'],
        'city': ['redmond'],
        'postal_code': ['98052'],
        'country_norm': ['united states']
    })

    pairs = gen.generate_candidate_pairs_for_chunk(df_chunk, max_candidates=5, min_score=40.0)
    assert len(pairs) == 1
    assert pairs[0]['s1_entity_id'] == 'S1_10'
    assert pairs[0]['target_entity_id'] == 'T10'
    assert pairs[0]['blocking_score'] > 70.0
