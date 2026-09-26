"""
Candidate generation engine for business entity resolution.
Wraps CountryPartitionedIndex to retrieve, score, and rank candidate matches
for Source 1 entities with high recall and bounded candidate counts.
"""

from typing import Dict, List, Tuple, Any, Optional, Iterator
import logging
from rapidfuzz import fuzz
import polars as pl
from .indexer import CountryPartitionedIndex, TargetRecord

logger = logging.getLogger(__name__)

# Scored candidate representation:
# (target_entity_id, source_id, combined_score, target_record)
ScoredCandidate = Tuple[str, int, float, TargetRecord]


def score_candidate_pair(
    s1_name: str,
    s1_addr: str,
    t_name: str,
    t_addr: str
) -> float:
    """
    Computes lightweight pre-ranking similarity score between Source 1 entity and a target candidate.
    Uses RapidFuzz ratio, token_sort_ratio, token_set_ratio and adaptive address weighting.
    """
    if not s1_name or not t_name:
        return 0.0

    # Fast name similarity metrics
    name_sim = fuzz.ratio(s1_name, t_name)
    ts_sim = fuzz.token_sort_ratio(s1_name, t_name)
    tset_sim = fuzz.token_set_ratio(s1_name, t_name)
    n_score = max(name_sim, ts_sim, 0.92 * tset_sim)

    # Address similarity
    addr_sim = fuzz.token_set_ratio(s1_addr, t_addr) if s1_addr and t_addr else 0.0

    # Adaptive combination logic
    if not s1_addr or not t_addr:
        return float(n_score)
    
    if n_score >= 80.0:
        return float(0.85 * n_score + 0.15 * addr_sim)
    elif n_score < 30.0 and addr_sim > 50.0:
        # Cross-script transliterated name with matching address
        return float(addr_sim)
    else:
        return float(0.60 * n_score + 0.40 * addr_sim)


class CandidateGenerator:
    """
    Generates and pre-ranks target candidates for Source 1 records.
    """

    def __init__(self, index: CountryPartitionedIndex):
        self.index = index

    def generate_candidates_for_record(
        self,
        record: Dict[str, Any],
        max_candidates: int = 25,
        min_score: float = 35.0
    ) -> List[ScoredCandidate]:
        """
        Generates top-K candidates for a single Source 1 record.
        Returns list of (target_entity_id, source_id, combined_score, target_record)
        sorted by combined_score descending.
        """
        country = record.get('country_norm', '') or 'unknown'
        cand_indices = self.index.get_candidate_indices(record)
        if not cand_indices:
            return []

        s1_name = record.get('name_norm') or ''
        s1_addr = record.get('clean_address') or ''

        scored_candidates: List[ScoredCandidate] = []
        for idx in cand_indices:
            t_record = self.index.get_target(country, idx)
            # t_record: (entity_id, source_id, name_norm, clean_address, city, postal_code, country_norm)
            t_id = t_record[0]
            s_id = t_record[1]
            t_name = t_record[2]
            t_addr = t_record[3]

            score = score_candidate_pair(s1_name, s1_addr, t_name, t_addr)
            if score >= min_score:
                scored_candidates.append((t_id, s_id, score, t_record))

        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        return scored_candidates[:max_candidates]

    def generate_candidate_pairs_for_chunk(
        self,
        s1_chunk: pl.DataFrame,
        max_candidates: int = 25,
        min_score: float = 35.0
    ) -> List[Dict[str, Any]]:
        """
        Generates candidate pairs for a chunk of Source 1 records as a list of dicts
        ready for feature extraction.
        Each dict contains:
            s1_entity_id, target_entity_id, source_id, blocking_score,
            s1_name, s1_address, s1_city, s1_postal, s1_country,
            target_name, target_address, target_city, target_postal, target_country
        """
        pairs: List[Dict[str, Any]] = []

        for row in s1_chunk.iter_rows(named=True):
            s1_id = row['entity_id']
            cands = self.generate_candidates_for_record(
                row,
                max_candidates=max_candidates,
                min_score=min_score
            )
            for t_id, s_id, score, t_record in cands:
                pairs.append({
                    's1_entity_id': s1_id,
                    'target_entity_id': t_id,
                    'source_id': s_id,
                    'blocking_score': score,
                    's1_name': row.get('name_norm') or '',
                    's1_address': row.get('clean_address') or '',
                    's1_city': row.get('city') or '',
                    's1_postal': row.get('postal_code') or '',
                    's1_country': row.get('country_norm') or '',
                    'target_name': t_record[2] or '',
                    'target_address': t_record[3] or '',
                    'target_city': t_record[4] or '',
                    'target_postal': t_record[5] or '',
                    'target_country': t_record[6] or ''
                })

        return pairs
