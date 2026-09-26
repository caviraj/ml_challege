"""
Country-partitioned inverted index for candidate generation in business entity resolution.
Maintains inverted indexes partitioned by country to ensure zero cross-country candidate leakage,
fast lookups, and bounded memory usage.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging
import gc
import polars as pl
from .keys import extract_blocking_keys

logger = logging.getLogger(__name__)

# Lightweight representation of a target entity (Source 2 or Source 3)
# (entity_id, source_id, name_norm, clean_address, city, postal_code, country_norm)
TargetRecord = Tuple[str, int, str, str, str, str, str]


class CountryPartitionedIndex:
    """
    Inverted index partitioned by country.
    Stores targets from Source 2 and Source 3 and maps blocking keys to target indices.
    """

    def __init__(self, max_postlist_size: int = 500):
        self.max_postlist_size = max_postlist_size
        # country -> list of TargetRecord
        self.targets: Dict[str, List[TargetRecord]] = defaultdict(list)
        # country -> key -> list of target index
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.total_targets = 0

    def add_records_from_parquet(
        self,
        parquet_path: str,
        source_id: int,
        batch_size: int = 100_000,
        filter_country: Optional[str] = None
    ) -> int:
        """
        Reads records in streaming chunks from a Parquet file and populates the index.
        Optionally filters for a specific country to enable modular memory management.
        """
        logger.info(f"Indexing source {source_id} from {parquet_path} (filter_country={filter_country})...")
        
        # Read with polars
        columns = ['entity_id', 'name_norm', 'clean_address', 'city', 'postal_code', 'country_norm']
        df = pl.read_parquet(parquet_path, columns=columns)
        
        if filter_country:
            df = df.filter(pl.col('country_norm') == filter_country)
            
        count = 0
        for row in df.iter_rows(named=True):
            country = row['country_norm'] or 'unknown'
            t_record: TargetRecord = (
                row['entity_id'],
                source_id,
                row['name_norm'] or '',
                row['clean_address'] or '',
                row['city'] or '',
                row['postal_code'] or '',
                country
            )
            
            c_targets = self.targets[country]
            idx = len(c_targets)
            c_targets.append(t_record)
            
            c_index = self.index[country]
            keys = extract_blocking_keys(row)
            for k in keys:
                c_index[k].append(idx)
                
            count += 1
            if count % batch_size == 0:
                logger.info(f"Indexed {count:,} records from source {source_id}...")
                
        self.total_targets += count
        logger.info(f"Finished indexing {count:,} records from source {source_id}.")
        gc.collect()
        return count

    def get_candidate_indices(self, record: Dict[str, Any]) -> Set[int]:
        """
        Retrieves candidate target indices for a Source 1 record based on blocking keys.
        Only queries the target index for the matching country.
        """
        country = record.get('country_norm', '') or 'unknown'
        if country not in self.index:
            return set()

        c_index = self.index[country]
        keys = extract_blocking_keys(record)
        
        candidates: Set[int] = set()
        for k in keys:
            posting = c_index.get(k)
            if posting:
                if len(posting) <= self.max_postlist_size:
                    candidates.update(posting)
                else:
                    # Slice the most specific first items if postlist exceeds threshold
                    candidates.update(posting[:100])
                    
        return candidates

    def get_target(self, country: str, idx: int) -> TargetRecord:
        """Retrieves a TargetRecord by country and index."""
        return self.targets[country][idx]

    def clear_country(self, country: str):
        """Clears all index and target records for a specific country to release memory."""
        if country in self.targets:
            del self.targets[country]
        if country in self.index:
            del self.index[country]
        gc.collect()
        logger.info(f"Cleared index for country '{country}'.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on indexed records and keys across countries."""
        stats = {}
        for country, t_list in self.targets.items():
            k_dict = self.index[country]
            stats[country] = {
                'targets': len(t_list),
                'keys': len(k_dict)
            }
        return stats
