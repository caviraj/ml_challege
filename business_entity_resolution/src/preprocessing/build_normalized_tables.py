"""
Build Normalized Parquet Tables from Raw TSV Files.

Processes train and test TSV files in streaming chunks (chunksize <= 50,000)
to ensure peak memory consumption strictly remains under 800MB RAM.
Outputs compressed Snappy Parquet files into data/processed/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.normalize import (
    normalize_address,
    normalize_country,
    normalize_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Canonical PyArrow schema for normalized entity tables
ENTITY_PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("source", pa.string()),
    ("name_norm", pa.string()),
    ("clean_address", pa.string()),
    ("street_tokens", pa.list_(pa.string())),
    ("city", pa.string()),
    ("state", pa.string()),
    ("postal_code", pa.string()),
    ("raw_landmark", pa.string()),
    ("country_norm", pa.string()),
    ("raw_name", pa.string()),
    ("raw_address", pa.string()),
])

# Canonical PyArrow schema for ground truth table
GROUND_TRUTH_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("matched_entity_ids", pa.string()),
    ("match_count", pa.int32()),
    ("is_singleton", pa.bool_()),
])


def process_entity_tsv(
    tsv_path: Path,
    output_parquet_path: Path,
    source_name: str,
    chunksize: int = 50000,
    limit_chunks: Optional[int] = None,
) -> int:
    """
    Reads an entity TSV in streaming chunks, normalizes each record,
    and writes out a Snappy-compressed Parquet table.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input file not found: {tsv_path}")

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_parquet_path.with_suffix(".tmp.parquet")

    logger.info(f"Starting processing: {tsv_path.name} -> {output_parquet_path.name}")
    start_time = time.time()
    total_rows = 0
    chunk_idx = 0

    writer = pq.ParquetWriter(
        temp_output,
        schema=ENTITY_PARQUET_SCHEMA,
        compression="snappy",
    )

    try:
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk_idx += 1
            n_rows = len(chunk)

            # Ensure expected columns exist
            for col in ["entity_id", "business_name", "business_address", "country"]:
                if col not in chunk.columns:
                    raise KeyError(f"Missing column '{col}' in {tsv_path}")

            e_ids = chunk["entity_id"].tolist()
            raw_names = chunk["business_name"].tolist()
            raw_addrs = chunk["business_address"].tolist()
            raw_countries = chunk["country"].tolist()

            name_norms = [normalize_name(n) for n in raw_names]
            countries_norm = [normalize_country(c) for c in raw_countries]
            parsed_addrs = [normalize_address(a) for a in raw_addrs]

            table_dict = {
                "entity_id": e_ids,
                "source": [source_name] * n_rows,
                "name_norm": name_norms,
                "clean_address": [pa_dict["clean_address"] for pa_dict in parsed_addrs],
                "street_tokens": [pa_dict["street_tokens"] for pa_dict in parsed_addrs],
                "city": [pa_dict["city"] for pa_dict in parsed_addrs],
                "state": [pa_dict["state"] for pa_dict in parsed_addrs],
                "postal_code": [pa_dict["postal_code"] for pa_dict in parsed_addrs],
                "raw_landmark": [pa_dict["raw_landmark"] for pa_dict in parsed_addrs],
                "country_norm": countries_norm,
                "raw_name": raw_names,
                "raw_address": raw_addrs,
            }

            pa_table = pa.Table.from_pydict(table_dict, schema=ENTITY_PARQUET_SCHEMA)
            writer.write_table(pa_table)
            total_rows += n_rows

            del chunk, table_dict, pa_table, parsed_addrs, name_norms, countries_norm
            del e_ids, raw_names, raw_addrs, raw_countries
            gc.collect()

            if chunk_idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_rows / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{tsv_path.name}] Chunk {chunk_idx}: {total_rows:,} rows written "
                    f"({rate:,.0f} rows/s, elapsed {elapsed:.1f}s)"
                )

            if limit_chunks is not None and chunk_idx >= limit_chunks:
                logger.info(f"Reached limit of {limit_chunks} chunks.")
                break

    finally:
        writer.close()

    # Atomically rename temp file to final destination
    if output_parquet_path.exists():
        output_parquet_path.unlink()
    temp_output.rename(output_parquet_path)

    elapsed = time.time() - start_time
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Completed {output_parquet_path.name}: {total_rows:,} rows in {elapsed:.1f}s "
        f"({file_size_mb:.2f} MB, {total_rows / elapsed:,.0f} rows/s)"
    )
    return total_rows


def process_ground_truth_tsv(
    tsv_path: Path,
    output_parquet_path: Path,
    chunksize: int = 50000,
    limit_chunks: Optional[int] = None,
) -> int:
    """
    Reads train_ground_truth.tsv in chunks, extracts match counts and singleton status,
    and writes out a Snappy-compressed Parquet table.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input file not found: {tsv_path}")

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_parquet_path.with_suffix(".tmp.parquet")

    logger.info(f"Starting ground truth processing: {tsv_path.name} -> {output_parquet_path.name}")
    start_time = time.time()
    total_rows = 0
    chunk_idx = 0

    writer = pq.ParquetWriter(
        temp_output,
        schema=GROUND_TRUTH_SCHEMA,
        compression="snappy",
    )

    try:
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk_idx += 1
            n_rows = len(chunk)

            s1_ids = chunk["source1_entity_id"].tolist()
            matched_raw = chunk["matched_entity_ids"].tolist()

            match_counts = []
            is_singletons = []
            for m in matched_raw:
                m_str = m.strip()
                if not m_str:
                    match_counts.append(0)
                    is_singletons.append(True)
                else:
                    parts = [p.strip() for p in m_str.split(",") if p.strip()]
                    match_counts.append(len(parts))
                    is_singletons.append(len(parts) == 0)

            table_dict = {
                "source1_entity_id": s1_ids,
                "matched_entity_ids": matched_raw,
                "match_count": match_counts,
                "is_singleton": is_singletons,
            }

            pa_table = pa.Table.from_pydict(table_dict, schema=GROUND_TRUTH_SCHEMA)
            writer.write_table(pa_table)
            total_rows += n_rows

            del chunk, table_dict, pa_table, s1_ids, matched_raw, match_counts, is_singletons
            gc.collect()

            if limit_chunks is not None and chunk_idx >= limit_chunks:
                break

    finally:
        writer.close()

    if output_parquet_path.exists():
        output_parquet_path.unlink()
    temp_output.rename(output_parquet_path)

    elapsed = time.time() - start_time
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Completed {output_parquet_path.name}: {total_rows:,} rows in {elapsed:.1f}s "
        f"({file_size_mb:.2f} MB)"
    )
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Build normalized Parquet tables from TSV files.")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Dataset split to process (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../student_resource/dataset",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to output processed directory",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Number of rows per chunk (default: 50,000)",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Optional limit on number of chunks per file (for smoke testing)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    if args.split in ("train", "all"):
        train_dir = data_dir / "train"
        tasks.extend([
            (train_dir / "train_source1.tsv", output_dir / "train_source1.parquet", "source1", "entity"),
            (train_dir / "train_source2.tsv", output_dir / "train_source2.parquet", "source2", "entity"),
            (train_dir / "train_source3.tsv", output_dir / "train_source3.parquet", "source3", "entity"),
            (train_dir / "train_ground_truth.tsv", output_dir / "train_ground_truth.parquet", "ground_truth", "gt"),
        ])

    if args.split in ("test", "all"):
        test_dir = data_dir / "test"
        tasks.extend([
            (test_dir / "test_source1.tsv", output_dir / "test_source1.parquet", "source1", "entity"),
            (test_dir / "test_source2.tsv", output_dir / "test_source2.parquet", "source2", "entity"),
            (test_dir / "test_source3.tsv", output_dir / "test_source3.parquet", "source3", "entity"),
        ])

    total_start = time.time()
    for in_path, out_path, source, task_type in tasks:
        if not in_path.exists():
            logger.warning(f"File not found, skipping: {in_path}")
            continue

        if task_type == "entity":
            process_entity_tsv(
                tsv_path=in_path,
                output_parquet_path=out_path,
                source_name=source,
                chunksize=args.chunksize,
                limit_chunks=args.limit_chunks,
            )
        elif task_type == "gt":
            process_ground_truth_tsv(
                tsv_path=in_path,
                output_parquet_path=out_path,
                chunksize=args.chunksize,
                limit_chunks=args.limit_chunks,
            )

    logger.info(f"All processing complete in {time.time() - total_start:.1f}s.")


if __name__ == "__main__":
    main()
