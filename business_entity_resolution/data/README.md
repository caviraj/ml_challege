# Dataset Directory Structure

This repository includes functional working sample datasets out-of-the-box so you can run the audit notebooks, feature engineering, and model training immediately without external downloads.

## Included Sample Datasets (`data/processed/sample/`)
- `train_ground_truth.parquet` (~4.3 MB) — Ground truth entity matches for the sample split
- `train_source1.parquet` (~13.8 MB) — Cleaned and normalized Source 1 records
- `train_source2.parquet` (~14.3 MB) — Cleaned and normalized Source 2 records
- `train_source3.parquet` (~14.0 MB) — Cleaned and normalized Source 3 records

## Full Competition Datasets (`data/processed/` & `student_resource/dataset/`)
The full raw `.tsv` files and uncompressed full `.parquet` files (~5.5 GB total, ranging from 120 MB to 770 MB per file) exceed GitHub's 100 MB per-file upload limit.

To train on the full dataset:
1. Place the full competition TSV files into `student_resource/dataset/train/` and `student_resource/dataset/test/`.
2. Run the preprocessing pipeline to generate the full normalized parquet files:
   ```bash
   python -m business_entity_resolution.src.preprocessing.build_normalized_tables
   ```
3. The normalized parquet files will be generated under `business_entity_resolution/data/processed/`.
