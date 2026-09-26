# Business Entity Resolution

An enterprise-scale entity resolution and record linkage pipeline designed to resolve ambiguous business entity records across heterogeneous data sources (Source 1, Source 2, and Source 3) into unified ground truth clusters.

---

## 1. Project Architecture

The repository implements a modular, high-throughput pipeline designed for execution both in local Python environments and on distributed cloud infrastructure (such as AWS SageMaker):

```
business_entity_resolution/
├── src/
│   ├── preprocessing/     # Canonical normalization (names, addresses, country codes)
│   ├── blocking/          # Scalable candidate pair generation & indexers (TF-IDF, MinHash/LSH)
│   ├── features/          # Pairwise similarity featurizers (string distance, token overlap, geo)
│   ├── modeling/          # Binary classifiers & rankers (LightGBM, XGBoost, threshold tuning)
│   └── utils/             # I/O utilities, metric calculators, evaluation harnesses
├── notebooks/             # Milestone workflows & exploratory data audits
│   └── 00_data_audit.ipynb # Data schema audit, prefix checks, singleton & noise pattern analysis
├── data/
│   ├── train/             # Training partitions (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/              # Evaluation partitions (test_source1/2/3.tsv)
├── output/                # Predictions, candidate pairs, models, evaluation logs
├── README.md              # Project documentation & guidelines
└── requirements.txt       # Environment specifications
```

---

## 2. Setup & Installation

### Environment Setup
Create and activate an isolated Python 3.10+ virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Data Ingestion & Directory Configuration

The pipeline ingests tab-separated value (`.tsv`) files with `sep="\t"`:

- **Source 1 (`train_source1.tsv` / `test_source1.tsv`)**: Anchor table containing primary entity records (`entity_id`, `business_name`, `business_address`, `country`).
- **Source 2 & Source 3 (`train_source2.tsv`, `train_source3.tsv`)**: Secondary tables containing candidate matching records.
- **Ground Truth (`train_ground_truth.tsv`)**: Maps each `source1_entity_id` to a comma-separated list of `matched_entity_ids`.

Data can be placed either inside `business_entity_resolution/data/train/` and `data/test/` or referenced directly from the challenge folder `student_resource/dataset/`. The notebooks and pipeline modules dynamically detect available data roots.

---

## 4. Key Findings & Design Constraints

From the exploratory data audit ([`00_data_audit.ipynb`](notebooks/00_data_audit.ipynb)):

1. **ID Prefix Guarantees**:
   - Source 1 entity IDs strictly follow the `S1-` prefix.
   - Source 2 entity IDs strictly follow the `S2-` prefix.
   - Source 3 entity IDs strictly follow the `S3-` prefix.

2. **Crucial Country Constraint ("France" in Test)**:
   - The training set contains only two countries: **United States** (~60%) and **India** (~40%).
   - The test set introduces **France** (~15% of records).
   - **Constraint**: Normalization and feature pipelines must treat `country` as an open string attribute without hardcoding fixed categorical enums or one-hot vectors.

3. **Singleton Entities**:
   - ~5.58% (123,247) of Source 1 entities have **0 matches** (singletons) in Source 2 and Source 3.
   - Match distributions range from 0 to 11 matches per entity, with an average of ~3.46 matches.
   - Ground truth matches are nearly equally distributed between Source 2 (~48.4%) and Source 3 (~51.6%).

4. **Noise Taxonomy in Real Matched Pairs**:
   - **Legal Suffixes**: Variations such as `INC`, `CORP`, `LLC`, `PVT LTD`, `LIMITED`, `L.L.C.`.
   - **Street & Directional Tokens**: `ST` vs `STREET`, `RD` vs `ROAD`, `STE` vs `SUITE`, `AVE` vs `AVENUE`.
   - **Punctuation & Ampersands**: `&` vs `AND`, trailing hyphens, commas.
   - **Address Inversions & Reordering**: Building names preceding street names vs street names followed by suite numbers.

---

## 5. Development Roadmap

- [x] **Milestone 0**: Project Scaffolding & Comprehensive Data Audit (`00_data_audit.ipynb`)
- [ ] **Milestone 1**: Robust Preprocessing & Normalization Engine (`normalize.py`)
- [ ] **Milestone 2**: Candidate Generation & High-Recall Blocking (`blocking.py`)
- [ ] **Milestone 3**: Feature Extraction & Pairwise Similarity Metrics
- [ ] **Milestone 4**: Supervised Model Training & Threshold Calibration
- [ ] **Milestone 5**: End-to-End Inference Pipeline & Submission Verification
