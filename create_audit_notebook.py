import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(content):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [line + "\n" for line in content.split("\n")]
    })

def add_code(content):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else [line + "\n" for line in content.split("\n")]
    })

# --- Title and Overview ---
add_md("""# Milestone 0: Comprehensive Data Audit & Profiling
## Amazon ML Challenge 2026 - Business Entity Resolution

This notebook performs a deep forensic data audit on all raw datasets:
- `train_source1.tsv`
- `train_source2.tsv`
- `train_source3.tsv`
- `train_ground_truth.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`

### Objectives:
1. Verify tab-separated encoding, file schemas, shapes, column types, and null counts.
2. Confirm strict entity ID prefix formatting (`S1-`, `S2-`, `S3-`).
3. Audit the ground truth topology: singleton percentage, match count distribution, and source origin of matches.
4. Uncover the **open-set country distribution shift** between train and test sets (the "France" factor).
5. Extract and analyze 15 representative matched pairs side-by-side to catalog real-world noise patterns.""")

# --- Setup and Imports ---
add_md("""## 1. Setup & Environment""")
add_code("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set dataset paths
TRAIN_DIR = Path("../../student_resource/dataset/train")
TEST_DIR = Path("../../student_resource/dataset/test")

print("Train directory exists:", TRAIN_DIR.exists())
print("Test directory exists:", TEST_DIR.exists())""")

# --- Section 1: File Loading & Basic Profiling ---
add_md("""## 2. File Loading, Shapes, Dtypes & Null Checks
All source files are loaded using explicit `sep='\\t'` with string dtypes to prevent precision loss in IDs or postal codes.""")

add_code("""# Load train files with explicit tab separator
train_files = {
    'source1': TRAIN_DIR / 'train_source1.tsv',
    'source2': TRAIN_DIR / 'train_source2.tsv',
    'source3': TRAIN_DIR / 'train_source3.tsv',
    'ground_truth': TRAIN_DIR / 'train_ground_truth.tsv'
}

for name, path in train_files.items():
    size_mb = os.path.getsize(path) / (1024 * 1024)
    # Read first 5 rows for schema and head inspection
    head_df = pd.read_csv(path, sep='\\t', nrows=5, dtype=str, keep_default_na=False)
    print(f"=== {name.upper()} ({path.name}) ===")
    print(f"File size: {size_mb:.2f} MB")
    print(f"Columns: {list(head_df.columns)}")
    display(head_df.head(2))
    print("-" * 50)""")

add_code("""# Compute exact row counts, memory footprints, and null counts
summary_stats = []

for name, path in train_files.items():
    df = pd.read_csv(path, sep='\\t', dtype=str, keep_default_na=False)
    null_counts = (df == '').sum().to_dict()
    summary_stats.append({
        'Table': name,
        'Rows': len(df),
        'Cols': len(df.columns),
        'Empty / Null Cells': sum(null_counts.values()),
        'Null Details': null_counts
    })

pd.DataFrame(summary_stats)""")

# --- Section 2: Entity ID Prefix Validation ---
add_md("""## 3. Entity ID Prefix Integrity Verification
Verify that:
- `train_source1.tsv` entities strictly start with `S1-`
- `train_source2.tsv` entities strictly start with `S2-`
- `train_source3.tsv` entities strictly start with `S3-`
- `train_ground_truth.tsv` source entities strictly start with `S1-`""")

add_code("""s1 = pd.read_csv(train_files['source1'], sep='\\t', dtype=str)
gt = pd.read_csv(train_files['ground_truth'], sep='\\t', dtype=str, keep_default_na=False)

s1_valid = s1['entity_id'].str.startswith('S1-').all()
gt_valid = gt['source1_entity_id'].str.startswith('S1-').all()

# Check S2 and S3 via streaming chunks
s2_valid = all(chunk['entity_id'].str.startswith('S2-').all() for chunk in pd.read_csv(train_files['source2'], sep='\\t', dtype=str, chunksize=1000000))
s3_valid = all(chunk['entity_id'].str.startswith('S3-').all() for chunk in pd.read_csv(train_files['source3'], sep='\\t', dtype=str, chunksize=1000000))

print("Prefix Validation Results:")
print(f"  Source 1 IDs all start with 'S1-': {s1_valid}")
print(f"  Source 2 IDs all start with 'S2-': {s2_valid}")
print(f"  Source 3 IDs all start with 'S3-': {s3_valid}")
print(f"  Ground Truth S1 IDs all start with 'S1-': {gt_valid}")""")

# --- Section 3: Ground Truth Distribution & Singletons ---
add_md("""## 4. Ground Truth Topology: Singleton Percentage & Match Distribution
A critical parameter of the problem is the distribution of matches per Source 1 entity.""")

add_code("""# Analyze matched_entity_ids column
def parse_match_count(val):
    if not val or val == '':
        return 0
    return len(val.split(','))

match_counts = gt['matched_entity_ids'].apply(parse_match_count)

total_s1 = len(match_counts)
singletons = (match_counts == 0).sum()
singleton_pct = (singletons / total_s1) * 100

print(f"Total Source 1 Entities: {total_s1:,}")
print(f"Singletons (0 matches):  {singletons:,} ({singleton_pct:.2f}%)")
print(f"Entities with matches:   {total_s1 - singletons:,} ({100 - singleton_pct:.2f}%)")
print(f"Min matches: {match_counts.min()}, Max matches: {match_counts.max()}")
print(f"Mean matches: {match_counts.mean():.2f}, Median: {match_counts.median():.0f}, 95th Percentile: {match_counts.quantile(0.95):.0f}")""")

add_code("""# Match count frequency distribution
dist_df = match_counts.value_counts().sort_index().reset_index()
dist_df.columns = ['Num Matches', 'Count']
dist_df['Percentage (%)'] = (dist_df['Count'] / total_s1 * 100).round(2)
display(dist_df)

# Plot distribution
plt.figure(figsize=(10, 4))
plt.bar(dist_df['Num Matches'], dist_df['Count'], color='#1f77b4', edgecolor='black', alpha=0.8)
plt.title('Distribution of Number of Matches per Source 1 Entity', fontsize=12, fontweight='bold')
plt.xlabel('Number of Matched Entities')
plt.ylabel('Count of Source 1 Entities')
plt.xticks(dist_df['Num Matches'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()""")

add_code("""# Deconstruct target matches: Source 2 vs Source 3
s2_match_count = 0
s3_match_count = 0

for val in gt['matched_entity_ids']:
    if not val or val == '':
        continue
    for m in val.split(','):
        if m.startswith('S2-'):
            s2_match_count += 1
        elif m.startswith('S3-'):
            s3_match_count += 1

total_target_matches = s2_match_count + s3_match_count
print(f"Total Match Links: {total_target_matches:,}")
print(f"  Source 2 Matches: {s2_match_count:,} ({s2_match_count / total_target_matches * 100:.2f}%)")
print(f"  Source 3 Matches: {s3_match_count:,} ({s3_match_count / total_target_matches * 100:.2f}%)")""")

# --- Section 4: Country Distribution & "France" Open-Set Shift ---
add_md("""## 5. Country Distribution & The "France" Open-Set Shift

> ### ⚠️ CRITICAL ARCHITECTURAL CONSTRAINT: Open-Set Country Domain Shift
> In the training set, all records come exclusively from **US** (~60%) and **India** (~40%).
> In the test set, **France** (~15%) is introduced alongside India (~47%) and US (~38%).
>
> **Design Rules:**
> 1. **Never** use closed categorical encoders (e.g. `LabelEncoder`, fixed one-hot vectors) that assume only `{'US', 'India'}`.
> 2. **Never** hardcode country names in conditional branching for blocking or feature logic.
> 3. All country features must be formulated as **pairwise relation predicates**:
>    - `country_exact_match = (country_1 == country_2)`
>    - Standardized ISO-country normalization.
> 4. Blocking must perform soft or exact country partitioning that adapts dynamically to any country string present at test time.""")

add_code("""# Inspect country values in Train vs Test
def get_country_counts(tsv_path):
    df = pd.read_csv(tsv_path, sep='\\t', usecols=['country'], dtype=str)
    return df['country'].value_counts()

print("--- TRAIN SOURCE 1 COUNTRY BREAKDOWN ---")
print(get_country_counts(train_files['source1']))

print("\\n--- TEST SOURCE 1 COUNTRY BREAKDOWN ---")
print(get_country_counts(TEST_DIR / 'test_source1.tsv'))

print("\\n--- TEST SOURCE 2 COUNTRY BREAKDOWN ---")
print(get_country_counts(TEST_DIR / 'test_source2.tsv'))

print("\\n--- TEST SOURCE 3 COUNTRY BREAKDOWN ---")
print(get_country_counts(TEST_DIR / 'test_source3.tsv'))""")

# --- Section 5: Side-by-Side Analysis of 15 Matched Pairs ---
add_md("""## 6. Deep Forensic Analysis of 15 Representative Matched Pairs
Here we examine 15 ground-truth matched pairs randomly sampled across US and India entities to catalog noise, corruption, and variation patterns.""")

# Load sampled pairs from json
with open('sampled_pairs.json', 'r', encoding='utf-8') as f:
    sample_pairs = json.load(f)

pairs_md = ["| # | Source 1 (Name & Address) | Matched Entity (Name & Address) | Observed Noise Patterns |",
            "|---|---|---|---|"]

noise_notes = [
    "Typo ('Offie' vs 'Office'), Case variation, Street abbreviation ('CIR' vs 'Circle').",
    "Legal entity suffix variation ('Corp' vs '[Consultancy]'), Bracket noise, Address case variation.",
    "Address token reordering ('TX, 158 Simpson Lane' vs '158 Simpson Lane... Texas'), State name expansion.",
    "Character level OCR/typo corruption ('Team' vs 'Thg'), Special unicode address characters.",
    "Punctuation noise ('##19821' vs '19821'), Street abbreviation ('DR' vs 'Drive'), Extra whitespace.",
    "Legal suffix mutation ('Connecticut LLC' vs 'LLC Service'), State prefix relocation ('IL' at end vs 'Illinois' at start).",
    "Token order inversion ('Capital Partners PLLC' vs 'CAPITAL PLLC-PARTNERS'), Period punctuation ('120.').",
    "Extra whitespace padding ('Chem  LLP'), Address capitalization.",
    "Typo corruption ('Rceoad' vs 'Record'), Parentheses noise ('(Limited)'), State abbreviation ('MP' vs 'Madhya Pradesh').",
    "Abbreviation hyphenation ('Pvt-Ltd.' vs 'Pvt Ltd'), State abbreviation ('RJ' vs 'Rajasthan'), Address truncation.",
    "Whitespace variation, Address case difference.",
    "**Domain Name Conversion**: Name replaced by website URL ('cardiologymetrocare.com' vs 'Cardiology Metro Care Associates Inc').",
    "Extreme name divergence / Alias ('Shri Verafayemira' vs 'Achyut Colonisers Pvt Ltd'), Identical address.",
    "Prefix noise ('#118' vs 'Plot No.118'), Legal suffix variation ('LIMITED SERVICE' vs 'Private Limited').",
    "OCR corruption / placeholder ('Sarasva ?ndia' vs 'Sarasva India Limited'), Prefix addition ('DOOR NO')."
]

for p, note in zip(sample_pairs, noise_notes):
    s1_str = f"**[{p['s1_id']} ({p['s1_country']})]**<br>`{p['s1_name']}`<br>*{p['s1_address']}*"
    m_str = f"**[{p['match_id']} ({p['match_country']})]**<br>`{p['match_name']}`<br>*{p['match_address']}*"
    # Clean pipes to avoid markdown table breaks
    s1_str = s1_str.replace("|", "/")
    m_str = m_str.replace("|", "/")
    pairs_md.append(f"| {p['pair_idx']} | {s1_str} | {m_str} | {note} |")

add_md("\n".join(pairs_md))

# --- Section 6: Key Findings & Strategy ---
add_md("""## 7. Key Findings & Engineering Roadmap

### Summary of Findings:
1. **Scale & Footprint**:
   - Source 1: ~2.2M train, ~1.7M test.
   - Source 2: ~5.0M train, ~4.9M test.
   - Source 3: ~5.3M train, ~5.1M test.
   - **Strategy**: Pure in-memory cross-joins are infeasible. We must use streaming/chunking, convert raw TSVs to compressed columnar Parquet (`data/processed/`), and execute highly selective blocking.
2. **Topology**:
   - Singletons comprise **5.58%** of Source 1. The vast majority of entities have between 2 and 5 matches.
   - Matches are almost evenly split between Source 2 (48.4%) and Source 3 (51.6%).
3. **Open-Set Domain Shift**:
   - France makes up **~15%** of test records, completely absent in train.
   - All country-based logic must be relational and generic.
4. **Noise Patterns**:
   - Legal form variations (`LLC`, `Inc`, `Corp`, `Pvt Ltd`, `Limited`, `PLLC`, `Co`).
   - Domain names as company names (`something.com` vs `Something Inc`).
   - State & street abbreviations (`TX` -> `Texas`, `St` -> `Street`, `Cir` -> `Circle`, `Dr` -> `Drive`).
   - Landmark and prefix noise (`Door No`, `Plot No`, `#`, `##`, `Opp`, `Near`).
   - Typos and character mutations (`Offie`, `Rceoad`, `?ndia`).

### Next Step: Milestone 1
- Build `src/preprocessing/normalize.py` with robust regex-based cleaners for business names, addresses, and countries.
- Build comprehensive unit test suite in `tests/test_normalize.py`.
- Build `src/preprocessing/build_normalized_tables.py` to stream all raw TSVs into `data/processed/*.parquet`.""")

# Save notebook
notebook_path = "business_entity_resolution/notebooks/00_data_audit.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully generated {notebook_path}")
