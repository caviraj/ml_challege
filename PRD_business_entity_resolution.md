# PRD: Business Entity Resolution Challenge

## 1. Overview

**Problem:** Build an ML system that matches business records coming from **3 independent data sources** (Source 1, Source 2, Source 3) that describe the same real-world business, even though the records have **no shared ID** and are full of noisy/inconsistent fields (typos, abbreviations, missing address parts, etc.).

- **Source 1** = deduplicated reference source (the "anchor" list of unique businesses).
- **Source 2 / Source 3** = other sources that may contain 0, 1, or many records matching each Source 1 entity.

**Goal:** For every Source 1 entity, output the list of Source 2/Source 3 entity_ids that refer to the same business.

---

## 2. Data

### 2.1 Input files (all `.tsv`, tab-separated — NOT comma)
| File | Description |
|---|---|
| `dataset/train/train_source1.tsv` | Source 1 training records |
| `dataset/train/train_source2.tsv` | Source 2 training records |
| `dataset/train/train_source3.tsv` | Source 3 training records |
| `dataset/train/train_ground_truth.tsv` | Ground truth labels |
| `dataset/test/test_source1.tsv` | Source 1 test records |
| `dataset/test/test_source2.tsv` | Source 2 test records |
| `dataset/test/test_source3.tsv` | Source 3 test records |

### 2.2 Columns in each source file
| Column | Notes |
|---|---|
| `entity_id` | Prefixed `S1-`, `S2-`, `S3-` (indicates source; no separate source column) |
| `business_name` | Abbreviations, typos, transliterations, DBA names |
| `business_address` | Partial addresses, landmark refs, format variation |
| `country` | Train = {US, India}. **Test also has France** — must NOT hardcode to only US/India |

### 2.3 Ground truth (`train_ground_truth.tsv`)
| Column | Notes |
|---|---|
| `source1_entity_id` | Source 1 entity |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; empty = no match (singleton) |

### 2.4 Noise to handle
- **Name:** `Corp` vs `Corporation`, `Pvt` vs `Private`, `Ltd` vs `Limited`, `&` vs `and`, word-order swaps, typos, DBA/trade names.
- **Address:** `Rd` vs `Road`, `St` vs `Street`, transliteration variants, missing PIN/state, landmark refs ("Near SBI ATM"), municipal numbering, reordered components.

---

## 3. Required Outputs (in `output/` folder)

### 3.1 `matching_results.tsv` — **the only file scored on leaderboard**
| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 entity |
| `matched_entity_ids` | Comma-separated S2/S3 IDs (empty if singleton) |

**Hard rules:**
1. Exactly **one row per Source 1 test entity** (all must be present).
2. Empty `matched_entity_ids` for no-match cases.
3. **No duplicate IDs** within a row's list.
4. IDs must be valid S2-/S3- IDs that exist in the **test** set (no Source 1 self-matches, no invented IDs).
5. No duplicate `source1_entity_id` rows.

### 3.2 `candidate_pairs.tsv` — NOT scored, used to audit blocking quality
| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 entity |
| `candidate_entity_ids` | Comma-separated candidate S2/S3 IDs considered before final scoring |

- This must be the **final** candidate set fed to the matching model at inference (last stage of blocking, not an early raw pass).
- Every ID that appears in `matching_results.tsv` **must** appear in `candidate_pairs.tsv` for that same Source 1 entity (matches ⊆ candidates).

### 3.3 Validation
Run before every submission:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Prints `PASS` (exit 0) or a numbered issue list (exit 1). It checks format only — not your F_0.5 score.

---

## 4. Evaluation Metric

**F_0.5 score**, macro-averaged **per Source 1 entity**, then averaged across all entities.

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Precision weighted **2× more than recall** → false merges (wrong match) hurt more than missed matches.
- **Singletons matter:** a Source 1 entity with no true matches scores **1.0** if you correctly predict empty, **0.0** if you predict any match for it.
- Use a **held-out validation split** from training data to self-score (no test ground truth given).

---

## 5. Suggested Pipeline Architecture

### Stage A — Preprocessing / Normalization
- Lowercase, strip punctuation, normalize legal suffixes (`Corp.` → `corp`, `Pvt Ltd` → `pvt ltd`), expand common abbreviations (`Rd` → `road`, `St` → `street`).
- Normalize addresses: tokenize into components (street, city, state, PIN/zip) where possible; handle missing components gracefully.
- Keep `country` as an **open-set string field** — do not filter/one-hot to only US/India (test includes France).

### Stage B — Blocking / Candidate Generation (determines recall ceiling)
- Goal: cheaply narrow down, for each Source 1 record, a manageable candidate set from Source 2 + Source 3 (avoid full O(N×M) comparison).
- Techniques: token/n-gram blocking on business name, TF-IDF + approximate nearest neighbor (e.g., cosine similarity top-K), phonetic keys (Soundex/Metaphone), geographic/address token blocking, country-based partitioning (but not exclusion).
- Output = `candidate_pairs.tsv` (this is scored for blocking quality, not accuracy).

### Stage C — Pairwise Feature Engineering
- Name similarity: Jaccard, Levenshtein/edit distance, token-sort ratio, TF-IDF cosine, common-token overlap.
- Address similarity: same metrics + component-level matching (street, city, PIN if present).
- Country match (exact/boolean feature — remember France appears only at test time, so features must generalize, not memorize country values).

### Stage D — Matching Model
- Binary/probability classifier over candidate pairs (match vs non-match) OR a ranking/clustering approach.
- **Constraint: model must be MIT/Apache 2.0 licensed and ≤ 8B parameters.**
- Tune the decision threshold toward **precision** (since F_0.5 penalizes false positives 2×).
- Explicitly handle the "predict empty list" case for likely singletons.

### Stage E — Post-processing
- Enforce output constraints: no self-source1 matches, no duplicate IDs, one row per Source 1 test entity, IDs must exist in test set.
- Ensure `matching_results.tsv` matches are a strict subset of `candidate_pairs.tsv`.

---

## 6. Constraints & Fair-Play Rules

1. **No external data/API lookups** — no entity resolution APIs, no government registry lookups, no geocoding APIs, no internet data augmentation. Any evidence → immediate disqualification.
2. Final model: **MIT/Apache 2.0 license, ≤ 8B parameters.**
3. Output format must exactly match spec or submission is rejected (not scored).
4. Every Source 1 test entity must appear in the submission (missing → rejection).

---

## 7. Final Submission Package Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          # exact reproduction steps: data → blocking → matching → output
│       └── requirements.txt   # pinned dependencies
└── Documentation_template.md  # filled-in methodology write-up
```

Methodology doc must cover:
- Overall methodology
- Blocking/candidate generation strategy
- Model architecture + feature engineering
- Any other relevant approach details
- No page limit — prioritize clarity/depth over brevity.

---

## 8. Leaderboard Process

- **Public leaderboard:** live feedback during challenge, scored on a test subset.
- **Private leaderboard:** revealed after challenge ends, scored on remaining test subset — **this decides final ranking**.
- You submit predictions for the **full test set** both times; splitting happens on the scoring side.
- Top teams' final zip packages get manually audited (fair play + license/param checks) before final rankings are confirmed.

---

## 9. Success Checklist (Definition of Done)

- [ ] Preprocessing pipeline normalizes name/address across all 3 sources, generalizes to unseen country (France).
- [ ] Blocking stage produces `candidate_pairs.tsv` with high recall ceiling and reasonable reduction ratio.
- [ ] Matching model is trained/validated on a held-out split, self-scored with F_0.5.
- [ ] Precision-oriented threshold tuning done (F_0.5 favors precision).
- [ ] Singleton (no-match) prediction handled explicitly and validated.
- [ ] `matching_results.tsv` and `candidate_pairs.tsv` pass `validate_submission.py` with PASS.
- [ ] Matches ⊆ candidates verified.
- [ ] Model is MIT/Apache-2.0, ≤ 8B params.
- [ ] No external lookups/APIs used anywhere in pipeline.
- [ ] Final zip assembled per required structure with README, requirements.txt, and filled methodology doc.
