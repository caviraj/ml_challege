# Antigravity Prompt Playbook — Business Entity Resolution Challenge

**Setup context:** Coding agent = Antigravity. Model training = AWS SageMaker (Antigravity only needs to produce the **notebook** — actual training run happens on SageMaker).

Paste each phase's prompt into Antigravity **one at a time**, in order. Don't skip ahead — each phase assumes the previous phase's files already exist in the repo. Review the diff/output before moving to the next phase.

---

## Milestone 0 — Project Scaffolding & Data Audit

**Goal:** Repo structure ready, raw data explored, no modeling yet.

```
Set up a Python project called `business_entity_resolution` with this exact structure:

business_entity_resolution/
├── src/
│   ├── preprocessing/
│   ├── blocking/
│   ├── features/
│   ├── modeling/
│   └── utils/
├── notebooks/
├── data/
│   ├── train/
│   └── test/
├── output/
├── README.md
└── requirements.txt

Then write a data exploration notebook at notebooks/00_data_audit.ipynb that:
1. Loads train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
   with sep="\t" explicitly (these files are tab-separated, NOT comma).
2. Prints shape, dtypes, null counts, and 5 sample rows per file.
3. Checks entity_id prefix consistency (S1-/S2-/S3-).
4. Parses train_ground_truth.tsv's matched_entity_ids column (comma-separated) and reports:
   - % of Source 1 entities with zero matches (singletons)
   - distribution of match counts per Source 1 entity
   - how many matches come from Source 2 vs Source 3
5. Inspects the `country` column distribution in each source file (note: test set
   includes an extra country "France" not present in train — flag this explicitly
   in a markdown cell as a design constraint, since our pipeline must NOT hardcode
   to {US, India}).
6. Samples 15 random matched pairs from ground truth and prints their business_name
   and business_address side by side, so we can visually catalog noise patterns
   (abbreviations, typos, transliteration, address reordering).

Use pandas, and keep the notebook read-only / no writes to data files.
```

---

## Milestone 1 — Preprocessing & Normalization

**Goal:** Reusable normalization functions for name and address fields.

```
In src/preprocessing/, create normalize.py with these functions, all pure and unit-testable:

1. normalize_name(name: str) -> str
   - lowercase, strip punctuation except alphanumerics/spaces
   - expand common legal-suffix abbreviations to a canonical form
     (corp/corporation, pvt/private, ltd/limited, inc/incorporated, co/company)
   - normalize "&" and "and" to the same token
   - collapse multiple spaces, strip

2. normalize_address(address: str) -> dict
   - lowercase, strip punctuation
   - expand common street-type abbreviations (rd->road, st->street, ave->avenue, etc.)
   - attempt to extract components where possible: street_tokens, city, state, postal_code
   - handle missing components gracefully (return None for missing fields, don't error)
   - keep landmark phrases ("near sbi atm") as a separate raw_landmark field rather than
     discarding them — they can be a weak signal later

3. normalize_country(country: str) -> str
   - lowercase/strip only. Do NOT map to a fixed enum or one-hot here — country must stay
     an open string field since test data includes "France" which never appears in train.

Write src/preprocessing/build_normalized_tables.py that loads all 6 source files
(train + test, source1/2/3), applies these functions, and writes normalized
parquet files to data/processed/ with columns:
entity_id, source, name_norm, address_norm (as struct/dict columns), country_norm, raw_name, raw_address

Add unit tests in tests/test_normalize.py covering at least: legal suffix variants,
"&" vs "and", missing address components, and an unseen country string.
```

---

## Milestone 2 — Blocking / Candidate Generation

**Goal:** Produce `candidate_pairs.tsv`-ready candidate sets with a good recall ceiling before touching the ML model.

```
In src/blocking/, build a candidate generation module:

1. blocking.py should implement a blocking strategy that, for every Source 1 entity,
   returns a shortlist of Source 2 + Source 3 entity_ids to consider, using a
   combination of:
   - TF-IDF vectorization on normalized business_name + top-K cosine similarity
     (use sklearn's TfidfVectorizer + NearestNeighbors, no external APIs)
   - token/n-gram blocking keys (e.g. first 3 chars of sorted name tokens) as a
     fallback/union to catch cases TF-IDF misses
   - loose country-based soft filtering (don't hard-exclude — down-weight only,
     since country strings are noisy/open-set)

2. Make top-K and similarity threshold configurable via a config dict/yaml, not hardcoded.

3. Write src/blocking/generate_candidates.py, a CLI script that:
   - reads data/processed/*.parquet
   - runs blocking for a given split (train or test)
   - writes candidate_pairs.tsv in the exact competition format:
     source1_entity_id <TAB> candidate_entity_ids (comma-separated, no duplicates)
   - prints recall against train_ground_truth.tsv (how many true matches are inside
     the candidate set) and the reduction ratio (candidates generated vs full cross join),
     so we know the recall ceiling before modeling.

Run it on the train split and report recall — target should be >97% recall on train
before we move to modeling, since blocking recall caps the whole pipeline's ceiling.
```

---

## Milestone 3 — Pairwise Feature Engineering

**Goal:** Turn each (Source1, candidate) pair into a feature vector for the model.

```
In src/features/, create pairwise_features.py that, given a Source1 record and a
candidate record (both normalized), computes a feature dict:

Name features:
- levenshtein_ratio, jaccard_token_similarity, token_sort_ratio,
  tfidf_cosine_similarity (reuse vectorizer fit during blocking if possible),
  common_token_count, name_length_diff

Address features:
- street-token jaccard/levenshtein, city exact-match flag, postal_code exact-match flag
  (None-safe: missing components should produce a neutral/NaN-safe feature, not crash),
  full-address levenshtein_ratio as fallback when component parsing failed

Country feature:
- exact_match flag (works for any country string, including "france" at test time)

Write src/features/build_feature_table.py, a CLI script that:
- reads candidate_pairs.tsv + normalized parquet tables
- explodes each Source1->candidates row into individual pair rows
- computes the feature dict per pair
- joins the label (1/0) from train_ground_truth.tsv when building train features
  (label=1 if that S2/S3 id is in the ground-truth matched list for that S1 id, else 0)
- writes data/processed/train_pairs_features.parquet and, separately, a version for
  test candidates (no label column)

Print class balance (%positive vs %negative pairs) after building train features.
```

---

## Milestone 4 — Model Training Notebook for SageMaker

**Goal:** Antigravity should produce ONLY the notebook + training script; the actual SageMaker training job execution happens outside Antigravity, on AWS.

```
Create notebooks/01_train_matching_model.ipynb designed to run in an AWS SageMaker
notebook instance (assume boto3/sagemaker SDK and standard sklearn/xgboost/lightgbm
are available; do not assume internet access to any external entity-resolution API
or service — none is allowed per competition rules).

The notebook should:
1. Load data/processed/train_pairs_features.parquet.
2. Do a GroupShuffleSplit or GroupKFold split, GROUPED BY source1_entity_id (not a
   random row split), so we don't leak the same Source1 entity's pairs across
   train/validation. Hold out ~15-20% of Source1 entities for validation.
3. Train a gradient-boosted classifier (choose one: XGBoost, LightGBM, or
   scikit-learn GradientBoosting — pick whichever has an MIT/Apache-licensed
   implementation and keep total model size well under 8B params, this is a
   tabular model so that ceiling is trivial to satisfy) predicting match probability
   per pair.
4. Do NOT just optimize AUC/accuracy — implement a custom scoring function that
   reconstructs, per Source1 entity in the validation fold, the predicted match set
   at a given probability threshold, and computes F_0.5 per entity (including
   singletons scoring 1.0 for correctly predicting empty), then macro-averages.
   Sweep thresholds (e.g. 0.05 to 0.95 step 0.05) and plot validation F_0.5 vs
   threshold — pick the threshold that maximizes macro F_0.5, remembering F_0.5
   favors precision 2x over recall, so expect the optimal threshold to sit above
   the naive 0.5.
5. Save the trained model artifact (joblib/pickle or native format) to
   a models/ directory with a versioned filename, plus a metadata.json recording:
   model type, hyperparameters, chosen threshold, validation F_0.5, feature list,
   library name+version+license.
6. Include markdown cells documenting each step for the methodology writeup later.

Also produce src/modeling/train.py as a plain script version of the same logic
(no notebook-only code) so the training is reproducible outside Jupyter too, per
the submission package's reproducibility requirement.
```

*(You run the notebook/script on the SageMaker instance yourself — Antigravity just needs to generate correct, runnable code here, not execute the training job.)*

---

## Milestone 5 — Inference & Submission File Generation

**Goal:** Turn model + test candidates into the two required output files.

```
Create src/modeling/predict.py and notebooks/02_generate_submission.ipynb that:

1. Load the trained model artifact + metadata.json (chosen threshold) from Milestone 4.
2. Load data/processed/test_pairs_features.parquet (built from test candidate_pairs.tsv,
   no labels).
3. Score every pair, apply the chosen threshold to decide match/no-match.
4. Reconstruct matching_results.tsv:
   - exactly one row per Source1 entity_id present in test_source1.tsv (even if it
     had zero candidates from blocking — still must appear, with empty matched_entity_ids)
   - matched_entity_ids = comma-separated, deduplicated, sorted for determinism
   - only include S2-/S3- ids that exist in the test set
5. Copy the test-time candidate_pairs.tsv (from Milestone 2, run on the test split)
   into output/candidate_pairs.tsv unchanged — this is what's submitted, not regenerated.
6. Assert as a script-level sanity check: every id in matching_results.tsv appears
   in candidate_pairs.tsv for that same source1_entity_id (matches ⊆ candidates),
   and raise a clear error if not.
7. Write both files to output/ with sep="\t", no index, no quoting.

Finally, run:
python3 utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir dataset/test
and paste the output back to me.
```

---

## Milestone 6 — Local F_0.5 Self-Evaluation Harness

**Goal:** Score your own held-out validation split the same way the leaderboard will, before you burn a submission.

```
Create src/evaluation/score_f05.py, a standalone script that:
- takes a predicted matching_results-style tsv and a ground-truth-style tsv
  (same schema as train_ground_truth.tsv) as arguments
- computes per-Source1-entity precision, recall, F_0.5 (correctly scoring
  singletons as 1.0/0.0 per the competition's exact rule)
- macro-averages across all entities
- prints overall F_0.5 plus a breakdown: mean F_0.5 for singleton entities only,
  and mean F_0.5 for entities that do have true matches, so we can see whether
  errors are concentrated in false merges vs missed matches

Run this against your Milestone 4 validation fold predictions (not the real test set,
since we have no test ground truth) and report the number.
```

---

## Milestone 7 — Methodology Documentation & Final Packaging

**Goal:** Assemble the exact submission zip structure required.

```
Fill in Documentation_template.md (copy it into the repo root if not present) covering:
- Methodology overview
- Blocking/candidate generation strategy (recall ceiling and reduction ratio numbers
  from Milestone 2)
- Model architecture + feature engineering (pull details from models/metadata.json)
- Validation F_0.5 results from Milestone 6
- Any limitations / things we'd improve with more time

Then write a script scripts/build_submission_zip.py that assembles:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

README.md inside code/business_entity_resolution/ must give exact, copy-pasteable
commands to reproduce the full pipeline end-to-end from raw data to output/ files,
including which steps run locally vs which notebook needs to run on a SageMaker
instance. requirements.txt must pin exact versions of every library used.

Confirm no code anywhere in src/ makes any network call to an external API,
geocoding service, or entity-resolution service — grep for requests./urllib/boto3
calls outside the SageMaker training step and list every hit so we can manually
verify fair-play compliance before zipping.
```

---

## Notes on Using Plugins Wisely

- Run **Milestone 0's data audit prompt first, standalone** — don't bundle it with Milestone 1, so you can eyeball the noise patterns before writing normalization rules.
- For Milestones 1–3 (pure Python/pandas), let Antigravity use its file-edit + terminal/test-run plugins to actually execute and fix its own code rather than just writing it blind — ask it to "run the tests and fix failures" as a follow-up in the same phase before moving on.
- For Milestone 4, treat Antigravity purely as a **code generator** — do not let it try to execute SageMaker training itself; run that notebook/script yourself in the SageMaker instance, then bring the resulting `models/metadata.json` back into the repo before starting Milestone 5.
- Re-run Milestone 6's scorer after any change to blocking (M2), features (M3), or the model (M4) — treat it as your regression check.
- Only run Milestone 7 once Milestone 6's validation F_0.5 is at a number you're happy with — packaging early just means repackaging later.
