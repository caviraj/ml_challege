#!/usr/bin/env python3
"""
Submission Validator for Business Entity Resolution (ML Challenge 2026).
Compatible with both test_source1.tsv and test_source1.parquet!
"""

import argparse
import os
import sys

DELIM = "\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_ids_from_file(path):
    """Return set of S1 IDs from TSV or Parquet."""
    if path.endswith(".parquet"):
        try:
            import polars as pl
            df = pl.read_parquet(path)
            return set(df["entity_id"].to_list())
        except Exception as e:
            print(f"Error reading parquet with polars: {e}")
            import pandas as pd
            df = pd.read_parquet(path, columns=["entity_id"])
            return set(df["entity_id"].tolist())
    else:
        with open(path, encoding="utf-8") as f:
            next(f, None)
            return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def examples(items):
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def validate_id_list_file(path, expected_header, col_label, required, errors):
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping = {}
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix = set(), set()
    n_rows = empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but contains commas. "
                "Must be tab-separated (.tsv)."
            )
            return None
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(
                f"{name}: unexpected header {cols}. Expected exactly {expected_header}."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row at line {line_num}: {line.rstrip()!r}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)

    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}."),
        (intra_dupes, "{name}: repeated ID inside a {col} list for: {ex}."),
        (self_matches, "{name}: {col} contains Source-1 IDs (self-matches): {ex}."),
        (wrong_prefix, "{name}: {col} contains IDs without an S2-/S3- prefix: {ex}."),
        (required - seen, "{name}: required S1 entity(ies) missing: {ex}. Every S1 entity needs a row."),
        (seen - required, "{name}: row(s) using an S1 ID that is not in the test set: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))

    print(f"  {name}: {n_rows:,} rows ({empties:,} singletons, {n_rows - empties:,} non-empty).")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Validate ML Challenge 2026 submission files.")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-s1", "-t", default="data/processed/test_source1.parquet", help="Path to test_source1.parquet or test_source1.tsv")
    args = parser.parse_args()

    print("=== Submission Validator ===")
    print(f"Loading required S1 entities from {args.test_s1}...")
    if not os.path.isfile(args.test_s1):
        print(f"FAIL: Source 1 test file not found at {args.test_s1}")
        return 1

    required = read_ids_from_file(args.test_s1)
    print(f"Total required S1 entities: {len(required):,}")

    errors, warnings = [], []
    matched = validate_id_list_file(args.matching, MATCHING_HEADER, "matched_entity_ids", required, errors)
    candidate = validate_id_list_file(args.candidate, CANDIDATE_HEADER, "candidate_entity_ids", required, errors)

    if matched is not None and candidate is not None:
        offenders = {s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())}
        if offenders:
            warnings.append(
                f"{len(offenders)} S1 entity(ies) have matched IDs not present in candidate_pairs.tsv, e.g. {examples(offenders)}."
            )

    print()
    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        print(f"FAIL — {len(errors)} issue(s) found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1

    print("PASS — All validation rules passed! Ready for submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
