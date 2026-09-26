#!/bin/bash
# ==============================================================================
# AWS Pipeline Runner: Business Entity Resolution (ML Challenge 2026)
# ==============================================================================
# Automates:
# 1. Dependency installation
# 2. Model training & threshold tuning (LightGBM + GroupKFold)
# 3. Memory-efficient test inference (Country partitioned)
# 4. Strict submission validation (Matching & Candidate TSVs)
# 5. Packaging into submission.zip
# ==============================================================================

set -e

SAMPLE_SIZE=${1:-150000}
MODEL_PATH="models/lgb_model.joblib"
OUTPUT_DIR="output"
TEST_S1="data/processed/test_source1.parquet"

echo "=============================================================================="
echo " Starting ML Challenge Pipeline on AWS"
echo " Date: $(date)"
echo " System Info: $(uname -a)"
echo " CPU Cores: $(nproc)"
echo " Memory Free: $(free -h | grep Mem | awk '{print $4}')"
echo " Sample Size for Training: $SAMPLE_SIZE Source 1 Entities"
echo "=============================================================================="

# 1. Ensure required directories exist
mkdir -p models output

# 2. Install/verify Python dependencies
echo ""
echo ">>> [1/5] Checking and Installing Dependencies..."
pip install -r requirements.txt --quiet

# 3. Model Training
echo ""
echo ">>> [2/5] Training LightGBM Matching Model..."
export PYTHONPATH=.
python -m src.modeling.train \
    --s1-path data/processed/train_source1.parquet \
    --s2-path data/processed/train_source2.parquet \
    --s3-path data/processed/train_source3.parquet \
    --gt-path data/processed/train_ground_truth.parquet \
    --sample-size "$SAMPLE_SIZE" \
    --output-model "$MODEL_PATH"

# 4. Test Inference
echo ""
echo ">>> [3/5] Running Country-Partitioned Inference on Test Set..."
python -m src.modeling.predict \
    --model-path "$MODEL_PATH" \
    --test-s1 "$TEST_S1" \
    --test-s2 data/processed/test_source2.parquet \
    --test-s3 data/processed/test_source3.parquet \
    --output-dir "$OUTPUT_DIR"

# 5. Validate Submission TSVs
echo ""
echo ">>> [4/5] Validating Output TSVs..."
python scripts/validate_submission.py \
    --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" \
    --test-s1 "$TEST_S1"

# 6. Compress Output into submission.zip
echo ""
echo ">>> [5/5] Packaging Submission Files into submission.zip..."
python -c "
import zipfile, os
zip_path = '$OUTPUT_DIR/submission.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write('$OUTPUT_DIR/matching_results.tsv', 'matching_results.tsv')
    z.write('$OUTPUT_DIR/candidate_pairs.tsv', 'candidate_pairs.tsv')
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f'Created {zip_path} successfully ({size_mb:.2f} MB).')
"

echo ""
echo "=============================================================================="
echo " Pipeline Complete!"
echo " Outputs available in $OUTPUT_DIR/:"
ls -lh "$OUTPUT_DIR"
echo "=============================================================================="
echo " You can now download $OUTPUT_DIR/submission.zip and submit!"
echo " Don't forget to STOP your AWS SageMaker/EC2 instance to avoid extra charges."
echo "=============================================================================="
