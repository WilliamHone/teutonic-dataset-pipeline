#!/bin/bash
# generate_dataset_pipeline.sh - Script to launch the shard download and dataset generation pipeline

set -e  # Exit on error

# Default configuration
NUM_SHARDS=1
NUM_SAMPLES=2
SEQ_LEN=2048
OUTPUT_FILE="datasets/dataset_pipeline_v001.jsonl"
SHARDS_JSON="shards.json"
DOWNLOAD_DIR="datasets/temp"
BASE_URL="https://s3.hippius.com/teutonic-sn3/"

echo "🚀 Starting dataset generation pipeline..."
echo "  Num shards   : $NUM_SHARDS"
echo "  Samples/shard: $NUM_SAMPLES"
echo "  Seq len      : $SEQ_LEN"
echo "  Output       : $OUTPUT_FILE"
echo "  Shards JSON  : $SHARDS_JSON"
echo "  Temp Dir     : $DOWNLOAD_DIR"
echo "  Base URL     : $BASE_URL"
echo ""

# Ensure the output directory exists before the python script tries to write to it
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run the python pipeline script
# (Make sure your python script from the previous step is named process_shards.py)
python gen_dataset_pipeline.py \
    --num_shards "$NUM_SHARDS" \
    --num_samples "$NUM_SAMPLES" \
    --seq_len "$SEQ_LEN" \
    --output "$OUTPUT_FILE" \
    --shards_json "$SHARDS_JSON" \
    --download_dir "$DOWNLOAD_DIR" \
    --base_url "$BASE_URL"

echo "✅ Done! Output: $OUTPUT_FILE"