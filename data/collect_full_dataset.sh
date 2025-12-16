#!/bin/bash
# Full dataset collection script
# This script collects Python vulnerabilities from multiple queries

set -e  # Exit on error

OUTPUT_DIR="./dataset"
LIMIT_PER_QUERY=50  # Limit per query to avoid rate limits

echo "=========================================="
echo "NNAST Dataset Collection - Full Run"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Limit per query: $LIMIT_PER_QUERY"
echo ""

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN is not set"
    echo "Please set it using: export GITHUB_TOKEN=your_token"
    echo "Or create .env file: echo 'GITHUB_TOKEN=your_token' > .env"
    exit 1
fi

echo "✓ GITHUB_TOKEN is set"
echo ""

# Array of queries to search
QUERIES=(
    "CVE-2023"
    "CVE-2024"
    "fix CVE"
    "security CVE"
    "vulnerability fix"
)

# Collect data for each query
TOTAL_COLLECTED=0

for query in "${QUERIES[@]}"; do
    echo "=========================================="
    echo "Collecting: $query"
    echo "=========================================="
    
    python3 -m data.collect_dataset \
        --github-query "$query" \
        --limit $LIMIT_PER_QUERY \
        --output-dir "$OUTPUT_DIR" || {
        echo "Warning: Collection for '$query' failed, continuing..."
    }
    
    # Count collected records
    if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
        COUNT=$(wc -l < "$OUTPUT_DIR/metadata.jsonl" | tr -d ' ')
        echo "Total records so far: $COUNT"
    fi
    
    echo ""
    echo "Waiting 10 seconds before next query..."
    sleep 10
done

# Final summary
if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
    TOTAL=$(wc -l < "$OUTPUT_DIR/metadata.jsonl" | tr -d ' ')
    echo "=========================================="
    echo "Collection Complete!"
    echo "=========================================="
    echo "Total records collected: $TOTAL"
    echo "Dataset location: $OUTPUT_DIR"
    echo ""
    echo "Next step: Prepare training data"
    echo "  python -m data.prepare_training_data --dataset-dir $OUTPUT_DIR --output-dir ./training_data"
else
    echo "Warning: No metadata file found"
fi

