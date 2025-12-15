#!/bin/bash
# Collect diverse vulnerability types

OUTPUT_DIR="./dataset"
LIMIT_PER_TYPE=20

echo "=========================================="
echo "Collecting Diverse Vulnerability Types"
echo "=========================================="
echo ""

# Vulnerability types to collect
TYPES=(
    "sql_injection"
    "xss"
    "command_injection"
    "path_traversal"
    "deserialization"
)

for vtype in "${TYPES[@]}"; do
    echo "=========================================="
    echo "Collecting: $vtype"
    echo "=========================================="
    
    python3 -m data.collect_specific_vulns \
        --type "$vtype" \
        --limit-per-query 5 \
        --total-limit $LIMIT_PER_TYPE \
        --output-dir "$OUTPUT_DIR" || {
        echo "Warning: Collection for '$vtype' had errors, continuing..."
    }
    
    # Check progress
    if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
        COUNT=$(wc -l < "$OUTPUT_DIR/metadata.jsonl" | tr -d ' ')
        echo "Total records: $COUNT"
    fi
    
    echo ""
    echo "Waiting 15 seconds before next type..."
    sleep 15
done

echo "=========================================="
echo "Collection Complete!"
echo "=========================================="
if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
    TOTAL=$(wc -l < "$OUTPUT_DIR/metadata.jsonl" | tr -d ' ')
    echo "Total records: $TOTAL"
    echo ""
    echo "Run quality check:"
    echo "  python3 -m data.check_quality --dataset-dir $OUTPUT_DIR"
fi

