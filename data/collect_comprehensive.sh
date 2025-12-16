#!/bin/bash
# Comprehensive vulnerability collection script
# Collects diverse vulnerability types to improve dataset diversity

OUTPUT_DIR="./dataset"
LIMIT_PER_TYPE=20

echo "=========================================="
echo "Comprehensive Vulnerability Collection"
echo "=========================================="
echo ""

# Collect multiple vulnerability types
TYPES=(
    "sql_injection"
    "xss"
    "command_injection"
    "path_traversal"
    "deserialization"
    "authentication"
    "authorization"
    "crypto"
)

TOTAL_COLLECTED=0

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
        TOTAL_COLLECTED=$COUNT
    fi
    
    echo ""
    echo "Waiting 20 seconds before next type..."
    sleep 20
done

echo "=========================================="
echo "Collection Complete!"
echo "=========================================="
echo "Total records: $TOTAL_COLLECTED"
echo ""
echo "Run quality check:"
echo "  python3 -m data.check_quality --dataset-dir $OUTPUT_DIR"

