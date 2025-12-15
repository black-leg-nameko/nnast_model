#!/bin/bash
# Check progress of dataset collection

DATASET_DIR="./dataset"

if [ ! -f "$DATASET_DIR/metadata.jsonl" ]; then
    echo "Collection not started or metadata file not found"
    exit 1
fi

echo "=== Dataset Collection Progress ==="
echo ""

# Count total records
TOTAL=$(wc -l < "$DATASET_DIR/metadata.jsonl" | tr -d ' ')
echo "Total records collected: $TOTAL"
echo ""

# Show CVE distribution
echo "=== CVE Distribution ==="
grep -o '"cve_id":"[^"]*"' "$DATASET_DIR/metadata.jsonl" | \
    sed 's/"cve_id":"//;s/"//' | \
    sort | uniq -c | sort -rn | head -10

echo ""
echo "=== Vulnerability Types ==="
grep -o '"vulnerability_type":"[^"]*"' "$DATASET_DIR/metadata.jsonl" | \
    sed 's/"vulnerability_type":"//;s/"//' | \
    sort | uniq -c | sort -rn | head -10

echo ""
echo "=== Recent Records ==="
tail -5 "$DATASET_DIR/metadata.jsonl" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    try:
        r = json.loads(line)
        print(f\"{i}. CVE: {r.get('cve_id', 'N/A')}, Type: {r.get('vulnerability_type', 'N/A')}, File: {r.get('file_path', 'N/A').split('/')[-1]}\")
    except:
        pass
"

echo ""
echo "=== CPG Graphs ==="
if [ -d "$DATASET_DIR/processed" ]; then
    GRAPH_COUNT=$(find "$DATASET_DIR/processed" -name "*.jsonl" | wc -l | tr -d ' ')
    echo "Generated CPG graphs: $GRAPH_COUNT"
else
    echo "No CPG graphs generated yet"
fi

