#!/bin/bash
# Generate synthetic samples for OWASP Top 10 vulnerability types that are difficult to collect

OUTPUT_DIR="./dataset_synthetic"
PROVIDER="${LLM_PROVIDER:-openai}"  # openai or anthropic
MODEL="${LLM_MODEL:-gpt-4}"  # gpt-4, claude-3-opus, etc.

echo "=========================================="
echo "OWASP Top 10 Synthetic Data Generation"
echo "=========================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo ""

# Check API key
if [ "$PROVIDER" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Set it with: export OPENAI_API_KEY=your_key"
    exit 1
fi

if [ "$PROVIDER" = "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Set it with: export ANTHROPIC_API_KEY=your_key"
    exit 1
fi

# Generate for difficult-to-collect types
TYPES=(
    "XSS:15:CWE-79"
    "SSRF:20:CWE-918"
    "Deserialization:20:CWE-502"
    "Cryptographic Weakness:15:CWE-327"
    "Code Injection:15:CWE-94"
)

TOTAL_GENERATED=0

for type_info in "${TYPES[@]}"; do
    IFS=':' read -r vtype count cwe <<< "$type_info"
    
    echo "=========================================="
    echo "Generating: $vtype ($count samples)"
    echo "=========================================="
    
    python3 -m data.generate_synthetic \
        --type "$vtype" \
        --count "$count" \
        --output-dir "$OUTPUT_DIR" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --cwe-id "$cwe" || {
        echo "Warning: Generation for '$vtype' had errors, continuing..."
    }
    
    # Check progress
    if [ -f "$OUTPUT_DIR/metadata.jsonl" ]; then
        COUNT=$(wc -l < "$OUTPUT_DIR/metadata.jsonl" | tr -d ' ')
        echo "Total synthetic samples: $COUNT"
        TOTAL_GENERATED=$COUNT
    fi
    
    echo ""
    echo "Waiting 5 seconds before next type..."
    sleep 5
done

echo "=========================================="
echo "Synthetic Generation Complete!"
echo "=========================================="
echo "Total synthetic samples: $TOTAL_GENERATED"
echo ""
echo "Next step: Merge with real dataset"
echo "  python -m data.merge_datasets --real-dir ./dataset --synthetic-dir $OUTPUT_DIR --output-dir ./dataset_merged"

