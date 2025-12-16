#!/bin/bash
# Collect OWASP Top 10 2021 vulnerabilities

OUTPUT_DIR="./dataset"
LIMIT_PER_TYPE=25  # Target 25 records per vulnerability type

echo "=========================================="
echo "OWASP Top 10 2021 Vulnerability Collection"
echo "=========================================="
echo ""

# OWASP Top 10 2021 mapping
# A01: Broken Access Control -> authorization, path_traversal
# A02: Cryptographic Failures -> crypto
# A03: Injection -> sql_injection, xss, command_injection, code_injection
# A04: Insecure Design -> information_disclosure
# A05: Security Misconfiguration -> misconfiguration
# A06: Vulnerable Components -> vulnerable_dependency
# A07: Auth Failures -> authentication
# A08: Data Integrity -> deserialization
# A09: Logging Failures -> information_disclosure
# A10: SSRF -> ssrf

OWASP_TYPES=(
    # A03: Injection (highest priority)
    "sql_injection"
    "xss"
    "command_injection"
    "code_injection"
    
    # A01: Broken Access Control
    "authorization"
    "path_traversal"
    
    # A10: SSRF
    "ssrf"
    
    # A08: Data Integrity
    "deserialization"
    
    # A02: Cryptographic Failures
    "crypto"
    
    # A07: Authentication Failures
    "authentication"
    
    # A04/A09: Information Disclosure
    "information_disclosure"
    
    # A06: Vulnerable Components
    "vulnerable_dependency"
    
    # A05: Misconfiguration
    "misconfiguration"
)

TOTAL_COLLECTED=0

for vtype in "${OWASP_TYPES[@]}"; do
    echo "=========================================="
    echo "Collecting: $vtype (OWASP Top 10)"
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
echo "OWASP Top 10 Collection Complete!"
echo "=========================================="
echo "Total records: $TOTAL_COLLECTED"
echo ""
echo "Run quality check:"
echo "  python3 -m data.check_quality --dataset-dir $OUTPUT_DIR"
echo ""
echo "Check OWASP Top 10 coverage:"
echo "  python3 -m data.check_quality --dataset-dir $OUTPUT_DIR | grep -A 20 'Vulnerability Types'"
