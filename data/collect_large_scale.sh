#!/bin/bash
# 大規模データ収集スクリプト
# 論文レベルのデータセット構築のための包括的な収集

set -e

OUTPUT_BASE_DIR="./dataset_large_scale"
LOG_FILE="./collection_log_$(date +%Y%m%d_%H%M%S).txt"

# 出力ディレクトリの作成
mkdir -p "$OUTPUT_BASE_DIR"

# ログファイルの初期化
echo "=== Large Scale Dataset Collection ===" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 関数: 収集の実行とログ記録
collect_with_log() {
    local query=$1
    local output_dir=$2
    local limit=$3
    local description=$4
    
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Collecting: $description" | tee -a "$LOG_FILE"
    echo "Query: $query" | tee -a "$LOG_FILE"
    echo "Limit: $limit" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    mkdir -p "$output_dir"
    
    python3 -m data.collect_dataset \
        --github-query "$query" \
        --limit "$limit" \
        --output-dir "$output_dir" 2>&1 | tee -a "$LOG_FILE"
    
    # 収集結果の確認
    if [ -f "$output_dir/metadata.jsonl" ]; then
        COUNT=$(wc -l < "$output_dir/metadata.jsonl" | tr -d ' ')
        echo "Collected: $COUNT samples" | tee -a "$LOG_FILE"
    else
        echo "Warning: No metadata.jsonl found" | tee -a "$LOG_FILE"
    fi
    
    echo "Completed: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # GitHub APIレート制限を考慮して待機
    echo "Waiting 30 seconds before next collection..." | tee -a "$LOG_FILE"
    sleep 30
}

# Phase 1: CVE関連コミットの収集
echo "=== Phase 1: CVE-related commits ===" | tee -a "$LOG_FILE"

for year in 2020 2021 2022 2023 2024; do
    collect_with_log \
        "CVE-$year python" \
        "$OUTPUT_BASE_DIR/cve_$year" \
        200 \
        "CVE-$year Python vulnerabilities"
done

# Phase 2: セキュリティ修正コミットの収集
echo "=== Phase 2: Security fix commits ===" | tee -a "$LOG_FILE"

SECURITY_QUERIES=(
    "security fix python"
    "vulnerability fix python"
    "fix CVE python"
    "security patch python"
    "vulnerability patch python"
    "CVE fix python"
    "security update python"
    "vulnerability update python"
)

for query in "${SECURITY_QUERIES[@]}"; do
    # クエリをファイル名に変換（スペースをアンダースコアに）
    safe_query=$(echo "$query" | tr ' ' '_')
    collect_with_log \
        "$query" \
        "$OUTPUT_BASE_DIR/security_fixes_$safe_query" \
        100 \
        "Security fixes: $query"
done

# Phase 3: 特定脆弱性タイプの収集
echo "=== Phase 3: Specific vulnerability types ===" | tee -a "$LOG_FILE"

VULN_TYPES=(
    "sql_injection"
    "xss"
    "command_injection"
    "path_traversal"
    "deserialization"
    "ssrf"
    "authentication"
    "authorization"
    "crypto"
)

for vtype in "${VULN_TYPES[@]}"; do
    python3 -m data.collect_diverse_vulnerabilities \
        --vuln-types "$vtype" \
        --limit-per-type 100 \
        --output-dir "$OUTPUT_BASE_DIR/vuln_$vtype" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$OUTPUT_BASE_DIR/vuln_$vtype/metadata.jsonl" ]; then
        COUNT=$(wc -l < "$OUTPUT_BASE_DIR/vuln_$vtype/metadata.jsonl" | tr -d ' ')
        echo "Collected $vtype: $COUNT samples" | tee -a "$LOG_FILE"
    fi
    
    echo "Waiting 30 seconds..." | tee -a "$LOG_FILE"
    sleep 30
done

# Phase 4: セキュリティプロジェクトの収集
echo "=== Phase 4: Security projects ===" | tee -a "$LOG_FILE"

SECURITY_PROJECTS=(
    "OWASP python"
    "security scanner test python"
    "CTF python web"
    "vulnerable web app python"
    "security testing python"
)

for query in "${SECURITY_PROJECTS[@]}"; do
    safe_query=$(echo "$query" | tr ' ' '_')
    collect_with_log \
        "$query" \
        "$OUTPUT_BASE_DIR/projects_$safe_query" \
        200 \
        "Security projects: $query"
done

# 最終統計
echo "==========================================" | tee -a "$LOG_FILE"
echo "Collection Complete!" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 全データセットの統合
echo "Merging all collected datasets..." | tee -a "$LOG_FILE"
python3 -m data.merge_datasets \
    --input-dirs "$OUTPUT_BASE_DIR"/* \
    --output-dir "$OUTPUT_BASE_DIR/merged" \
    2>&1 | tee -a "$LOG_FILE"

# 最終統計
if [ -f "$OUTPUT_BASE_DIR/merged/metadata.jsonl" ]; then
    TOTAL=$(wc -l < "$OUTPUT_BASE_DIR/merged/metadata.jsonl" | tr -d ' ')
    echo "Total samples collected: $TOTAL" | tee -a "$LOG_FILE"
else
    echo "Warning: Merged dataset not found" | tee -a "$LOG_FILE"
fi

echo ""
echo "Collection log saved to: $LOG_FILE"
echo "Merged dataset: $OUTPUT_BASE_DIR/merged"

