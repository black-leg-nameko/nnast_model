#!/bin/bash
# 学習結果をEC2からダウンロードするスクリプト
# ローカルマシンから実行

set -e

# 設定（変更が必要）
KEY_FILE="${AWS_KEY_FILE:-nnast-key.pem}"
EC2_IP="${EC2_IP:-}"
EC2_USER="${EC2_USER:-ubuntu}"

if [ -z "$EC2_IP" ]; then
    echo "Error: EC2_IP environment variable is not set"
    echo "Usage: EC2_IP=54.123.45.67 ./aws_setup_scripts/download_results.sh"
    exit 1
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    exit 1
fi

echo "=========================================="
echo "Downloading Results from EC2"
echo "=========================================="
echo "EC2 IP: $EC2_IP"
echo ""

# キーファイルの権限を設定
chmod 400 "$KEY_FILE"

# チェックポイントのダウンロード
echo "1. Downloading checkpoints..."
mkdir -p checkpoints_ec2
scp -i "$KEY_FILE" -r "$EC2_USER@$EC2_IP:~/nnast_model/checkpoints/" ./checkpoints_ec2/ || {
    echo "  Warning: Checkpoints not found"
}

# ログのダウンロード
echo ""
echo "2. Downloading logs..."
scp -i "$KEY_FILE" "$EC2_USER@$EC2_IP:~/nnast_model/training.log" ./ 2>/dev/null || {
    echo "  Warning: Training log not found"
}

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "⚠️ IMPORTANT: Stop the EC2 instance to avoid charges!"
echo "  AWS Console → EC2 → Instances → Select instance → Stop"

