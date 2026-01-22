#!/bin/bash
# データセットをEC2にアップロードするスクリプト
# ローカルマシンから実行

set -e

# 設定（変更が必要）
KEY_FILE="${AWS_KEY_FILE:-nnast-key.pem}"
EC2_IP="${EC2_IP:-}"
EC2_USER="${EC2_USER:-ubuntu}"
PROJECT_DIR="${PROJECT_DIR:-nnast_model}"

if [ -z "$EC2_IP" ]; then
    echo "Error: EC2_IP environment variable is not set"
    echo "Usage: EC2_IP=54.123.45.67 ./aws_setup_scripts/upload_data.sh"
    exit 1
fi

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    echo "Please set AWS_KEY_FILE environment variable or place key file in current directory"
    exit 1
fi

echo "=========================================="
echo "Uploading Data to EC2"
echo "=========================================="
echo "EC2 IP: $EC2_IP"
echo "Key file: $KEY_FILE"
echo ""

# キーファイルの権限を設定
chmod 400 "$KEY_FILE"

# プロジェクトディレクトリの確認
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

# training_dataのアップロード
echo "1. Uploading training_data..."
scp -i "$KEY_FILE" -r "$PROJECT_DIR/training_data/" "$EC2_USER@$EC2_IP:~/nnast_model/" || {
    echo "  Creating directory on EC2..."
    ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" "mkdir -p ~/nnast_model"
    scp -i "$KEY_FILE" -r "$PROJECT_DIR/training_data/" "$EC2_USER@$EC2_IP:~/nnast_model/"
}

# プロジェクトファイルのアップロード（必要なファイルのみ）
echo ""
echo "2. Uploading project files..."
scp -i "$KEY_FILE" "$PROJECT_DIR/ml/" "$EC2_USER@$EC2_IP:~/nnast_model/" -r || true
scp -i "$KEY_FILE" "$PROJECT_DIR/cpg/" "$EC2_USER@$EC2_IP:~/nnast_model/" -r || true
scp -i "$KEY_FILE" "$PROJECT_DIR/ir/" "$EC2_USER@$EC2_IP:~/nnast_model/" -r || true
scp -i "$KEY_FILE" "$PROJECT_DIR/patterns.yaml" "$EC2_USER@$EC2_IP:~/nnast_model/" || true
scp -i "$KEY_FILE" "$PROJECT_DIR/requirements.txt" "$EC2_USER@$EC2_IP:~/nnast_model/" || true

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "=========================================="
echo ""
echo "Next: SSH to EC2 and run training:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_IP"
echo "  cd ~/nnast_model"
echo "  python3 -m ml.train --graphs training_data/train_graphs.jsonl --labels training_data/train_labels.jsonl --output-dir checkpoints --epochs 20 --batch-size 32 --device cuda"

