#!/bin/bash
# 学習実行と自動停止スクリプト
# EC2インスタンス上で実行

set -e

echo "=========================================="
echo "NNAST Training with Auto-Stop"
echo "=========================================="
echo ""

# 学習パラメータ
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"

echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Train ratio: $TRAIN_RATIO"
echo ""

# GPU確認
echo "Checking GPU..."
nvidia-smi || echo "Warning: GPU not available"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 学習実行
echo ""
echo "Starting training..."
python3 -m ml.train \
    --graphs training_data/train_graphs.jsonl \
    --labels training_data/train_labels.jsonl \
    --output-dir checkpoints \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --train-ratio "$TRAIN_RATIO" \
    --device cuda \
    2>&1 | tee training.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

# インスタンスIDの取得
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)

if [ -n "$INSTANCE_ID" ]; then
    echo "Instance ID: $INSTANCE_ID"
    echo ""
    echo "⚠️ Stopping instance to avoid charges..."
    echo "   (This will stop the instance automatically)"
    echo ""
    read -p "Do you want to stop the instance now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
        echo "Instance stop command sent."
    else
        echo "Instance will continue running. Please stop it manually:"
        echo "  AWS Console → EC2 → Instances → Stop"
        echo "  Or: aws ec2 stop-instances --instance-ids $INSTANCE_ID"
    fi
else
    echo "Warning: Could not get instance ID"
    echo "Please stop the instance manually from AWS Console"
fi

echo ""
echo "To download results, run:"
echo "  ./aws_setup_scripts/download_results.sh"

