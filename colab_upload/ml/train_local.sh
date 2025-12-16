#!/bin/bash
# ローカル（CPU）での学習実行スクリプト

# データセットパス（学習データ準備が必要）
GRAPHS_FILE="${GRAPHS_FILE:-./training_data/train_graphs.jsonl}"
LABELS_FILE="${LABELS_FILE:-./training_data/train_labels.jsonl}"

# CPUモード用の最適化設定
BATCH_SIZE=4          # メモリ節約のため小さめ
EPOCHS=5              # 最初は少なめ（動作確認後、増やす）
MAX_NODES=500         # 処理速度向上
HIDDEN_DIM=128        # メモリ節約（デフォルト256から削減）

# 出力ディレクトリ
OUTPUT_DIR="./checkpoints_cpu"

echo "=========================================="
echo "ローカル（CPU）学習実行"
echo "=========================================="
echo "グラフファイル: $GRAPHS_FILE"
echo "ラベルファイル: $LABELS_FILE"
echo "バッチサイズ: $BATCH_SIZE"
echo "エポック数: $EPOCHS"
echo "最大ノード数: $MAX_NODES"
echo ""

# ファイル存在確認
if [ ! -f "$GRAPHS_FILE" ]; then
    echo "エラー: グラフファイルが見つかりません: $GRAPHS_FILE"
    echo ""
    echo "学習データを準備してください:"
    echo "  python3 -m data.prepare_training_data --dataset-dir ./dataset --output-dir ./training_data"
    exit 1
fi

if [ ! -f "$LABELS_FILE" ]; then
    echo "警告: ラベルファイルが見つかりません: $LABELS_FILE"
    echo "ラベルなしで実行します（taint recordsを使用）"
    LABELS_ARG=""
else
    LABELS_ARG="--labels $LABELS_FILE"
fi

# 学習実行
python3 -m ml.train \
    --graphs "$GRAPHS_FILE" \
    $LABELS_ARG \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --hidden-dim $HIDDEN_DIM \
    --max-nodes $MAX_NODES \
    --device cpu \
    --gnn-type GAT \
    --lr 1e-4

echo ""
echo "学習完了！"
echo "チェックポイント: $OUTPUT_DIR"

