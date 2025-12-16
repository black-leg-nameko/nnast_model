#!/bin/bash
# データセットをGoogle Colab用に準備するスクリプト

echo "=========================================="
echo "Google Colab用データセット準備"
echo "=========================================="

# 出力ディレクトリ
OUTPUT_DIR="./colab_upload"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "1. 学習データのコピー..."
cp -r training_data "$OUTPUT_DIR/"

echo "2. データセットメタデータのコピー..."
mkdir -p "$OUTPUT_DIR/dataset"
if [ -f "dataset/metadata.jsonl" ]; then
    cp dataset/metadata.jsonl "$OUTPUT_DIR/dataset/"
fi

echo "3. 必要なコードファイルのコピー..."
mkdir -p "$OUTPUT_DIR/ml"
cp -r ml/*.py "$OUTPUT_DIR/ml/" 2>/dev/null || true
cp -r ml/*.sh "$OUTPUT_DIR/ml/" 2>/dev/null || true

mkdir -p "$OUTPUT_DIR/data"
cp -r data/*.py "$OUTPUT_DIR/data/" 2>/dev/null || true

mkdir -p "$OUTPUT_DIR/ir"
cp -r ir/*.py "$OUTPUT_DIR/ir/" 2>/dev/null || true

mkdir -p "$OUTPUT_DIR/cpg"
cp -r cpg/*.py "$OUTPUT_DIR/cpg/" 2>/dev/null || true

# cli.pyもコピー
if [ -f "cli.py" ]; then
    cp cli.py "$OUTPUT_DIR/"
fi

echo "4. zipファイルの作成..."
cd "$OUTPUT_DIR"
zip -r ../nnast_colab_dataset.zip . -x "*.pyc" "__pycache__/*" "*.log"
cd ..

echo ""
echo "=========================================="
echo "準備完了！"
echo "=========================================="
echo ""
echo "作成されたファイル:"
echo "  - $OUTPUT_DIR/ (ディレクトリ)"
echo "  - nnast_colab_dataset.zip (アップロード用)"
echo ""
echo "次のステップ:"
echo "  1. nnast_colab_dataset.zip をGoogle Driveにアップロード"
echo "  2. Google Colabで ml/train_colab.ipynb を開く"
echo "  3. ノートブックの指示に従って実行"
echo ""

