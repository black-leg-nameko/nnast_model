#!/bin/bash
# AWS EC2 g4dn.xlarge セットアップスクリプト
# EC2インスタンス上で実行

set -e  # エラー時に停止

echo "=========================================="
echo "AWS EC2 Setup Script for NNAST Training"
echo "=========================================="
echo ""

# 1. システムアップデート
echo "1. Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Python環境の確認
echo ""
echo "2. Checking Python environment..."
python3 --version

# 3. PyTorchのインストール（CUDA 11.8）
echo ""
echo "3. Installing PyTorch with CUDA 11.8..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. GPU確認
echo ""
echo "4. Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi not found. GPU may not be available."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 5. 依存関係のインストール
echo ""
echo "5. Installing dependencies..."
pip3 install torch-geometric transformers tqdm numpy pyyaml

# 6. プロジェクトディレクトリの確認
echo ""
echo "6. Checking project directory..."
if [ -d "nnast_model" ]; then
    echo "  Project directory found: nnast_model"
    cd nnast_model
else
    echo "  Warning: Project directory not found"
    echo "  Please upload the project using scp or git clone"
fi

# 7. データセットの確認
echo ""
echo "7. Checking dataset..."
if [ -d "training_data" ]; then
    echo "  Dataset found: training_data"
    echo "  Files:"
    ls -lh training_data/*.jsonl 2>/dev/null | head -5 || echo "    No JSONL files found"
else
    echo "  Warning: Dataset directory not found"
    echo "  Please upload training_data using scp"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload training_data if not already done:"
echo "   scp -i your-key.pem -r training_data/ ubuntu@<IP>:~/nnast_model/"
echo ""
echo "2. Start training:"
echo "   python3 -m ml.train \\"
echo "       --graphs training_data/train_graphs.jsonl \\"
echo "       --labels training_data/train_labels.jsonl \\"
echo "       --output-dir checkpoints \\"
echo "       --epochs 20 \\"
echo "       --batch-size 32 \\"
echo "       --device cuda"
echo ""
echo "3. After training, STOP the instance to avoid charges!"

