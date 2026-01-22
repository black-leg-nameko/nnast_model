# GPUクラウドでのGNN学習ガイド

## GPUクラウドの利点

### 学習速度の比較

**CPU（ローカル）**:
- エポックあたり: 10-30分（データセットサイズによる）
- 20エポック: **3-10時間**

**GPU（クラウド）**:
- エポックあたり: 1-5分（GPU性能による）
- 20エポック: **20分-1.5時間**

**速度向上**: **5-10倍**の高速化が期待できる

## 推奨クラウドサービス

### 1. Google Colab（推奨）

**無料版**:
- GPU: T4（16GB VRAM）
- 制限: セッション12時間、連続使用制限あり
- コスト: **無料**

**有料版（Colab Pro）**:
- GPU: T4/V100/A100（より高性能）
- 制限: より長いセッション
- コスト: **月額$10**

**使用方法**:
```python
# Colabで実行
!git clone https://github.com/your-repo/nnast_model
!cd nnast_model && pip install -r requirements.txt

# データセットをアップロード（Google Drive経由）
from google.colab import drive
drive.mount('/content/drive')

# 学習実行
!cd nnast_model && python -m ml.train \
    --graphs /content/drive/MyDrive/training_data/train_graphs.jsonl \
    --labels /content/drive/MyDrive/training_data/train_labels.jsonl \
    --output-dir /content/drive/MyDrive/checkpoints \
    --epochs 20 \
    --batch-size 32
```

### 2. AWS EC2（柔軟性高い）

**推奨インスタンス**:
- **g4dn.xlarge**: T4 GPU（約**$0.50/時間**）
- **p3.2xlarge**: V100 GPU（約**$3.00/時間**）

**セットアップ**:
```bash
# EC2インスタンス起動後
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# データセットをアップロード
scp -r training_data/ ec2-user@your-instance:/home/ec2-user/nnast_model/

# 学習実行
python3 -m ml.train \
    --graphs training_data/train_graphs.jsonl \
    --labels training_data/train_labels.jsonl \
    --output-dir checkpoints \
    --epochs 20 \
    --batch-size 32
```

### 3. Google Cloud Platform（GCP）

**推奨インスタンス**:
- **n1-standard-4 + NVIDIA T4**: 約**$0.35/時間**

### 4. Azure

**推奨インスタンス**:
- **NC6s_v3**: V100 GPU（約**$3.00/時間**）

## ローカル vs クラウドの比較

| 項目 | ローカル（CPU） | クラウド（GPU） |
|------|----------------|----------------|
| 学習時間（20エポック） | **3-10時間** | **20分-1.5時間** |
| コスト | 無料 | $0.50-3.00/時間 |
| セットアップ | 簡単 | 中程度 |
| データ転送 | 不要 | 必要 |

## 推奨アプローチ（3日計画）

### Day 2（学習）の戦略

1. **まずローカルで小規模テスト**（1-2エポック、30分-1時間）
   - コードが正しく動作するか確認
   - データローダーが正常に動作するか確認
   - メモリ使用量を確認

2. **クラウドで本格学習**（20エポック、20分-1.5時間）
   - **Google Colab Pro推奨**（$10/月、簡単、高速）
   - または**AWS EC2 g4dn.xlarge**（$0.50/時間、柔軟性高い）
   - データセットをアップロード
   - 学習実行

**理由**:
- ローカルでテストしてからクラウドで実行することで、時間を節約
- クラウドのコストを最小限に抑える

## データセットサイズとGPUメモリ

### 現在のデータセット（450サンプル）

**メモリ使用量の目安**:
- **CPU**: 2-4GB RAM
- **GPU**: 1-2GB VRAM（T4で十分）

**バッチサイズ**:
- CPU: 4-8
- GPU: 32-64（可能）

### より大きなデータセット（1,000+サンプル）

**GPU推奨**:
- T4: バッチサイズ32-64
- V100: バッチサイズ64-128
- A100: バッチサイズ128-256

## コスト見積もり

### Google Colab Pro
- **月額**: $10
- **学習時間**: 20エポックで約1時間
- **総コスト**: $10/月（他のプロジェクトでも使用可能）

### AWS EC2（g4dn.xlarge）
- **時間単価**: $0.50
- **学習時間**: 20エポックで約1時間
- **総コスト**: **$0.50**（1回の学習）

### AWS EC2（p3.2xlarge）
- **時間単価**: $3.00
- **学習時間**: 20エポックで約30分
- **総コスト**: **$1.50**（1回の学習）

## 結論

**3日計画の場合**:
- ✅ **Google Colab Proを推奨**（$10/月、簡単、高速）
- ✅ または**AWS EC2 g4dn.xlarge**（$0.50/時間、柔軟性高い）

**学習時間**:
- CPU: 3-10時間
- GPU: **20分-1.5時間**（5-10倍高速）

**推奨**: まずローカルで動作確認してから、クラウドで本格学習を実行

