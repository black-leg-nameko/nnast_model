# 評価指標モジュール

## 概要

設計書v2のセクション7で要求されている評価指標を実装したモジュールです。

## 実装された評価指標

### 1. 分類指標

#### Macro-F1
クラスごとのF1スコアの平均値。不均衡データセットでの性能評価に有効。

#### Per-class F1
各クラス（safe/vulnerable）ごとのF1スコア。

#### PR-AUC (Precision-Recall AUC)
Precision-Recall曲線の下の面積。不均衡データセットでの性能評価に適している。

#### その他の基本指標
- Accuracy
- Precision (Binary, Macro)
- Recall (Binary, Macro)
- Confusion Matrix

### 2. ローカライズ精度（行レベル）

脆弱性の検出位置（行番号）の精度を評価します。

```python
from ml.evaluation import calculate_localization_accuracy

metrics = calculate_localization_accuracy(
    predicted_spans=[
        {"file": "app.py", "lines": [42, 43, 44]},
        {"file": "app.py", "lines": [100, 101]}
    ],
    ground_truth_spans=[
        {"file": "app.py", "lines": [40, 41, 42, 43]},
        {"file": "app.py", "lines": [99, 100]}
    ],
    tolerance=5  # 5行以内の誤差を許容
)
```

### 3. フレームワーク別性能

Flask、Django、FastAPIなど、フレームワークごとの性能を評価します。

```python
from ml.evaluation import calculate_framework_metrics

framework_metrics = calculate_framework_metrics(
    results=[
        {"framework": "flask", "is_vulnerable": True, "predicted_vulnerable": True, "confidence": 0.9},
        {"framework": "django", "is_vulnerable": False, "predicted_vulnerable": False, "confidence": 0.1},
        ...
    ]
)
```

## 使用方法

### train.pyでの使用

評価モジュールは`train.py`に自動統合されています。学習時に以下の指標が計算されます：

```bash
python -m ml.train \
    --train-graphs training_data/train_graphs.jsonl \
    --train-labels training_data/train_labels.jsonl \
    --val-graphs training_data/val_graphs.jsonl \
    --val-labels training_data/val_labels.jsonl \
    --output-dir checkpoints
```

出力例：
```
Epoch 1/10: Train Loss: 0.5234, Train Acc: 75.23%, Val Loss: 0.4567, Val Acc: 78.45%, LR: 1.00e-03
  Val F1: 0.7234, Precision: 0.7123, Recall: 0.7345
  Val Macro-F1: 0.7234, PR-AUC: 0.8123
  Per-class F1: {'safe': 0.7123, 'vulnerable': 0.7345}
```

### プログラムでの使用

```python
from ml.evaluation import (
    calculate_classification_metrics,
    format_metrics_report
)
import numpy as np

y_true = np.array([0, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 1])
y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.85])

metrics = calculate_classification_metrics(
    y_true, y_pred, y_proba,
    class_names=['safe', 'vulnerable']
)

print(format_metrics_report(metrics))
```

## 出力される指標

### 基本指標
- `accuracy`: 全体の精度
- `f1_binary`: バイナリ分類のF1スコア
- `precision_binary`: バイナリ分類のPrecision
- `recall_binary`: バイナリ分類のRecall

### Macro指標
- `f1_macro`: Macro-F1スコア
- `precision_macro`: Macro-Precision
- `recall_macro`: Macro-Recall

### Per-class指標
- `f1_per_class`: クラスごとのF1スコア（辞書形式）

### PR-AUC
- `pr_auc`: Precision-Recall AUC

### Confusion Matrix
- `confusion_matrix`: 混同行列
- `tn`, `fp`, `fn`, `tp`: True Negative, False Positive, False Negative, True Positive

## 設計書との対応

| 設計書の指標 | 実装名 | 説明 |
|------------|--------|------|
| Macro-F1 | `f1_macro` | クラスごとのF1スコアの平均 |
| Per-class F1 | `f1_per_class` | 各クラスごとのF1スコア |
| PR-AUC | `pr_auc` | Precision-Recall曲線の下の面積 |
| ローカライズ精度 | `localization_accuracy` | 行レベルの位置精度 |
| フレームワーク別性能 | `calculate_framework_metrics` | フレームワークごとの性能 |

## 注意事項

- PR-AUCの計算には予測確率が必要です
- ローカライズ精度の計算には、予測された行番号と正解の行番号が必要です
- フレームワーク別性能の計算には、結果にフレームワーク情報が含まれている必要があります

