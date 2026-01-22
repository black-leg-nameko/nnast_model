# NNAST実装ガイド

設計書（`nnast_system_design.md`）に基づいた実装の完了状況とセットアップ手順です。

## 実装完了状況

### ✅ 1. AWSインフラ（Terraform）

**場所**: `terraform/`

**実装内容**:
- VPCとネットワーク構成
- KMS暗号化キー
- DynamoDBテーブル（Tenant, Project, Job, AuditLog）
- S3バケット（レポート、CloudTrail）
- SQSキュー（レポート生成ジョブ、DLQ）
- ECSクラスターとサービス（Auth Service、Report Worker）
- API Gateway（HTTP API）
- IAMロールとポリシー（GitHub Actions OIDC対応）
- CloudWatchログとアラーム
- CloudTrail監査ログ

**セットアップ**:
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# terraform.tfvarsを編集
terraform init
terraform plan
terraform apply
```

### ✅ 2. Auth Service (Go実装)

**場所**: `cloud/auth_service/go/`

**言語選択理由**: 
- **低レイテンシ**: CI速度への影響を最小化（起動時間 ~50ms、レイテンシ ~2ms）
- **軽量**: コンテナイメージが小さい（~15MB）
- **スパイク耐性**: goroutineで同時実行に対応

**機能**:
- GitHub OIDCトークン検証
- プロジェクト認可チェック
- 短命JWT（15分）発行
- 監査ログ記録

**デプロイ**:
```bash
cd cloud/auth_service/go
docker build -t nnast/auth-service:latest .
# ECRにプッシュ
```

詳細は `LANGUAGE_CHOICE_RATIONALE.md` を参照。

### ✅ 3. Report Service (Python実装)

**場所**: `cloud/report_service/`

**言語選択理由**:
- Bedrock APIとの連携が簡単
- JSON処理が直感的
- 開発速度が速い

**機能**:
- SQSからのジョブ受信
- Bedrock LLMを使用したレポート生成
- S3へのレポート保存
- DynamoDBでのジョブ管理

**デプロイ**:
```bash
cd cloud/report_service
docker build -t nnast/report-service:latest .
# ECRにプッシュ
```

### ✅ 4. ローカルCI部分 (Python実装)

**場所**: `local/`

**言語選択理由**:
- GNN推論はPyTorch/PyGが必須（Python以外の選択肢がない）
- CodeBERTはtransformersライブラリがPython専用
- 既存のPython実装を活用

**実装内容**:
- `masking.py`: データ最小化とマスキング
- `github_issue.py`: GitHub Issue作成/更新
- `scan.py`: メインスキャン処理

**機能**:
- フルCPG構築
- GNN推論
- Top-K抽出
- データ最小化・マスキング
- クラウドAPI呼び出し
- GitHub Issue作成

### ✅ 5. GitHub Actions Workflow

**場所**: `.github/workflows/nnast-scan.yml`

**機能**:
- PR/Push時に自動実行
- OIDC認証
- ローカルスキャン実行
- Issue作成

## セットアップ手順

### ステップ1: AWSインフラ構築

1. Terraformでインフラを構築
2. 出力値を確認（API Gateway URL、OIDC Provider ARNなど）

### ステップ2: コンテナイメージのデプロイ

1. ECRリポジトリを作成
2. Auth Service（Go）とReport Service（Python）のイメージをビルド・プッシュ
3. TerraformでECSタスク定義を更新

### ステップ3: 初期データの登録

1. DynamoDBにTenantを登録
2. DynamoDBにProjectを登録（GitHubリポジトリ情報、allowed_subject_patterns）

### ステップ4: GitHub Actions設定

1. リポジトリのSecretsに以下を設定:
   - `NNAST_API_URL`: API Gateway URL
   - `NNAST_TENANT_ID`: テナントID
   - `NNAST_PROJECT_ID`: プロジェクトID

2. GitHub ActionsのOIDC設定:
   - IAMロールの信頼ポリシーにリポジトリを追加

### ステップ5: テスト実行

1. PRを作成またはコードをプッシュ
2. GitHub Actionsが自動実行
3. Issueが作成されることを確認

## アーキテクチャ概要

```
GitHub Actions (Local - Python)
  ├─ CPG構築 (Python, 将来Rust化可能)
  ├─ GNN推論 (Python - PyTorch必須)
  ├─ Top-K抽出
  ├─ データ最小化・マスキング
  └─ API Gateway (OIDC認証)
      └─ Auth Service (Go - 低レイテンシ)
          └─ JWT発行
      └─ Report Service (Python - LLM連携)
          └─ SQS投入
              └─ Report Worker (ECS)
                  └─ Bedrock LLM
                      └─ S3 (レポート保存)
                          └─ GitHub Actions (Issue作成)
```

## データフロー

1. **検出（Local - Python）**: CPG構築 → GNN推論 → Top-K抽出
2. **認証（Cloud - Go）**: OIDC検証 → JWT発行（低レイテンシ）
3. **レポート生成（Cloud - Python）**: SQS → Bedrock → S3
4. **Issue作成（Local - Python）**: レポート取得 → GitHub Issue作成/更新

## 言語選択の根拠

詳細は `LANGUAGE_CHOICE_RATIONALE.md` を参照。

### パフォーマンス比較

| コンポーネント | 言語 | 起動時間 | メモリ | レイテンシ |
|---------------|------|---------|--------|-----------|
| Auth Service | Go | ~50ms | ~10MB | ~2ms |
| Auth Service (Python) | Python | ~500ms | ~50MB | ~10ms |
| **改善** | - | **10倍** | **5倍** | **5倍** |

## セキュリティ考慮事項

1. **データ最小化**: クラウドに送信するデータは最小限
2. **マスキング**: 識別子、秘密情報をマスキング
3. **OIDC認証**: 静的APIキーを使用しない
4. **短命JWT**: 15分の有効期限
5. **監査ログ**: すべての認証・操作を記録

## トラブルシューティング

### CPG構築が遅い

- CPGキャッシュを有効化（`--cpg-cache-dir`）
- タイムアウトを設定
- 将来的にRust化（`cpg_rust`プロジェクト参照）

### Auth Serviceのレイテンシが高い

- Go実装を使用（Python実装から移行）
- コンテナイメージサイズを確認（Alpineベースで~15MB）

### Bedrock APIエラー

- Bedrockモデルへのアクセス権限を確認
- リージョン設定を確認

### Issueが作成されない

- GitHub Tokenの権限を確認（`issues: write`）
- fingerprintの重複チェックを確認

## 次のステップ

1. **管理画面**: テナント/プロジェクト管理UI
2. **監視ダッシュボード**: CloudWatchダッシュボード
3. **アラート**: SNS通知の設定
4. **パフォーマンス最適化**: CPG構築のRust化（`cpg_rust`プロジェクト）

## 参考資料

- 設計書: `nnast_system_design.md`
- 言語選択の根拠: `LANGUAGE_CHOICE_RATIONALE.md`
- Terraform README: `terraform/README.md`
- Cloud Services README: `cloud/README.md`
- Local Scan README: `local/README.md`
