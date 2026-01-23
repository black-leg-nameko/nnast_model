# NNAST Services

このディレクトリには、NNASTクラウドサービスの実装が含まれています。

## サービス構成

### Auth Service (`auth_service/`)

GitHub Actions OIDCトークンを検証し、短命JWTを発行するサービス。

**機能:**
- GitHub OIDCトークンの検証
- プロジェクト情報の取得と認可チェック
- 短命JWT（15分）の発行
- 監査ログの記録

**エンドポイント:**
- `POST /auth/oidc`: OIDC認証とJWT発行
- `GET /health`: ヘルスチェック

### Report Service (`report_service/`)

SQSからレポート生成ジョブを受け取り、Bedrock LLMでMarkdownレポートを生成するサービス。

**機能:**
- SQSからのジョブ受信
- Bedrock LLMを使用したレポート生成
- S3へのレポート保存
- DynamoDBでのジョブステータス管理

**エンドポイント:**
- `POST /report`: レポート生成ジョブの作成（SQS投入）
- `GET /report/<job_id>`: レポート生成結果の取得
- `GET /health`: ヘルスチェック

**ワーカー:**
- SQSからメッセージを取得してレポート生成を実行

## デプロイ

### 1. Dockerイメージのビルド

```bash
# Auth Service
cd auth_service
docker build -t nnast/auth-service:latest .

# Report Service
cd ../report_service
docker build -t nnast/report-service:latest .
```

### 2. ECRへのプッシュ

```bash
# ECRリポジトリを作成（初回のみ）
aws ecr create-repository --repository-name nnast/auth-service
aws ecr create-repository --repository-name nnast/report-service

# ログイン
aws ecr get-login-password --region ap-northeast-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com

# タグ付けとプッシュ
docker tag nnast/auth-service:latest <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/nnast/auth-service:latest
docker push <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/nnast/auth-service:latest

docker tag nnast/report-service:latest <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/nnast/report-service:latest
docker push <account-id>.dkr.ecr.ap-northeast-1.amazonaws.com/nnast/report-service:latest
```

### 3. TerraformでECSタスク定義を更新

`infra/terraform/modules/ecs/main.tf`の`auth_service_image`と`report_service_image`をECRのURIに更新し、`terraform apply`を実行。

## 環境変数

### Auth Service

- `ENVIRONMENT`: 環境名（dev/staging/prod）
- `LOG_LEVEL`: ログレベル（INFO/DEBUG）
- `JWT_SECRET`: JWT署名用シークレット（Secrets Managerから取得）

### Report Service

- `ENVIRONMENT`: 環境名
- `LOG_LEVEL`: ログレベル
- `BEDROCK_MODEL_ID`: BedrockモデルID
- `REPORT_QUEUE_URL`: SQSレポートキューURL
- `AWS_REGION`: AWSリージョン

## 開発

### ローカル実行

```bash
# Auth Service
cd auth_service
pip install -r requirements.txt
export JWT_SECRET="your-secret-key"
export ENVIRONMENT="dev"
python main.py

# Report Service
cd report_service
pip install -r requirements.txt
export BEDROCK_MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"
export REPORT_QUEUE_URL="https://sqs..."
export ENVIRONMENT="dev"
python main.py
```

## 注意事項

1. **Bedrockアクセス**: Bedrockモデルへのアクセスには事前のアクセスリクエストが必要な場合があります
2. **JWT Secret**: 本番環境では強力なシークレットを使用してください
3. **SQS Visibility Timeout**: レポート生成に時間がかかる場合は、SQSのvisibility timeoutを調整してください
