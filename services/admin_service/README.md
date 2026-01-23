# NNAST Admin Service

管理画面用のAPIサービスとフロントエンドです。

## 構成

- **Backend (Go)**: `/go/` - 管理用REST API
- **Frontend (React + TypeScript)**: `/frontend/` - 管理画面UI

## 機能

### API機能
- テナント管理（一覧、作成、取得）
- プロジェクト管理（一覧、作成、取得、更新、削除）
- ジョブ管理（一覧、取得、フィルタリング）
- 監査ログ（一覧、フィルタリング）

### フロントエンド機能
- テナント一覧・作成
- プロジェクト一覧・作成・編集・削除
- ジョブ一覧・フィルタリング
- 監査ログ一覧・フィルタリング

## パフォーマンス最適化

### バックエンド
- **ページネーション**: デフォルト20件、最大100件
- **GSI活用**: DynamoDBのGlobal Secondary Indexを活用した高速クエリ
- **効率的なスキャン**: 必要な場合のみScanを使用、通常はQueryを使用

### フロントエンド
- **React Query**: データキャッシュと自動リフェッチ（5分間のstale time）
- **コード分割**: vendorとqueryライブラリを分離
- **Gzip圧縮**: Nginxで静的アセットを圧縮
- **キャッシュヘッダー**: 静的アセットに1年間のキャッシュ

## セットアップ

### Backend

```bash
cd services/admin_service/go
go mod download
go build -o admin-service .
./admin-service
```

### Frontend

```bash
cd services/admin_service/frontend
npm install
npm run dev
```

## 環境変数

### Backend
- `ENVIRONMENT`: 環境名（dev/staging/prod）
- `JWT_SECRET`: JWT署名用シークレット（必須）
- `PORT`: ポート番号（デフォルト: 8080）

### Frontend
- `VITE_API_BASE_URL`: APIベースURL（デフォルト: `/api/v1`）

## Dockerビルド

### Backend
```bash
cd services/admin_service/go
docker build -t nnast/admin-service:latest .
```

### Frontend
```bash
cd services/admin_service/frontend
docker build -t nnast/admin-frontend:latest .
```

## APIエンドポイント

### 認証
すべてのAPIエンドポイントはJWT認証が必要です。リクエストヘッダーに以下を追加：
```
Authorization: Bearer <JWT_TOKEN>
```

### エンドポイント一覧

#### Tenants
- `GET /api/v1/tenants` - テナント一覧
- `GET /api/v1/tenants/{tenant_id}` - テナント取得
- `POST /api/v1/tenants` - テナント作成

#### Projects
- `GET /api/v1/projects` - プロジェクト一覧（クエリパラメータ: `tenant_id`）
- `GET /api/v1/projects/{project_id}` - プロジェクト取得
- `POST /api/v1/projects` - プロジェクト作成
- `PUT /api/v1/projects/{project_id}` - プロジェクト更新
- `DELETE /api/v1/projects/{project_id}` - プロジェクト削除

#### Jobs
- `GET /api/v1/jobs` - ジョブ一覧（クエリパラメータ: `tenant_id`, `project_id`, `status`）
- `GET /api/v1/jobs/{job_id}` - ジョブ取得

#### Audit Logs
- `GET /api/v1/audit-logs` - 監査ログ一覧（クエリパラメータ: `tenant_id`, `actor`）

### ページネーション
すべての一覧APIは以下のクエリパラメータをサポート：
- `page`: ページ番号（デフォルト: 1）
- `page_size`: 1ページあたりの件数（デフォルト: 20、最大: 100）

レスポンス形式：
```json
{
  "items": [...],
  "page": 1,
  "page_size": 20,
  "total": 100,
  "total_pages": 5
}
```

## デプロイ

### ECSへのデプロイ
1. ECRにイメージをプッシュ
2. ECSタスク定義を作成
3. ECSサービスを作成
4. API Gatewayに統合

### フロントエンドのデプロイ
1. S3バケットにビルド成果物をアップロード
2. CloudFrontで配信（オプション）
3. またはNginxコンテナとしてECSで実行
