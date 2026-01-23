# NNAST Security Dashboard

NNAST のセキュリティダッシュボード（管理画面）です。

## 機能

- **ダッシュボード**: 脆弱性のサマリー、トレンド、OWASP分布
- **プロジェクト管理**: GitHubリポジトリの登録・設定
- **スキャン履歴**: 実行されたスキャンの一覧・詳細
- **Findings**: 検出された脆弱性の一覧・詳細・ステータス管理
- **監査ログ**: アクション履歴の追跡

## セットアップ

### 開発環境

```bash
cd services/admin_service/frontend
npm install
npm run dev
```

### 環境変数

```bash
cp .env.example .env
```

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `VITE_API_BASE_URL` | バックエンドAPIのURL | 空（デモモード） |
| `VITE_DEMO_MODE` | デモモード（モックデータ使用） | `true` |
| `VITE_GITHUB_CLIENT_ID` | GitHub OAuth Client ID | - |

### デモモード

`VITE_DEMO_MODE=true` または `VITE_API_BASE_URL` が未設定の場合、モックデータを使用したデモモードで動作します。

## Vercel へのデプロイ

### 1. Vercel CLIを使用

```bash
npm i -g vercel
cd services/admin_service/frontend
vercel
```

### 2. GitHub連携

1. Vercelダッシュボードで「New Project」
2. GitHubリポジトリを選択
3. Root Directory を `services/admin_service/frontend` に設定
4. Framework Preset を `Vite` に設定
5. Environment Variables を設定:
   - `VITE_DEMO_MODE`: `true`

### デプロイ後のURL例

```
https://nnast-dashboard.vercel.app
```

## 画面構成

```
/login          - ログイン画面
/               - ダッシュボード（サマリー）
/projects       - プロジェクト一覧
/scans          - スキャン履歴
/findings       - 脆弱性一覧
/findings/:id   - 脆弱性詳細
/audit-logs     - 監査ログ
```

## 技術スタック

- **React 18** - UIライブラリ
- **TypeScript** - 型安全な開発
- **Vite** - 高速ビルドツール
- **React Router** - ルーティング
- **React Query** - データフェッチング・キャッシュ
- **Axios** - HTTPクライアント

## ディレクトリ構成

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts      # APIクライアント
│   │   └── mockData.ts    # デモ用モックデータ
│   ├── components/
│   │   └── Layout.tsx     # レイアウトコンポーネント
│   ├── contexts/
│   │   └── AuthContext.tsx # 認証コンテキスト
│   ├── pages/
│   │   ├── LoginPage.tsx
│   │   ├── DashboardPage.tsx
│   │   ├── ProjectsPage.tsx
│   │   ├── ScansPage.tsx
│   │   ├── FindingsPage.tsx
│   │   ├── FindingDetailPage.tsx
│   │   └── AuditLogsPage.tsx
│   ├── types/
│   │   └── index.ts       # TypeScript型定義
│   ├── App.tsx
│   ├── App.css
│   └── main.tsx
├── vercel.json
└── package.json
```

## プロダクション環境への移行

デモモードからプロダクション環境に移行する場合：

1. バックエンドAPI（`services/admin_service/go`）をデプロイ
2. 環境変数を設定:
   ```
   VITE_API_BASE_URL=https://api.your-domain.com/api/v1
   VITE_DEMO_MODE=false
   VITE_GITHUB_CLIENT_ID=your-github-client-id
   ```
3. GitHub OAuth Appを設定
4. Vercelで再デプロイ

## デモデイ用

デモデイでは以下のポイントをプレゼン：

1. **プライバシー優位のアーキテクチャ**
   - 検出はローカルCI（GPG+GNN）
   - クラウドは説明生成（LLM）のみ
   - コード全体を送らない

2. **ダッシュボードの機能**
   - OWASP Top 10に沿った分類
   - データフローの可視化
   - 修正提案の自動生成

3. **CI/CD統合**
   - GitHub Actionsとの連携
   - PRごとの自動スキャン
   - Issue自動作成
