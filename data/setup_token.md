# GitHub Token設定ガイド

## 方法1: 環境変数として設定（推奨）

### 一時的な設定（現在のターミナルセッションのみ）

```bash
export GITHUB_TOKEN=your_token_here
```

### 永続的な設定（推奨）

1. `~/.zshrc`ファイルを編集：

```bash
# エディタで開く（例：VS Code）
code ~/.zshrc

# または、nanoで開く
nano ~/.zshrc
```

2. ファイルの末尾に以下を追加：

```bash
# GitHub API Token for NNAST
export GITHUB_TOKEN=your_token_here
```

3. 設定を反映：

```bash
source ~/.zshrc
```

4. 確認：

```bash
echo $GITHUB_TOKEN
```

## 方法2: .envファイルを使用（プロジェクト単位）

プロジェクトルートに`.env`ファイルを作成して、そこにトークンを保存する方法です。

### 1. .envファイルを作成

```bash
cd /Users/ryutokitajima/works/nnast_model
echo "GITHUB_TOKEN=your_token_here" > .env
```

### 2. python-dotenvを使用して読み込む

現在の実装では、環境変数を直接読み込むようになっていますが、`.env`ファイルをサポートするには、`python-dotenv`を使用できます。

```bash
pip install python-dotenv
```

その後、スクリプトの先頭で：

```python
from dotenv import load_dotenv
load_dotenv()
```

## 推奨スコープ設定

このプロジェクト（NNAST Dataset Collection）で使用するGitHub APIの機能に基づく推奨設定：

### ✅ 推奨設定（最小権限）

**スコープ: `public_repo` のみ**

- ✅ 公開リポジトリの検索（`/search/commits`）
- ✅ 公開リポジトリのコミット詳細取得（`/repos/{owner}/{repo}/commits/{sha}`）
- ✅ 公開リポジトリのファイル内容取得（`/repos/{owner}/{repo}/contents/{path}`）
- ✅ 認証されたリクエストとして扱われ、レート制限が緩和（5,000リクエスト/時）

### ⚠️ 追加スコープ（通常は不要）

**スコープ: `repo`**
- プライベートリポジトリへのアクセスが必要な場合のみ
- このプロジェクトは公開リポジトリのみを対象とするため、通常は不要です
- セキュリティリスクを最小化するため、不要な場合は選択しないことを強く推奨します

## GitHub Personal Access Tokenの取得方法

1. GitHubにログイン
2. 右上のプロフィール画像をクリック → **Settings**
3. 左サイドバーの一番下 → **Developer settings**
4. **Personal access tokens** → **Tokens (classic)**
5. **Generate new token (classic)** をクリック
6. 以下の情報を入力：
   - **Note**: `NNAST Dataset Collection`（任意の名前）
   - **Expiration**: 必要に応じて設定（90日、1年など）
   - **Select scopes**: 以下のスコープを選択
     - ✅ `public_repo` - 公開リポジトリへの読み取りアクセス（**推奨・必須**）
       - このプロジェクトは公開リポジトリの検索と読み取りのみを行います
       - コミット検索、コミット詳細取得、ファイル内容取得に必要です
       - 認証されたリクエストとして扱われ、レート制限が緩和されます（5,000リクエスト/時）
     - ⚠️ `repo` - プライベートリポジトリへのアクセス（**通常は不要**）
       - プライベートリポジトリを対象とする場合のみ選択してください
       - セキュリティリスクを最小化するため、不要な場合は選択しないことを推奨します
7. **Generate token** をクリック
8. **トークンをコピー**（この画面でしか表示されません！）

## セキュリティのベストプラクティス

1. **最小限の権限を付与（最重要）**
   - ✅ **推奨**: `public_repo` スコープのみを選択
   - ❌ **非推奨**: 不要な `repo` スコープは選択しない
   - このプロジェクトは公開リポジトリの読み取りのみを行うため、`public_repo` で十分です

2. **トークンは絶対にGitにコミットしない**
   - `.env`ファイルは既に`.gitignore`に含まれています
   - 環境変数もコードに直接書かないでください

3. **定期的にトークンを更新**
   - 期限切れ前に新しいトークンを生成
   - 有効期限は90日〜1年程度を推奨（用途に応じて調整）

4. **トークンが漏洩した場合**
   - すぐにGitHubでトークンを削除
   - 新しいトークンを生成

## 確認方法

設定が正しく行われているか確認：

```bash
# 環境変数が設定されているか確認
echo $GITHUB_TOKEN

# 実際にAPIが動作するかテスト
python3 -c "
import os
token = os.getenv('GITHUB_TOKEN')
if token:
    print('✓ GITHUB_TOKEN is set')
    print(f'  Token length: {len(token)} characters')
    print(f'  Token preview: {token[:10]}...')
else:
    print('✗ GITHUB_TOKEN is not set')
"
```

## トラブルシューティング

### 環境変数が設定されない

1. `~/.zshrc`を編集した後、`source ~/.zshrc`を実行しましたか？
2. 新しいターミナルウィンドウを開いてみてください
3. 設定ファイルの構文エラーがないか確認：

```bash
# 構文チェック
zsh -n ~/.zshrc
```

### トークンが無効

1. GitHubでトークンが有効か確認
2. スコープが正しく設定されているか確認
3. トークンの有効期限が切れていないか確認

