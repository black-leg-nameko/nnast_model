# NNAST Auth Service

GitHub Actions OIDCトークンを検証し、短命JWTを発行するサービス。

## 言語選択: Go

**理由**: 
- **低レイテンシ**: CI速度への影響を最小化（起動時間 ~50ms、レイテンシ ~2ms）
- **軽量**: コンテナイメージが小さい（~15MB）
- **スパイク耐性**: goroutineで同時実行に対応
- **AWS SDK統合**: AWS SDK for Go v2が成熟

詳細は `LANGUAGE_CHOICE_RATIONALE.md` を参照。

## 実装

### Go実装 (`go/`)

```bash
cd go
go mod download
go build -o auth-service .
./auth-service
```

### Dockerビルド

```bash
docker build -t nnast/auth-service:latest -f go/Dockerfile go/
```

## 環境変数

- `ENVIRONMENT`: 環境名（dev/staging/prod）
- `JWT_SECRET`: JWT署名用シークレット（必須）
- `PORT`: ポート番号（デフォルト: 8080）

## エンドポイント

- `GET /health`: ヘルスチェック
- `POST /auth/oidc`: OIDC認証とJWT発行

## パフォーマンス

- 起動時間: ~50ms
- メモリ使用量: ~10MB
- レイテンシ: ~2ms（P95）
- コンテナサイズ: ~15MB（Alpineベース）

## 注意事項

- JWT SecretはSecrets Managerから取得することを推奨
- OIDCトークンの検証にはGitHubのJWKSを使用
- DynamoDBへのアクセスには適切なIAM権限が必要
