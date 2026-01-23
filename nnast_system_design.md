# NNAST Hybrid SAST System Design Doc (Final)
Version: v1.0  
Owner: Ryuto  
Mode: Hybrid Only (Local Detect + Cloud Report)  
Target: Enterprise-friendly SAST with privacy-first architecture

---

## 1. 背景 / 問題意識

LLM + CPG は精度が強い一方で、実運用では以下が障壁になる。

- 推論コストが高い（GPU/トークン課金）
- レイテンシが大きい（CIが遅くなり無効化されがち）
- 機密コードの外部送信が導入審査で詰む
- 監査・再現性が取りづらい（LLM単体はブラックボックス化しやすい）

そこで NNAST は以下の戦略を取る。

- **検出（Detection）の本体はローカル（CI内）で完結**
- **クラウドは “診断レポート生成（Explain / Fix）” に限定**
- **送信データは最小化（diff + サブグラフ + マスキング）**
- **レポートの最終出力は GitHub Issue として残す（運用しやすい）**

---

## 2. NNAST の優位性（勝ち筋）

### 2.1 プライバシー優位
- コード全体をクラウドへ送らない（ポリシー/設計で担保）
- LLMは“説明生成”中心で、クラウドに送るデータ量を制御可能
- 企業の法務・監査に刺さる（導入審査に通る）

### 2.2 運用優位（現場で嫌われにくい）
- 開発者は管理画面を見なくていい
- PR/CI上で自動検査、Issueとして蓄積しトリアージできる
- CI速度の劣化を Top-K / レート制御で抑える

### 2.3 技術優位（LLMを主役にしない）
- CPG（構造） + CodeBERT（意味圧縮） + GNN（推論）で検出
- LLMは出力（レポート・修正案）に寄せる
- 根拠（グラフパス/データフロー）を示せる

---

## 3. 要求仕様

### 3.1 機能要件
- GitHub Actions から NNAST を実行できる
- **フルCPG** を構築して解析する（新旧関数のつながり重視）
- CodeBERTでノード特徴量を付与し、GNNで脆弱候補をランキング
- 上位候補のみクラウドへ送信し、LLMで診断レポート（Markdown）を生成
- 生成された Markdown を GitHub Issue として作成
- 重複Issueを回避し、既存Issueがある場合はコメント追記する
- 管理画面でテナント/リポジトリ登録、実行履歴、監査ログが見れる

### 3.2 非機能要件
- マルチテナント（複数企業/複数リポジトリ同時）対応
- スパイク耐性（同時実行・バーストを落とさない）
- 監査可能（誰がいつどのrepoで何をしたか）
- 最小権限（OIDC+短命トークン、静的Secrets依存を避ける）
- コスト制御（Top-K、レート制御、キャッシュ）

---

## 4. 全体アーキテクチャ（Hybrid Only）

### 4.1 鳥瞰図

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Customer GitHub Org                                                  │
│                                                                     │
│  Developer → PR/Push                                                 │
│      │                                                              │
│      ▼                                                              │
│  GitHub Actions Runner (Local)                                       │
│    - Full CPG build                                                  │
│    - CodeBERT embeddings                                             │
│    - GNN inference                                                   │
│    - Top-K findings                                                  │
│    - Subgraph extract + Masking                                      │
│      │                                                              │
│      │  OIDC + minimized payload                                     │
│      ▼                                                              │
│  NNAST Cloud (AWS)                                                   │
│    API Gateway → Auth(OIDC verify) → SQS → ECS Worker → Bedrock LLM   │
│      │                                      │                       │
│      └──────────────────────────────┬───────┘                       │
│                                     ▼                               │
│                               Report (Markdown)                      │
│                                     │                               │
│                                     ▼                               │
│  GitHub Actions creates Issue (issues:write)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 方針（固定）
- **Detection = Local**
- **Explain/Fix = Cloud**
- **Issue作成 = Local（GitHub ActionsのGITHUB_TOKENで実施）**

> AWS側がGitHubへの書き込み権限を持たない。これがセキュアで運用も楽。

---

## 5. ローカル（CI）設計

### 5.1 配布形態
- GitHub Action: `nnast/scan-action@v1`
- 内部は Docker 実行（再現性・依存固定）
- 解析エンジンは CLI でも動くが、ユーザーに手動インストールさせない

### 5.2 GitHub Actions 権限
```yaml
permissions:
  contents: read
  issues: write
  id-token: write
  pull-requests: read
```

### 5.3 ローカル処理フロー（Full CPG）
1. Repository checkout
2. **フルCPG生成**（AST/CFG/DFG統合）
3. CPGノードに CodeBERT embedding を付与
4. GNN推論（ノード/サブグラフ分類 or ranking）
5. Findingsを整形（severity, rule, location, evidence）
6. Top-K 抽出（例：K=3〜5）
7. 送信用 payload を作る
   - サブグラフ抽出
   - 周辺コード断片（必要最小）
   - **マスキング/匿名化**
8. クラウドに report 生成リクエスト
9. 返ってきた Markdown で Issue を作成/更新（重複防止）

### 5.4 Full CPG のコスト対策（必須）
フルCPGは重いので、以下を実装する。

- **CPGキャッシュ**  
  - `commit_sha` をキーにローカルキャッシュ（Actions cache）
  - 近いcommit間は差分更新（Incremental build）を検討
- **解析の分離**  
  - 1回のCIで「フルCPG構築」はやるが、GNN推論はTop-Kを厳格に制御
- **タイムアウト**  
  - 一定時間で fail-safe（Issue生成はスキップ、CIを落とさない）
- **nightlyは採用しない（Hybrid一本）**  
  - ただし “PRごとに実行するが、Top-K制限でクラウド負荷を抑える”

---

## 6. クラウド（AWS）設計

### 6.1 役割
- ローカルから送られた最小データを元に
  - 診断レポート（Markdown）を生成
  - 修正案（diff例）を生成
  - （任意）FPっぽい場合の二次判定コメントを付加
- ジョブのキューイングとスケール
- 監査ログ/メタデータ管理

### 6.2 AWS コンポーネント
- **API Gateway**
  - `/auth/oidc` : OIDC token検証 → 短命JWT発行
  - `/report` : report生成ジョブ投入
- **Auth Service（ECS/Fargate or Lambda）**
  - GitHub OIDC検証（issuer/audience/sub）
  - tenant/projectの紐付け
- **SQS (Standard)**
  - report生成ジョブをバッファ
  - DLQ（Dead Letter Queue）を用意
- **ECS Fargate Workers**
  - SQSから取り出し、Bedrockへ問い合わせ
- **Amazon Bedrock**
  - LLM推論（report生成）
- **DynamoDB**
  - `Job` メタデータ（status, tenant_id, repo, timestamps）
- **S3**
  - reportアーティファクト保存（機密を含まない最小情報）
- **CloudWatch / CloudTrail / KMS**
  - 監査、ログ、暗号化

### 6.3 AWS ネットワーク
- VPC内でECS稼働
- 必要に応じて VPC Endpoint（S3など）
- 外部公開は API Gateway のみ

---

## 7. 認証/認可（ベストプラクティス）

### 7.1 方針
- 静的APIキーを配らない（漏れる前提だから）
- **GitHub Actions OIDC** を利用し、短命トークンを発行する

### 7.2 OIDC検証
- issuer: `https://token.actions.githubusercontent.com`
- audience: `nnast-cloud`
- subject: `repo:{org}/{repo}:environment:{env}` 等で制限

管理画面で repo/env を登録しておき、
一致する OIDC `sub` のみ許可。

### 7.3 発行するJWT（短命）
- exp: 5〜15分
- claims:
  - tenant_id
  - project_id
  - repo
  - scopes: `report:generate`

---

## 8. データ最小化（Privacy Policy as Architecture）

### 8.1 クラウドに送るもの（OK）
- findings（位置/重大度/ルールID）
- サブグラフ（抽象化したノード/エッジ/型/ラベル）
- diffの該当断片（マスキング済み）
- 周辺コード断片（必要最小、マスキング済み）

### 8.2 送らないもの（NG）
- リポジトリ全体
- 生の秘密情報（keys, tokens, .env）
- 大量の生コード
- DB接続文字列など

### 8.3 マスキング戦略（最低限）
- 識別子：`FUNC_1`, `VAR_2` へ置換 or HMACハッシュ化
- 文字列リテラル：redact（URL/トークンっぽい文字列）
- 秘密パターン検知：JWT, API keys, private keys などを削除

---

## 9. Issue 運用（最終成果物）

### 9.1 なぜIssueか
- PRコメントは流れる（忘れられる）
- Issueはトリアージできる（担当/ラベル/期限）
- 組織運用に刺さる

### 9.2 Issue作成主体
- **GitHub Actions が作成する**
  - `GITHUB_TOKEN` を利用
  - AWS側がGitHub権限を持たない

### 9.3 重複防止（Idempotency）
Issue fingerprint を生成し、同一指摘は更新に回す。

例 fingerprint:
`SHA256(repo + rule_id + file + function + sink_line)`

- タイトル例  
  `[NNAST][HIGH][CWE-22] Path Traversal in src/a.py (fp:abcd1234)`

処理：
- 既存Issue検索（label + fp 文字列）
- あればコメント追記（new evidence / new commit）
- なければ新規Issue作成

### 9.4 Issue数制御
- PRあたり最大 Issue 作成数: 3〜5
- それ以上はまとめIssue（1件）に集約


## 10. API設計（Cloud）

### 10.1 `POST /auth/oidc`
入力:
- github_oidc_token

出力:
- short_lived_jwt

### 10.2 `POST /report`
入力（minimized payload）:
```json
{
  "tenant_id": "t_xxx",
  "project_id": "p_xxx",
  "repo": "org/repo",
  "job_id": "j_xxx",
  "findings": [
    {
      "fingerprint": "abcd1234",
      "severity": "HIGH",
      "rule_id": "NNAST-PT-001",
      "cwe": "CWE-22",
      "location": {"file": "src/a.py", "line": 142},
      "subgraph": {"nodes": [...], "edges": [...]},
      "code_snippet_masked": "..."
    }
  ],
  "limits": {"max_issues": 5}
}
```

出力:
```json
{
  "job_id": "j_xxx",
  "reports": [
    {
      "fingerprint": "abcd1234",
      "title": "...",
      "labels": ["security", "nnast", "severity:high"],
      "markdown": "..."
    }
  ]
}
```

---

## 11. データモデル（管理画面 / 内部）

### 11.1 Tenant
- tenant_id
- name
- plan
- created_at

### 11.2 Project
- project_id
- tenant_id
- github_org
- github_repo
- allowed_subject_patterns（OIDC sub allowlist）
- settings（Top-K, issue limits, severity thresholds）

### 11.3 Job
- job_id
- tenant_id
- project_id
- repo
- status（queued/running/done/failed）
- created_at / finished_at
- counts（findings/report_generated）

### 11.4 AuditLog
- tenant_id
- actor（repo/workflow）
- action（auth/report）
- timestamp
- result

---

## 12. スケーラビリティ / 同時実行

### 12.1 想定負荷
- 複数テナントが同時に `/report` を叩く
- PRが集中する時間帯にスパイクする

### 12.2 対策（必須）
- SQSでバッファ
- ECS worker を水平スケール
- Bedrockのレート制限に備えて
  - concurrency制限
  - 429バックオフ・リトライ
- Top-K（PRあたりクラウド呼び出し上限）でコスト制御

### 12.3 ボトルネック
- Bedrock throttling（最重要）
- GitHub Issue API rate limit（Issue乱立防止で回避）
- DynamoDB hot partition（PK設計で回避）

---

## 13. 監視 / 運用

### 13.1 監視項目
- SQS backlog（キュー深さ）
- ECS task数、失敗率
- Bedrock 429率、平均レイテンシ
- Job status（失敗率）
- コスト（LLM呼び出し回数）

### 13.2 障害時の設計
- クラウド障害時
  - ローカル検出は継続可能
  - Issue作成はスキップ or 簡易結果のみ
- DLQに溜まったジョブは再実行可能

---

## 14. セキュリティ（最小権限の徹底）

- GitHub Actions:
  - `issues: write` は必要最小
  - `contents: read` のみ
- AWS:
  - KMSで暗号化
  - CloudTrailで監査
- データ:
  - 生コードを保存しない
  - S3には機密を含まない成果物だけ

---

## 15. トレードオフ（正直に書く）

### フルCPG採用の利点
- 新旧関数のつながりを保持できる
- 影響伝播・長距離データフローに強い
- 精度と根拠の質が上がる

### フルCPG採用の欠点
- 計算コスト増（CI時間が伸びる）
- 実装が難しい（増分更新やキャッシュが必要）

→ 対策として **CPGキャッシュ** と **Top-K制御** を必須とする。


## 18. まとめ（この設計の核）
- **検出はローカル（フルCPG + CodeBERT + GNN）**
- **クラウドは説明生成（LLM）だけ**
- **結果はIssue化で運用に乗る**
- **OIDCでSecrets不要、監査と導入審査に強い**

この設計は「精度だけの研究」じゃなく、
**実導入できるSAST**として成立する。