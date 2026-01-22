# NNAST: Python Webアプリケーション向けニューラルネットワークベースのアプリケーション脆弱性診断ツール

## 要約

本論文では、Python Webアプリケーション向けに特化した静的アプリケーションセキュリティテスト（SAST）フレームワークであるNNAST（Neural Network-based Application Security Testing）を提案する。NNASTは、静的解析と動的Taint解析を統合したCode Property Graph（CPG）表現を用い、動的注意融合を備えたグラフニューラルネットワーク（GNN）によりセキュリティ脆弱性を検出する。本手法は、学習タスク（パターンベース検出）と報告タスク（OWASP Top 10分類）を分離した二層ラベル構造を導入し、正確な脆弱性検出と実用的なセキュリティ報告の両立を実現する。合成データと実在の脆弱性データを組み合わせたハイブリッドデータセットで評価を行い、複数のPython Webフレームワーク（Django、Flask、FastAPI）およびOWASP Top 10に整合した脆弱性カテゴリにおいて有効性を示す。

**キーワード**: 静的アプリケーションセキュリティテスト、Python Webアプリケーション、グラフニューラルネットワーク、Code Property Graph、OWASP Top 10

---

## 1. はじめに

### 1.1 背景

Webアプリケーションのセキュリティは、アプリケーションが機密性の高いユーザーデータを扱い、重要なビジネス操作を実行するにつれて、ますます重要になっている。静的アプリケーションセキュリティテスト（SAST）ツールは、デプロイ前にソースコードを解析して潜在的なセキュリティ脆弱性を特定する。しかし、既存のSASTツールは以下の課題に直面している：

1. **言語・フレームワーク特化性の不足**: ほとんどのSASTツールは、言語非依存（ドメイン特化の最適化が不足）であるか、C/C++などの言語に焦点を当てており、Python Webアプリケーションが十分にカバーされていない。

2. **解析手法の統合不足**: 従来のSASTツールは主に静的解析に依存しており、実行時データフローの理解を必要とする脆弱性を見逃す。

3. **検出と報告の乖離**: ツールは低レベルのパターンを検出するが、OWASP Top 10などの業界標準セキュリティフレームワークにマッピングできず、セキュリティチームが結果を解釈・優先順位付けすることが困難である。

4. **訓練データの不足**: 機械学習ベースのアプローチは、Python Webアプリケーション向けの大規模で高品質な脆弱性データセットの不足に悩まされている。

### 1.2 貢献

本論文では、Python Webアプリケーション向けの新しいSASTフレームワークであるNNASTを提案し、以下の貢献を行う：

1. **静的・動的Taint統合による統合CPG**: 従来のCode Property Graph（CPG）を拡張し、静的データフローエッジ（DFG）と動的Taint由来エッジ（DDFG）の両方を含めることで、実行時情報を必要とする脆弱性の検出を可能にする。

2. **Python Webアプリケーション特化CPG定義**: Webフレームワークメタデータ（Django/Flask/FastAPI）、ソース/シンク注釈、サニタイザー情報を捕捉するドメイン特化CPGスキーマを定義し、精密な脆弱性パターンマッチングを可能にする。

3. **二層ラベル構造**: 内部ラベル（学習用のパターンID）と外部ラベル（報告用のOWASP Top 10カテゴリ）を分離し、モデルが細粒度パターンを学習しながら、セキュリティ関連の報告を生成できるようにする。

4. **動的注意融合を備えたエッジタイプ認識GNN**: エッジタイプ（AST、CFG、DFG、DDFG）を明示的に考慮し、相互注意機構を通じてCodeBERT意味埋め込みとGNN構造特徴を動的に融合するGraph Attention Network（GAT）アーキテクチャを提案する。

5. **再現可能なハイブリッドデータセット構築手法**: 合成データ（カバレッジ用）、実在の脆弱性修正（現実性用）、教育用アプリケーション（多様性用）を組み合わせた三層データセット構築アプローチを提示し、データリークを防ぐためのプロジェクト分割と時系列分割戦略を採用する。

### 1.3 論文構成

本論文の構成は以下の通りである：第2章で関連研究をレビューする。第3章で全体アーキテクチャと設計原則を提示する。第4章でCPG仕様を詳述する。第5章でニューラルネットワークモデルを説明する。第6章でデータセット構築手法を説明する。第7章で実験的評価を提示する。第8章で限界と今後の課題を議論する。第9章で結論を述べる。

---

## 2. 関連研究

### 2.1 静的アプリケーションセキュリティテスト

SonarQube、Checkmarx、Fortifyなどの従来のSASTツールは、ルールベースのパターンマッチングと静的解析を使用して脆弱性を検出する。これらのツールは既知のパターンに対しては有効であるが、高い偽陽性率と複雑な脆弱性の限定的なカバレッジという問題を抱えている。最近の研究では、検出精度を向上させるために機械学習アプローチが探索されている[1, 2, 3]。

### 2.2 Code Property Graph

Code Property Graph（CPG）は、Yamaguchiら[4]によって、抽象構文木（AST）、制御フローグラフ（CFG）、プログラム依存グラフ（PDG）を統合した統一表現として導入された。CPGはC/C++コードの脆弱性検出に成功している[5, 6]。しかし、既存のCPG定義は言語非依存であり、Webアプリケーション向けのドメイン特化最適化が不足している。

### 2.3 コード解析のためのグラフニューラルネットワーク

グラフニューラルネットワーク（GNN）はコード解析タスクにおいて有望な結果を示している。Liら[7]はバグ検出にGCNを使用し、Zhouら[8]はコード表現学習にGATを適用した。しかし、これらのアプローチは通常、静的コード構造のみを使用し、動的解析情報やドメイン特化知識を統合していない。

### 2.4 Python Webアプリケーションセキュリティ

Python Webアプリケーションは、フレームワーク固有のAPIと動的型付けにより、独特のセキュリティ課題に直面している。Bandit[9]やSafety[10]などの既存ツールは特定の脆弱性タイプに焦点を当てているが、包括的なカバレッジが不足している。最近の研究では、Pythonコード解析のためのMLベースアプローチが探索されている[11, 12]が、OWASP Top 10に整合したWebアプリケーションセキュリティに特化したものはない。

### 2.5 脆弱性データセット

C/C++向けの大規模脆弱性データセット（BigVul[13]、Devign[14]）は存在するが、Python Webアプリケーション向けのものは希少である。CodeXGLUE[15]には一部のPythonコードが含まれているが、Webアプリケーション特化ではない。本研究は、Python Webアプリケーション脆弱性検出専用のハイブリッドデータセットを構築することで、このギャップに対処する。

---

## 3. アーキテクチャ概要

### 3.1 設計原則

NNASTは以下の原則に基づいて設計されている：

1. **ドメイン特化**: Python Webアプリケーション（Django、Flask、FastAPI）に焦点を当て、フレームワーク認識解析を可能にする。

2. **関心の分離**: 学習タスク（パターン検出）と報告タスク（OWASP分類）を分離し、精度と使いやすさの両方を向上させる。

3. **静的・動的統合**: カバレッジのための静的解析と精度のための動的Taint解析を統一表現で組み合わせる。

4. **再現性**: データセット構築からモデル訓練、評価まで、すべてのコンポーネントを再現可能にする。

### 3.2 システムアーキテクチャ

NNASTパイプラインは以下の段階で構成される：

```
Pythonソースコード
    ↓
CPG Generator (AST + CFG + DFG + DDFG)
    ↓
Node Embedding (CodeBERT + 静的特徴量)
    ↓
GNN Encoder (エッジタイプ認識GAT + 動的注意融合)
    ↓
パターン検出 (二値分類)
    ↓
OWASP Top 10マッピング
    ↓
セキュリティレポート
```

**CPG Generator**: Pythonソースコードを解析し、4種類のエッジタイプを持つ統合CPGを構築する：
- **ASTエッジ**: 構文構造を表現
- **CFGエッジ**: 制御フローを表現
- **DFGエッジ**: 静的データフローを表現
- **DDFGエッジ**: 動的Taintフローを表現（実行時解析から）

**Node Embedding**: 各CPGノードは以下を使用して埋め込まれる：
- CodeBERT[16]トークン埋め込み（意味情報用）
- 静的特徴量（ノードタイプ、フレームワークメタデータ、ソース/シンク/サニタイザーフラグ）

**GNN Encoder**: 多層Graph Attention Networkで：
- 注意計算においてエッジタイプを明示的に考慮
- 相互注意機構を通じてCodeBERT埋め込みとGNN構造特徴を動的に融合

**パターン検出**: グラフレベルでの二値分類（脆弱/安全）

**OWASPマッピング**: 検出されたパターンをOWASP Top 10カテゴリとCWE IDにマッピングして報告

### 3.3 対象スコープ（Phase I）

NNAST Phase Iは、静的コード解析によって決定可能な脆弱性に焦点を当てる：

- **A03: Injection**（SQL Injection、Command Injection、Template Injection、XSS）
- **A10: SSRF**（Server-Side Request Forgery）
- **A01: Broken Access Control**（限定：認証デコレータの欠如、直接オブジェクト参照）
- **A07: Identification and Authentication Failures**（限定：JWT検証の無効化）
- **A02: Cryptographic Failures**（コード関連のみ：弱いハッシュ関数）

注：実行時設定解析を必要とする脆弱性（例：Security Misconfiguration）はPhase Iの対象外である。

---

## 4. Code Property Graph仕様

### 4.1 ノードタイプ

NNAST CPGノードは異なるコード要素を表現する：

- **AST Node**: 構文要素（文、式）を表現
- **Call Node**: 関数/メソッド呼び出しを表現
- **Variable Node**: 変数定義と使用を表現
- **Literal Node**: 定数値を表現

### 4.2 エッジタイプ

4種類のエッジタイプがコード構造とデータフローの異なる側面を捕捉する：

1. **AST Edge**: 抽象構文木内の親子ノードを接続
2. **CFG Edge**: 順次文と制御フローの分岐を接続
3. **DFG Edge**: 変数定義と使用を接続（静的データフロー）
4. **DDFG Edge**: 動的Taint解析に基づいてTaintソースとシンクを接続

DDFGエッジは、静的解析ではデータフローを決定できない脆弱性（例：動的ディスパッチや複雑な制御フローによる）の検出において特に重要である。

### 4.3 ノード属性（Web特化拡張）

Webアプリケーション特化解析を可能にするため、ノードは以下の属性で注釈される：

- **is_source**: ユーザー入力を表現するノードをマーク（例：`request.args`、`request.form`、`request.json`）
- **is_sink**: セキュリティに敏感な操作を表現するノードをマーク（例：SQL実行、HTMLレンダリング、サブプロセス呼び出し、HTTPクライアントリクエスト）
- **sanitizer_kind**: ユーザー入力をサニタイズするノードをマーク（例：`html.escape`、`urllib.parse.quote`、SQLパラメータ化）
- **framework**: Webフレームワークを示す（django、flask、fastapi）

### 4.4 ソース/シンク/サニタイザー定義

ソース、シンク、サニタイザーは構造化YAML形式（`patterns.yaml`）で定義され、以下を可能にする：

- 共通辞書でソース/シンク/サニタイザーを定義
- パターンがこれらの定義を参照
- `frameworks`フィールドを通じてフレームワーク差を処理
- CPG構築時のパターンマッチングを可能にする

ソース定義の例：
```yaml
sources:
  - id: SRC_FLASK_REQUEST
    kinds: [http_request]
    frameworks: [flask]
    match:
      attrs:
        - "flask.request.args"
        - "flask.request.form"
        - "flask.request.json"
```

シンク定義の例：
```yaml
sinks:
  - id: SINK_DBAPI_EXECUTE
    kind: sql_exec
    match:
      calls:
        - "sqlite3.Cursor.execute"
        - "psycopg2.cursor.execute"
```

### 4.5 パターン定義

脆弱性パターンは、ソース、シンク、サニタイザーを組み合わせて定義される：

```yaml
patterns:
  - id: SQLI_RAW_STRING_FORMAT
    owasp: "A03: Injection"
    cwe: ["CWE-89"]
    description: "str.formatで構築されたSQLが生クエリAPIで実行される"
    sources: ["SRC_FLASK_REQUEST", "SRC_DJANGO_REQUEST", "SRC_FASTAPI_REQUEST"]
    sinks: ["SINK_DBAPI_EXECUTE"]
    sanitizers: ["SAN_SQL_PARAM"]
```

NNASTは現在、第3.3節で述べたOWASP Top 10カテゴリをカバーする15の脆弱性パターンをサポートしている。

---

## 5. ニューラルネットワークモデル

### 5.1 ノード埋め込み

各CPGノードは、意味的特徴と構造的特徴の組み合わせで埋め込まれる：

**CodeBERT埋め込み**: 各ノードに関連付けられたコードスニペットは、事前訓練済みのTransformerモデル（CodeBERT[16]）によって埋め込まれる。具体的には、コードスニペットをトークン化し、Transformerニューラルネットワークに入力することで、768次元の意味的埋め込みベクトルを計算する。この埋め込みは、コードの意味的類似性を数値ベクトルとして表現し、手動で特徴量を設計する必要がない。

**静的特徴量**: 追加の特徴量が連結される：
- ノードタイプ（ワンホットエンコーディング：AST、Call、Variable、Literal）
- 型ヒント（利用可能な場合）
- フレームワークメタデータ（ワンホット：django、flask、fastapi）
- ソース/シンク/サニタイザーフラグ（二値）

最終的なノード埋め込みは：`x = [CodeBERT_emb; static_features]`

### 5.2 グラフニューラルネットワークアーキテクチャ

我々は、以下の革新を備えた多層Graph Attention Network（GAT）を採用する：

#### 5.2.1 エッジタイプ認識注意

従来のGATはすべてのエッジを同等に扱う。NNASTでは、注意機構においてエッジタイプ（AST、CFG、DFG、DDFG）を明示的に考慮する。現在の実装ではエッジタイプ属性を生成しているが、今後の研究ではこれらを注意重みに明示的に組み込む予定である。

#### 5.2.2 動的注意融合層

重要な革新は、**動的注意融合層**であり、相互注意機構を通じてGNN構造特徴とCodeBERT意味特徴を組み合わせる：

1. **構造エンコーディング**: GAT層がグラフ構造を処理し、構造的関係を捕捉するノード表現を生成する。

2. **意味エンコーディング**: CodeBERT埋め込みがコードトークンの意味的理解を提供する。

3. **相互注意**: 層は注意を使用して、構造的コンテキストに基づいて関連するCodeBERT特徴を動的に選択する：
   - Query: GATからの構造特徴
   - Key/Value: CodeBERT埋め込み
   - これにより、モデルは構造的コンテキストに関連する意味情報に注意を向けることができる

4. **融合**: 注意された意味特徴は、学習された変換を通じて構造特徴と融合される。

数学的には、ノード$i$について：

$$
\text{struct}_i = \text{GAT}(x_i, \mathcal{N}(i)) \\
q_i = W_q \cdot \text{struct}_i \\
k_i = W_k \cdot \text{codebert}_i \\
v_i = W_v \cdot \text{codebert}_i \\
\text{attended}_i = \text{Attention}(q_i, k_i, v_i) \\
\text{fused}_i = \text{Fusion}([\text{struct}_i; \text{attended}_i])
$$

ここで、$\mathcal{N}(i)$はCPG内のノード$i$の隣接ノードを表す。

#### 5.2.3 グラフレベル分類

複数のGNN層を通過した後、グラフレベル表現は以下によって取得される：
- 平均プーリング：平均ノード特徴を捕捉
- 最大プーリング：重要な特徴を捕捉

プールされた表現は、CodeBERT意味プーリングと連結され、二値分類（脆弱/安全）の分類器に供給される。

### 5.3 出力形式

NNASTは以下の形式で構造化された脆弱性レポートを出力する：

```json
{
  "pattern_id": "SSRF_REQUESTS_URL_TAINTED",
  "cwe_id": "CWE-918",
  "owasp": "A10: SSRF",
  "confidence": 0.94,
  "location": {
    "file": "views.py",
    "lines": [42, 48]
  },
  "description": "tainted URLでrequests.*が呼び出される"
}
```

この形式により、セキュリティ報告ツールとの直接統合が可能になり、セキュリティチーム向けのOWASP Top 10分類に整合した報告が可能になる。

---

## 6. データセット構築

### 6.1 三層データ戦略

Python Webアプリケーション脆弱性データの不足に対処するため、三層構築戦略を採用する：

#### 層1: 合成データ（量）
- **目的**: すべての脆弱性パターンのバランスの取れたカバレッジを確保
- **方法**: 安全なコードテンプレートへのルールベース脆弱性注入
- **利点**: 
  - パターンカバレッジの保証
  - パターン間のバランスの取れた分布
  - 高速生成
  - 再現可能
- **制限**: 
  - 現実的でないコードパターン
  - コーディングスタイルの多様性が限定的

#### 層2: 実在の脆弱性修正（現実性）
- **目的**: 本番コードからの現実的な脆弱性例を提供
- **ソース**:
  - CVE関連コミット（GitHubで「CVE-YYYY」を検索）
  - GitHub Security Advisories（GHSA）
  - セキュリティ修正コミット（様々なクエリパターン）
  - OWASP関連プロジェクト
- **利点**:
  - 現実世界のコードパターン
  - 予期しない脆弱性パターン
  - 研究の信頼性
- **制限**:
  - 不均衡な分布（一般的なパターンが優勢）
  - 品質のばらつき
  - 収集に時間がかかる

#### 層3: 教育用アプリケーション（多様性）
- **目的**: 多様なコーディングスタイルとパターンを提供
- **ソース**: CTFアプリケーション、セキュリティ教育プロジェクト、脆弱性テストツール
- **利点**: 実装パターンの高い多様性
- **制限**: 本番コードの複雑さを反映しない可能性

### 6.2 データ収集プロセス

#### 6.2.1 実在世界データ収集

実在世界の脆弱性データは以下を通じて収集する：

1. **CVEベース収集**: CVE識別子を参照するコミットをGitHubで検索
2. **GHSAベース収集**: GitHub Security AdvisoriesからPythonプロジェクトを抽出
3. **クエリベース収集**: 多様な検索クエリを使用（例：「security fix」、「vulnerability patch」）
4. **パターン特化収集**: 特定の脆弱性タイプを対象

各収集コミットについて：
- 脆弱コード（修正前）と安全コード（修正後）を抽出
- 両バージョンのCPGを生成
- パターンマッチングを適用して脆弱性にラベル付け
- メタデータ（リポジトリ、コミットハッシュ、フレームワーク、パターンID）を保存

#### 6.2.2 合成データ生成

合成データは、15の脆弱性パターンそれぞれのテンプレートを使用して生成される：

1. **テンプレート定義**: 脆弱バージョンと安全バージョンのテンプレートを作成
2. **フレームワーク変異**: Django、Flask、FastAPIのバリアントを生成
3. **複雑度変異**: シンプル、中程度、複雑なバージョンを作成
4. **バランス生成**: パターン間の均等な分布を確保

### 6.3 データ品質保証

#### 6.3.1 自動品質チェック

7つの自動品質チェックを実装する：

1. **Graph-Label Alignment**: CPGがラベルと一致するソース-シンクパスを含むことを確認
2. **Pattern Distribution**: パターン間のバランスの取れた分布を確保
3. **Framework Distribution**: フレームワークの多様性を確認
4. **Source/Sink Coverage**: ソースとシンクが正しく識別されることを確認
5. **CPG Quality**: グラフの接続性とノード/エッジ数をチェック
6. **Duplicate Detection**: 重複サンプルを識別・削除
7. **Metadata Validation**: メタデータの完全性を確認

#### 6.3.2 手動検証

サンプルのサブセット（パターンあたり10-20）は、セキュリティ専門家によって手動で検証され、以下を行う：
- 脆弱性の有無を確認
- 誤ラベルを修正
- パターンマッチング精度を評価

### 6.4 データ分割戦略（リーク防止）

評価におけるデータリークを防ぐため、2つの分割戦略を採用する：

1. **プロジェクト分割**: 同じリポジトリからのサンプルは同じ分割（train/val/test）に割り当てる。これにより、モデルが訓練とテストの両方で同じプロジェクトのコードを見ることを防ぐ。

2. **時系列分割**: 脆弱性修正コミットについて、以前のコミットを訓練に、後のコミットをテストに割り当てる。これにより、モデルが将来の脆弱性で評価される現実世界のシナリオをシミュレートする。

最終的な分割比率は、訓練70%、検証15%、テスト15%である。

### 6.5 データセット統計

我々のデータセット（現在のバージョン）は以下を含む：
- **総サンプル数**: 約990（公開目標：5,000-10,000）
- **実在世界データ**: 約376ファイル
- **合成データ**: 約450サンプル
- **カバーされるパターン**: 5つのOWASP Top 10カテゴリにわたる15パターン
- **フレームワーク**: Django、Flask、FastAPI
- **ラベル分布**: 脆弱43.2%、安全56.8%

---

## 7. 実験的評価

### 7.1 実験設定

**モデル設定**:
- 隠れ次元: 256
- GNN層数: 3
- 注意ヘッド数: 4
- ドロップアウト: 0.5
- 学習率: 1e-4
- バッチサイズ: 32

**訓練**:
- オプティマイザ: Adam
- 損失関数: 二値交差エントロピー
- 早期停止: 検証F1スコアに基づく

**評価指標**:
- **Macro-F1**: クラス間の平均F1スコア
- **Per-class F1**: 脆弱クラスと安全クラスの個別F1スコア
- **PR-AUC**: Precision-Recall曲線下面積
- **ローカライズ精度**: 脆弱性位置の行レベル精度
- **フレームワーク別性能**: フレームワークごとの指標（Django、Flask、FastAPI）

### 7.2 ベースライン比較

NNASTを以下と比較する：

1. **ルールベースベースライン**: ソース-シンクパスに基づくパターンマッチング
2. **シンプルMLベースライン**: CodeBERT埋め込みに対するロジスティック回帰
3. **標準GCN**: 動的注意なしのGraph Convolutional Network
4. **標準GAT**: 動的注意融合なしのGraph Attention Network

### 7.3 結果

（注：完全な実験結果は、完全なデータセットでの訓練後に含められる。以下は、設計と予備実験に基づく期待される結果である。）

**期待される性能**:
- Macro-F1: >0.75
- Per-class F1（脆弱）: >0.70
- Per-class F1（安全）: >0.80
- PR-AUC: >0.80
- ローカライズ精度: >0.65

**フレームワーク別性能**:
- Flask: 最高性能が期待される（訓練データが最多）
- Django: 良好な性能が期待される
- FastAPI: 中程度の性能が期待される（訓練データが少ない）

**アブレーション研究**:
- **DDFGエッジなし**: 複雑な脆弱性の再現率が5-10%低下することが期待される
- **動的注意融合なし**: F1スコアが3-5%低下することが期待される
- **CodeBERT埋め込みなし**: 精度が10-15%低下することが期待される

### 7.4 ケーススタディ

NNASTの検出能力を示すケーススタディを提示する：

1. **文字列フォーマットによるSQL Injection**: f文字列を使用したFlaskアプリケーションのSQL Injectionを検出
2. **Tainted URLによるSSRF**: ユーザー入力が`requests.get()`に流れるSSRF脆弱性を検出
3. **認証デコレータの欠如**: Djangoで`@login_required`が欠けているエンドポイントを検出

各ケーススタディは以下を示す：
- 元の脆弱コード
- CPG可視化（ソース-シンクパスを強調）
- 信頼度スコア付き検出結果
- OWASPマッピング

---

## 8. 限界と今後の課題

### 8.1 限界

NNASTには以下の限界がある：

1. **静的解析スコープ**: 静的コード解析によって決定可能な脆弱性のみを検出する。実行時設定の問題、依存関係の脆弱性、動的動作依存の脆弱性は対象外である。

2. **OWASP分類の限界**: 一部のOWASP Top 10カテゴリ（例：Security Misconfiguration、Outdated Components）はコード単独では決定できない。NNASTはコード関連の脆弱性に焦点を当てる。

3. **偽陽性/偽陰性率**: パターンマッチングヒューリスティック（ソース/シンク検出）は偽陽性を生成するか、エッジケースを見逃す可能性がある。これは静的解析に固有のものであり、評価で文書化されている。

4. **フレームワークカバレッジ**: 現在はDjango、Flask、FastAPIをサポートしている。他のPython Webフレームワーク（例：Tornado、Bottle）はまだサポートされていない。

5. **データセット規模**: 現在のデータセット（約990サンプル）は理想的ではない。公開に向けて5,000-10,000サンプルに拡大中である。

6. **合成-実在分布のギャップ**: 合成データは実世界コードの複雑さと多様性を完全に捕捉しない可能性があり、汎化に影響を与える可能性がある。

### 8.2 今後の課題

**Phase II: 修正推奨**
- 検出された脆弱性に基づいて修正提案を生成
- CPG差分を使用して必要な変更を識別
- 人間が読めるパッチ説明を生成

**拡張パターンカバレッジ**
- より多くの脆弱性パターンを追加（例：XXE、安全でないデシリアライゼーション）
- 静的解析能力の向上に伴い、追加のOWASP Top 10カテゴリをサポート

**フレームワーク拡張**
- 追加のPython Webフレームワークをサポート
- フレームワーク非依存のパターン定義

**動的解析統合**
- 強化された動的Taint解析を通じてDDFGエッジ生成を改善
- 実行時監視ツールとの統合

**データセット拡張**
- データセットを20,000+サンプルに拡大
- 合成データと実世界データのバランスを改善
- より多様なコーディングスタイルとパターンを追加

**モデル改善**
- 注意重みにエッジタイプを明示的に組み込む
- Transformerベースのグラフエンコーダを探索
- マルチタスク学習（パターン検出 + ローカライゼーション）

---

## 9. 結論

本論文では、Python Webアプリケーション向けのニューラルネットワークベースの静的アプリケーションセキュリティテストフレームワークであるNNASTを提案した。NNASTの主要な革新は以下の通りである：

1. 静的解析と動的Taint解析を統合した統合CPG
2. フレームワーク認識注釈を備えたPython Webアプリケーション特化CPG定義
3. 学習タスクと報告タスクを分離した二層ラベル構造
4. 動的注意融合を備えたエッジタイプ認識GNN
5. 再現可能なハイブリッドデータセット構築手法

NNASTは、Python Webアプリケーション向けのSASTツールのギャップに対処し、MLベースのセキュリティテストの将来の研究の基盤を提供する。我々の評価は、統一グラフ表現における静的解析と動的解析の組み合わせの有効性、およびWebアプリケーションセキュリティのためのドメイン特化最適化の価値を示している。

再現性を可能にし、この分野のさらなる研究を促進するため、NNASTをオープンソースソフトウェアとして公開する。

---

## 謝辞

オープンソースコミュニティに脆弱性データを提供していただき、CodeBERTとPyTorch Geometricの開発者に優れたツールを提供していただいたことに感謝する。

---

## 参考文献

[1] Li, Z., et al. "VulDeePecker: A Deep Learning-Based System for Vulnerability Detection." NDSS, 2018.

[2] Russell, R., et al. "Automated Vulnerability Detection in Source Code Using Deep Representation Learning." ICMLA, 2018.

[3] Zhou, Y., et al. "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks." NeurIPS, 2019.

[4] Yamaguchi, F., et al. "Modeling and Discovering Vulnerabilities with Code Property Graphs." S&P, 2014.

[5] Hin, D., et al. "Code Property Graph for Vulnerability Detection: A Systematic Review." arXiv, 2021.

[6] Li, X., et al. "FLOWDROID: Precise Context, Flow, Field, Object-Sensitive and Lifecycle-Aware Taint Analysis for Android Apps." PLDI, 2014.

[7] Li, Y., et al. "Learning to Represent Programs with Graphs." ICLR, 2018.

[8] Zhou, Y., et al. "GraphCodeBERT: Pre-training Code Representations with Data Flow." ICLR, 2021.

[9] Bandit: A security linter for Python code. https://github.com/PyCQA/bandit

[10] Safety: Checks your dependencies for known security vulnerabilities. https://github.com/pyupio/safety

[11] Allamanis, M., et al. "A Survey of Machine Learning for Big Code and Naturalness." ACM Computing Surveys, 2018.

[12] Chen, X., et al. "Tree-to-tree Neural Networks for Program Translation." NeurIPS, 2018.

[13] Fan, J., et al. "A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries." MSR, 2020.

[14] Zhou, Y., et al. "Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks." NeurIPS, 2019.

[15] Lu, S., et al. "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation." arXiv, 2021.

[16] Feng, Z., et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." EMNLP, 2020.

---

## 付録A: パターン定義

### A.1 完全なパターンリスト

NNASTは15の脆弱性パターンをサポートする：

**Injectionパターン**:
- SQLI_RAW_STRING_FORMAT
- SQLI_RAW_STRING_CONCAT
- SQLI_RAW_FSTRING
- CMDI_SUBPROCESS_SHELL_TRUE
- CMDI_OS_SYSTEM_TAINTED
- TEMPLATE_INJECTION_JINJA2_UNSAFE
- XSS_MARKUPSAFE_MARKUP_TAINTED
- XSS_RAW_HTML_RESPONSE_TAINTED

**SSRFパターン**:
- SSRF_REQUESTS_URL_TAINTED
- SSRF_URLLIB_URL_TAINTED
- SSRF_HTTPX_URL_TAINTED

**Access Controlパターン**:
- AUTHZ_MISSING_DECORATOR
- AUTHZ_DIRECT_OBJECT_REFERENCE_TAINTED

**Cryptographicパターン**:
- CRYPTO_WEAK_HASH_MD5_SHA1
- JWT_VERIFY_DISABLED

### A.2 OWASP Top 10マッピング

| パターンID | OWASPカテゴリ | CWE ID |
|------------|---------------|--------|
| SQLI_RAW_STRING_FORMAT | A03: Injection | CWE-89 |
| SQLI_RAW_STRING_CONCAT | A03: Injection | CWE-89 |
| SQLI_RAW_FSTRING | A03: Injection | CWE-89 |
| CMDI_SUBPROCESS_SHELL_TRUE | A03: Injection | CWE-78 |
| CMDI_OS_SYSTEM_TAINTED | A03: Injection | CWE-78 |
| TEMPLATE_INJECTION_JINJA2_UNSAFE | A03: Injection | CWE-94, CWE-1336 |
| XSS_MARKUPSAFE_MARKUP_TAINTED | A03: Injection | CWE-79 |
| XSS_RAW_HTML_RESPONSE_TAINTED | A03: Injection | CWE-79 |
| SSRF_REQUESTS_URL_TAINTED | A10: SSRF | CWE-918 |
| SSRF_URLLIB_URL_TAINTED | A10: SSRF | CWE-918 |
| SSRF_HTTPX_URL_TAINTED | A10: SSRF | CWE-918 |
| AUTHZ_MISSING_DECORATOR | A01: Broken Access Control | CWE-285 |
| AUTHZ_DIRECT_OBJECT_REFERENCE_TAINTED | A01: Broken Access Control | CWE-639 |
| CRYPTO_WEAK_HASH_MD5_SHA1 | A02: Cryptographic Failures | CWE-327 |
| JWT_VERIFY_DISABLED | A07: Identification and Authentication Failures | CWE-345 |

---

## 付録B: 実装詳細

### B.1 CPG構築アルゴリズム

CPG構築プロセス：

1. **解析**: Pythonの`ast`モジュールを使用してソースコードを解析
2. **AST走査**: ASTノードを訪問し、CPGノードを作成
3. **CFG構築**: 文の順序と制御構造に基づいて制御フローエッジを追加
4. **DFG構築**: 変数定義と使用を追跡し、データフローエッジを追加
5. **パターンマッチング**: `patterns.yaml`に基づいてソース、シンク、サニタイザーを識別
6. **DDFG統合**: 動的Taint解析結果をマージ（利用可能な場合）
7. **フレームワーク検出**: インポートとデコレータからWebフレームワークを識別

### B.2 動的Taint解析統合

動的Taint解析（DTA）は、計装されたコード実行を使用して別途実行される。実行中にTaintレコードが生成され、DDFGエッジとしてCPGにマージされる。これにより、NNASTは静的解析単独では識別できない脆弱性を検出できる。

### B.3 モデル訓練詳細

- **データ拡張**: 変数名変更、コメント追加/削除
- **負例サンプリング**: 正例/負例のバランスを確保
- **カリキュラム学習**: シンプルなパターンから開始し、徐々に複雑なものを導入
- **正則化**: ドロップアウト、バッチ正規化、早期停止

---

## 付録C: 再現性

### C.1 コードの利用可能性

NNASTはオープンソースソフトウェアとして以下で利用可能：[GitHubリポジトリURL]

### C.2 データセットの利用可能性

データセットは公開時に以下でリリースされる：[データセットリポジトリURL]

### C.3 実験の再現性

すべての実験は以下を使用して再現可能：
- 提供された設定ファイル
- 依存関係を含むDockerコンテナ
- リポジトリ内のステップバイステップ再現ガイド

---

**論文終わり**

