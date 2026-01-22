# OWASPマッピングモジュール

## 概要

OWASPマッピングモジュールは、設計書v2の「二層ラベル構造」を実装します：
- **内部ラベル**: Pattern ID（学習用）
- **外部ラベル**: OWASP Top 10 カテゴリ（報告用）

## 機能

1. **Pattern ID → OWASP/CWE マッピング**
2. **推論結果へのOWASP/CWE情報付与**
3. **設計書5.3の出力フォーマット対応**

## 使用方法

### 基本的な使用

```python
from ml.owasp_mapper import OWASPMapper

mapper = OWASPMapper()

# Pattern IDからOWASP/CWE情報を取得
pattern_id = "SQLI_RAW_STRING_FORMAT"
owasp = mapper.get_owasp(pattern_id)  # "A03: Injection"
cwe = mapper.get_primary_cwe(pattern_id)  # "CWE-89"
cwe_list = mapper.get_cwe(pattern_id)  # ["CWE-89"]
```

### Pattern IDの推論

```python
# Source/Sink情報からPattern IDを推論
pattern_id = mapper.infer_pattern_id(
    source_id="SRC_FLASK_REQUEST",
    sink_id="SINK_DBAPI_EXECUTE",
    sink_kind="sql_exec",
    node_code="cursor.execute(query)"
)
```

### 設計書5.3フォーマットでの出力

```python
result = mapper.format_result(
    pattern_id="SSRF_REQUESTS_URL_TAINTED",
    confidence=0.94,
    file_path="views.py",
    lines=[42, 48]
)

# 出力:
# {
#   "pattern_id": "SSRF_REQUESTS_URL_TAINTED",
#   "cwe_id": "CWE-918",
#   "owasp": "A10: SSRF",
#   "confidence": 0.94,
#   "location": {
#     "file": "views.py",
#     "lines": [42, 48]
#   }
# }
```

## 推論スクリプトでの使用

### 標準モード（OWASP/CWE情報を追加）

```bash
python -m ml.inference input.py --model checkpoints/best_model.pt --output results.json
```

出力例:
```json
{
  "file_path": "app.py",
  "is_vulnerable": true,
  "confidence": 0.94,
  "pattern_id": "SQLI_RAW_STRING_FORMAT",
  "owasp": "A03: Injection",
  "cwe_id": "CWE-89",
  "cwe": ["CWE-89"],
  "taint_flow": {
    ...
  }
}
```

### OWASPフォーマットモード（設計書5.3形式）

```bash
python -m ml.inference input.py --model checkpoints/best_model.pt --output results.json --owasp-format
```

出力例:
```json
{
  "file_path": "app.py",
  "is_vulnerable": true,
  "confidence": 0.94,
  "pattern_id": "SQLI_RAW_STRING_FORMAT",
  "owasp_mapping": {
    "pattern_id": "SQLI_RAW_STRING_FORMAT",
    "cwe_id": "CWE-89",
    "owasp": "A03: Injection",
    "confidence": 0.94,
    "location": {
      "file": "app.py",
      "lines": [42, 48]
    }
  }
}
```

## 実装されたパターン

現在、15個のパターンが実装されています：

### Injection (A03)
- SQLI_RAW_STRING_FORMAT
- SQLI_RAW_STRING_CONCAT
- SQLI_RAW_FSTRING
- CMDI_SUBPROCESS_SHELL_TRUE
- CMDI_OS_SYSTEM_TAINTED
- TEMPLATE_INJECTION_JINJA2_UNSAFE

### XSS (A03)
- XSS_MARKUPSAFE_MARKUP_TAINTED
- XSS_RAW_HTML_RESPONSE_TAINTED

### SSRF (A10)
- SSRF_REQUESTS_URL_TAINTED
- SSRF_URLLIB_URL_TAINTED
- SSRF_HTTPX_URL_TAINTED

### Broken Access Control (A01)
- AUTHZ_MISSING_DECORATOR
- AUTHZ_DIRECT_OBJECT_REFERENCE_TAINTED

### Cryptographic Failures (A02)
- CRYPTO_WEAK_HASH_MD5_SHA1

### Identification and Authentication Failures (A07)
- JWT_VERIFY_DISABLED

## Pattern ID推論の仕組み

`infer_pattern_id`メソッドは、以下の情報からPattern IDを推論します：

1. **Source ID**: CPGノードの`source_id`属性
2. **Sink ID**: CPGノードの`sink_id`属性
3. **Sink Kind**: CPGノードの`sink_kind`属性
4. **Node Code**: ノードのコード文字列（追加のヒューリスティック）

スコアリングシステム：
- Source ID一致: +10点
- Sink ID一致: +10点
- Sink Kind一致: +8点
- コードベースのヒューリスティック: +5点

最高スコアのパターンが選択されます。

## 注意事項

- Pattern IDの推論はヒューリスティックベースのため、100%正確ではありません
- より正確な推論には、モデルによるパターン分類の実装が推奨されます
- 現在の実装は、DTA（動的Taint解析）情報がある場合に最も正確に動作します

