#!/usr/bin/env python3
"""
統合コマンド: フォルダ内の全ファイルに対してGNN推論を実行し、
OpenAI APIでコード修正markdownを生成してローカルに保存する。

Usage:
    python ml/analyze_and_fix.py <directory> [options]
"""
import argparse
import json
import sys
import pathlib
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
from datetime import datetime
from tqdm import tqdm

# GNN推論関連
import torch
from ml.model import CPGTaintFlowModel
from ml.dataset import CPGGraphDataset
from ml.embed_codebert import CodeBERTEmbedder
from ml.inference import run_inference, generate_cpg_from_file, load_model, load_env_file

# LLMコード修正関連
from ml.code_fixer import LLMCodeFixer, FixSuggestion


def read_file_content(file_path: Path) -> Optional[str]:
    """ファイルの内容を読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return None


def generate_markdown_report(
    file_path: str,
    fix_suggestion: FixSuggestion,
    original_code: Optional[str] = None,
    confidence: Optional[float] = None,
    vulnerability_type: Optional[str] = None,
    repo_url: Optional[str] = None
) -> str:
    """
    Markdownレポートを生成する
    
    Args:
        file_path: ファイルパス
        fix_suggestion: LLMが生成した修正提案
        original_code: 元のコード
        confidence: 脆弱性検出の信頼度
        vulnerability_type: 脆弱性タイプ
        repo_url: リポジトリURL（オプション）
    
    Returns:
        Markdown形式のレポート文字列
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# セキュリティ脆弱性検出レポート

**生成日時**: {timestamp}

## ファイル情報

- **ファイルパス**: `{file_path}`
"""
    
    if repo_url:
        markdown += f"- **リポジトリ**: {repo_url}\n"
    
    if vulnerability_type:
        markdown += f"- **脆弱性タイプ**: {vulnerability_type}\n"
    
    if confidence is not None:
        markdown += f"- **検出信頼度**: {confidence:.2%}\n"
    
    markdown += "\n---\n\n"
    
    # 元のコード
    if original_code:
        markdown += f"""## 検出された脆弱なコード

```python
{original_code}
```

"""
    
    # 修正提案
    markdown += f"""## 修正提案

### 修正後のコード

```python
{fix_suggestion.fixed_code}
```

### 説明

{fix_suggestion.explanation}

"""
    
    if fix_suggestion.vulnerability_type:
        markdown += f"**脆弱性タイプ**: {fix_suggestion.vulnerability_type}\n\n"
    
    if fix_suggestion.confidence:
        markdown += f"**修正提案の信頼度**: {fix_suggestion.confidence:.2%}\n\n"
    
    markdown += "---\n\n"
    markdown += "*このレポートはNNASTモデルとOpenAI APIによって自動生成されました。*\n"
    
    return markdown


def save_markdown_report(
    markdown_content: str,
    file_path: str,
    output_dir: Path
) -> Path:
    """
    Markdownレポートをファイルに保存する
    
    Args:
        markdown_content: Markdownコンテンツ
        file_path: 元のファイルパス（ファイル名生成に使用）
        output_dir: 出力ディレクトリ
    
    Returns:
        保存されたファイルのパス
    """
    # ファイル名を生成（パスを安全なファイル名に変換）
    safe_name = Path(file_path).name.replace('.py', '') + '_vulnerability_report.md'
    # パスに含まれるディレクトリ構造も反映（必要に応じて）
    output_path = output_dir / safe_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return output_path


def process_directory(
    directory: Path,
    model_path: Optional[Path],
    output_dir: Path,
    code_fixer: LLMCodeFixer,
    device: torch.device,
    batch_size: int = 32,
    min_confidence: float = 0.7
) -> Dict[str, int]:
    """
    ディレクトリ内の全ファイルに対して推論→修正提案を実行
    
    Args:
        directory: 対象ディレクトリ
        model_path: モデルのパス
        output_dir: 出力ディレクトリ
        code_fixer: LLMコード修正器
        device: デバイス
        batch_size: バッチサイズ
        min_confidence: 最小信頼度閾値
    
    Returns:
        統計情報（処理済み、スキップ、エラー）
    """
    stats = {
        "total_files": 0,
        "processed": 0,
        "vulnerable": 0,
        "fixed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # 1. ディレクトリ内のPythonファイルをスキャン
    print(f"📁 ディレクトリをスキャン中: {directory}")
    python_files = list(directory.rglob("*.py"))
    stats["total_files"] = len(python_files)
    print(f"   見つかったPythonファイル: {len(python_files)}個")
    
    if len(python_files) == 0:
        print("⚠️  Pythonファイルが見つかりませんでした")
        return stats
    
    # 2. CPGグラフを生成
    print("\n🔍 CPGグラフを生成中...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
        tmp_jsonl = Path(tmp_file.name)
    
    graphs_generated = 0
    for py_file in tqdm(python_files, desc="CPG生成"):
        graph = generate_cpg_from_file(py_file)
        if graph:
            graph["file"] = str(py_file)
            with open(tmp_jsonl, "a", encoding='utf-8') as f:
                f.write(json.dumps(graph, ensure_ascii=False) + "\n")
            graphs_generated += 1
    
    print(f"   生成されたCPGグラフ: {graphs_generated}個")
    
    if graphs_generated == 0:
        print("❌ CPGグラフが生成されませんでした")
        return stats
    
    # 3. モデルをロード
    print("\n🤖 モデルをロード中...")
    if model_path is None:
        # デフォルトのモデルパスを探す
        default_paths = [
            Path("checkpoints_test_dynamic/best_model.pt"),
            Path("checkpoints/best_model.pt"),
        ]
        model_path = None
        for path in default_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            print("❌ モデルファイルが見つかりません")
            print("   --model オプションでモデルパスを指定してください")
            return stats
    
    if not model_path.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return stats
    
    try:
        model = load_model(model_path, device)
        print(f"   ✅ モデルをロードしました: {model_path}")
    except Exception as e:
        print(f"❌ モデルのロードに失敗しました: {e}")
        return stats
    
    # 4. CodeBERT embedderを初期化
    print("\n📝 CodeBERT embedderを初期化中...")
    embedder = CodeBERTEmbedder(device=str(device))
    
    # 5. データセットを作成
    print("\n📊 データセットを作成中...")
    dataset = CPGGraphDataset(
        graph_jsonl_path=str(tmp_jsonl),
        labels_jsonl_path=None,
        embedder=embedder,
        max_nodes=1000,
    )
    print(f"   ロードされたグラフ: {len(dataset)}個")
    
    if len(dataset) == 0:
        print("❌ データセットが空です")
        return stats
    
    # 6. GNN推論を実行
    print("\n🔬 GNN推論を実行中...")
    inference_results = run_inference(model, dataset, device, batch_size)
    stats["processed"] = len(inference_results)
    
    vulnerable_results = [r for r in inference_results if r.get("is_vulnerable", False)]
    stats["vulnerable"] = len(vulnerable_results)
    
    print(f"   処理済み: {len(inference_results)}個")
    print(f"   脆弱性検出: {len(vulnerable_results)}個")
    
    # 7. 脆弱性が検出されたファイルに対してLLMで修正提案を生成
    print("\n🔧 コード修正提案を生成中...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in tqdm(vulnerable_results, desc="修正提案生成"):
        file_path = result.get("file_path", "")
        confidence = result.get("confidence", 0.0)
        vulnerability_type = result.get("vulnerability_type")
        
        # 信頼度チェック
        if confidence < min_confidence:
            stats["skipped"] += 1
            continue
        
        try:
            # ファイル内容を読み込む
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"⚠️  ファイルが見つかりません: {file_path}")
                stats["skipped"] += 1
                continue
            
            original_code = read_file_content(file_path_obj)
            if not original_code:
                stats["skipped"] += 1
                continue
            
            # LLMで修正提案を生成
            fix_suggestion = code_fixer.generate_fix(
                vulnerable_code=original_code,
                file_path=file_path,
                vulnerability_type=vulnerability_type,
                context=f"GNNモデルが信頼度{confidence:.2%}で脆弱性を検出しました。"
            )
            
            if not fix_suggestion:
                print(f"⚠️  修正提案の生成に失敗: {file_path}")
                stats["errors"] += 1
                continue
            
            # Markdownレポートを生成
            markdown = generate_markdown_report(
                file_path=file_path,
                fix_suggestion=fix_suggestion,
                original_code=original_code,
                confidence=confidence,
                vulnerability_type=vulnerability_type,
                repo_url=result.get("repo_url")
            )
            
            # ファイルに保存
            saved_path = save_markdown_report(
                markdown_content=markdown,
                file_path=file_path,
                output_dir=output_dir
            )
            
            stats["fixed"] += 1
            print(f"   ✅ レポートを保存: {saved_path}")
            
        except Exception as e:
            print(f"❌ エラー ({file_path}): {e}")
            stats["errors"] += 1
    
    # 一時ファイルを削除
    try:
        tmp_jsonl.unlink()
    except:
        pass
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="フォルダ内の全ファイルに対してGNN推論→コード修正markdown生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # デフォルト設定で実行
  python ml/analyze_and_fix.py ./target_directory

  # モデルパスと出力ディレクトリを指定
  python ml/analyze_and_fix.py ./target_directory \\
    --model checkpoints/best_model.pt \\
    --output ./reports

  # 最小信頼度を変更
  python ml/analyze_and_fix.py ./target_directory \\
    --min-confidence 0.8
        """
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="分析対象のディレクトリパス"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="モデルチェックポイントのパス (デフォルト: checkpoints_test_dynamic/best_model.pt)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vulnerability_reports"),
        help="Markdownレポートの出力ディレクトリ (デフォルト: ./vulnerability_reports)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="推論時のバッチサイズ (デフォルト: 32)"
    )
    
    parser.add_argument(
        "--device",
        default=None,
        help="デバイス (cuda/cpu, 未指定時は自動検出)"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="修正提案を生成する最小信頼度 (デフォルト: 0.7)"
    )
    
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="LLMプロバイダー (デフォルト: openai)"
    )
    
    parser.add_argument(
        "--llm-model",
        default="gpt-4o",
        help="LLMモデル名 (デフォルト: gpt-4o)"
    )
    
    args = parser.parse_args()
    
    # 環境変数をロード
    load_env_file()
    
    # ディレクトリの存在確認
    if not args.directory.exists():
        print(f"❌ ディレクトリが見つかりません: {args.directory}")
        return 1
    
    if not args.directory.is_dir():
        print(f"❌ ディレクトリではありません: {args.directory}")
        return 1
    
    # デバイス設定
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"🖥️  使用デバイス: {device}")
    
    # LLMコード修正器を初期化
    print(f"\n🤖 LLMコード修正器を初期化中... (プロバイダー: {args.llm_provider}, モデル: {args.llm_model})")
    try:
        code_fixer = LLMCodeFixer(
            provider=args.llm_provider,
            model=args.llm_model
        )
    except Exception as e:
        print(f"❌ LLMコード修正器の初期化に失敗しました: {e}")
        print("   .envファイルにOPENAI_API_KEYまたはANTHROPIC_API_KEYが設定されているか確認してください")
        return 1
    
    # メイン処理
    print("\n" + "=" * 60)
    print("🚀 分析と修正提案の生成を開始します")
    print("=" * 60)
    
    stats = process_directory(
        directory=args.directory,
        model_path=args.model,
        output_dir=args.output,
        code_fixer=code_fixer,
        device=device,
        batch_size=args.batch_size,
        min_confidence=args.min_confidence
    )
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 処理結果サマリー")
    print("=" * 60)
    print(f"  総ファイル数: {stats['total_files']}")
    print(f"  処理済み: {stats['processed']}")
    print(f"  脆弱性検出: {stats['vulnerable']}")
    print(f"  修正提案生成: {stats['fixed']}")
    print(f"  スキップ: {stats['skipped']}")
    print(f"  エラー: {stats['errors']}")
    print(f"\n📁 レポート保存先: {args.output}")
    print("=" * 60)
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

