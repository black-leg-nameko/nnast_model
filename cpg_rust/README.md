# CPG Rust Implementation

High-performance CPG (Code Property Graph) builder in Rust with Python bindings.

## Status

🚧 **Work in Progress** - Phase 2 (AST parsing) implementation in progress.

## Building

### Prerequisites

- Rust 1.70+ (stable)
- Python 3.8+
- maturin (for building Python extension)

### Install maturin

```bash
pip install maturin
```

### Build

**重要**: PyO3拡張モジュールは`cargo build`ではビルドできません。必ず`maturin`を使用してください。

#### 方法1: 仮想環境を使用（推奨）

```bash
# プロジェクトルートで仮想環境を作成・アクティベート
cd /Users/ryutokitajima/works/nnast_model
python3 -m venv .venv
source .venv/bin/activate

# maturinをインストール
pip install maturin

# CPG Rustディレクトリに移動してビルド
cd cpg_rust
maturin develop  # 開発モード
# または
maturin build --release  # リリースビルド
```

#### 方法2: 仮想環境なし（wheelファイルをビルド）

```bash
cd cpg_rust
maturin build --release
pip install target/wheels/cpg_rust-*.whl
```

詳細は [SETUP.md](SETUP.md) を参照してください。

### Test

```bash
# Rust tests
cargo test

# Python tests (after building with maturin)
python -c "import cpg_rust; print('Module loaded successfully!')"
```

## Usage

```python
import cpg_rust

# Build CPG from Python source
graph = cpg_rust.build_cpg("example.py", source_code)
print(graph)
```

## Project Structure

- `src/lib.rs` - Main module and Python bindings
- `src/schema.rs` - CPG data structures
- `src/builder.rs` - CPG builder logic
- `src/cfg.rs` - Control Flow Graph construction
- `src/dfg.rs` - Data Flow Graph construction
- `src/scope.rs` - Scope management
- `src/ast_parser.rs` - AST parsing utilities
- `src/utils.rs` - Utility functions

## Development Plan

See [RUST_MIGRATION_PLAN.md](../RUST_MIGRATION_PLAN.md) for the complete implementation plan.

## Troubleshooting

### Link errors with `cargo build`

PyO3拡張モジュールは`cargo build`ではビルドできません。必ず`maturin`を使用してください。

```bash
# ❌ これは動作しません
cargo build

# ✅ これを使用してください
maturin develop
```

詳細は [LINKING_FIX.md](LINKING_FIX.md) を参照してください。

## License

MIT OR Apache-2.0
