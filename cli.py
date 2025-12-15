# nnast/cli.py
import argparse
import pathlib

from nnast.cpg.parse import parse_source
from nnast.cpg.build_ast import ASTCPGBuilder
from nnast.ir.schema import CPGGraph
from nnast.ir.io import write_graph_jsonl


def iter_py_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="target file or directory")
    parser.add_argument("--out", default="graphs.jsonl")
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    out = args.out

    files = [root] if root.is_file() else list(iter_py_files(root))

    for py in files:
        try:
            src = py.read_text(encoding="utf-8")
            tree = parse_source(src)
            builder = ASTCPGBuilder(str(py), src)
            builder.visit(tree)

            graph = CPGGraph(
                file=str(py),
                nodes=builder.nodes,
                edges=builder.edges
            )
            write_graph_jsonl(out, graph.to_dict())
            print(f"[OK] {py}")

        except Exception as e:
            print(f"[FAIL] {py}: {e}")


if __name__ == "__main__":
    main()
