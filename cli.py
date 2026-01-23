import argparse
import pathlib
import sys
from typing import Optional

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder
from cpg.pattern_matcher import PatternMatcher
from ir.schema import CPGGraph
from ir.io import write_graph_jsonl, iter_taint_records
from ir.taint_merge import add_ddfg_from_records


def iter_py_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        yield p


def log(message: str, level: str = "INFO", verbose: bool = False, quiet: bool = False):
    """Simple logging function."""
    if quiet and level != "ERROR":
        return
    if not verbose and level == "DEBUG":
        return
    prefix = f"[{level}]" if level != "INFO" else "[OK]"
    print(f"{prefix} {message}", file=sys.stderr if level == "ERROR" else sys.stdout)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate CPG (Code Property Graph) from Python code, optionally merging DDFG from taint logs."
    )
    parser.add_argument("path", help="Target Python file or directory to analyze")
    parser.add_argument("--out", default="graphs.jsonl", help="Output JSONL path for CPG/DDFG graphs")
    parser.add_argument(
        "--taint-log",
        help=(
            "Optional path to a DTA/taint JSONL log. Each line should be a JSON object with "
            '"source", "sink", optional "path" and "meta" fields as expected by ir.taint_merge.'
        ),
    )
    parser.add_argument(
        "--patterns",
        help="Path to patterns.yaml file (default: patterns.yaml in project root)",
        default=None,
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable pattern matching (skip source/sink/sanitizer detection)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output (show debug messages)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.path)
    if not root.exists():
        log(f"Error: Path does not exist: {root}", level="ERROR")
        return 1

    out = pathlib.Path(args.out)
    taint_log: Optional[str] = args.taint_log

    # Initialize pattern matcher (if not disabled)
    pattern_matcher = None
    if not args.no_patterns:
        try:
            patterns_path = pathlib.Path(args.patterns) if args.patterns else None
            pattern_matcher = PatternMatcher(patterns_path)
            log(f"Loaded pattern matcher from {pattern_matcher.patterns_yaml_path}", verbose=args.verbose)
        except FileNotFoundError as e:
            log(f"Warning: {e}. Pattern matching disabled.", level="ERROR" if args.verbose else "INFO")
        except Exception as e:
            log(f"Warning: Failed to load pattern matcher: {e}. Pattern matching disabled.", level="ERROR")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Ensure output directory exists
    if out.parent != pathlib.Path(".") and not out.parent.exists():
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            log(f"Created output directory: {out.parent}", verbose=args.verbose)
        except OSError as e:
            log(f"Error: Cannot create output directory {out.parent}: {e}", level="ERROR")
            return 1

    files = [root] if root.is_file() else list(iter_py_files(root))
    if not files:
        log(f"Warning: No Python files found in {root}", level="ERROR")
        return 1

    log(f"Found {len(files)} Python file(s) to process", verbose=args.verbose)

    # Pre-load taint records once if a log is provided.
    taint_records = []
    if taint_log:
        taint_log_path = pathlib.Path(taint_log)
        if not taint_log_path.exists():
            log(f"Error: Taint log file not found: {taint_log}", level="ERROR")
            return 1
        try:
            taint_records = list(iter_taint_records(taint_log))
            log(f"Loaded {len(taint_records)} taint record(s) from {taint_log}", verbose=args.verbose)
        except Exception as e:
            log(f"Error: Failed to read taint log {taint_log}: {e}", level="ERROR")
            return 1

    success_count = 0
    error_count = 0

    for py in files:
        try:
            src = py.read_text(encoding="utf-8")
            tree = parse_source(src)
            builder = ASTCPGBuilder(str(py), src, pattern_matcher=pattern_matcher)
            builder.visit(tree)

            # Add framework metadata to graph
            graph_metadata = {}
            if builder.pattern_matcher and builder._frameworks:
                graph_metadata["frameworks"] = list(builder._frameworks)
            
            graph = CPGGraph(
                file=str(py),
                nodes=builder.nodes,
                edges=builder.edges,
                metadata=graph_metadata if graph_metadata else None,
            )

            # Optionally merge dynamic data-flow (DDFG) edges from taint records.
            if taint_records:
                add_ddfg_from_records(graph, taint_records)
                ddfg_count = len([e for e in graph.edges if e.kind == "DDFG"])
                log(f"Merged {ddfg_count} DDFG edge(s) into {py}", verbose=args.verbose)

            write_graph_jsonl(str(out), graph.to_dict())
            log(f"{py}", verbose=args.verbose)
            success_count += 1

        except SyntaxError as e:
            log(f"{py}: Syntax error: {e}", level="ERROR")
            error_count += 1
        except UnicodeDecodeError as e:
            log(f"{py}: Encoding error (not UTF-8): {e}", level="ERROR")
            error_count += 1
        except Exception as e:
            log(f"{py}: {type(e).__name__}: {e}", level="ERROR")
            if args.verbose:
                import traceback
                traceback.print_exc()
            error_count += 1

    if not args.quiet:
        log(f"Processed {success_count} file(s), {error_count} error(s)", verbose=True)
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
