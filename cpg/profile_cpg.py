#!/usr/bin/env python3
"""
Performance profiling script for CPG generation

Usage:
    python -m cpg.profile_cpg <python_file> [--output <output_file>]
"""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path
from typing import Optional

from cpg.parse import parse_source
from cpg.build_ast import ASTCPGBuilder
from ir.schema import CPGGraph


def profile_cpg_generation(file_path: Path, output_file: Optional[Path] = None):
    """Profile CPG generation and display results"""
    
    print(f"Profiling target: {file_path}")
    
    # Read file
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error: Failed to read file: {e}", file=sys.stderr)
        return 1
    
    file_size = len(source)
    line_count = len(source.splitlines())
    print(f"File size: {file_size:,} bytes, Lines: {line_count:,}")
    
    # Setup profiler
    profiler = cProfile.Profile()
    
    # Measure timing
    start_time = time.perf_counter()
    
    profiler.enable()
    try:
        # Generate CPG
        tree = parse_source(source)
        builder = ASTCPGBuilder(str(file_path), source)
        builder.visit(tree)
        
        graph = CPGGraph(
            file=str(file_path),
            nodes=builder.nodes,
            edges=builder.edges,
        )
    except Exception as e:
        print(f"Error: CPG generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        profiler.disable()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    print(f"Generated nodes: {len(graph.nodes):,}")
    print(f"Generated edges: {len(graph.edges):,}")
    print(f"Nodes/second: {len(graph.nodes) / elapsed_time:.2f}")
    print(f"Edges/second: {len(graph.edges) / elapsed_time:.2f}")
    print(f"{'='*60}\n")
    
    # Display profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("Top 20 by cumulative time:")
    print("-" * 60)
    stats.print_stats(20)
    
    print("\nTop 20 by call count:")
    print("-" * 60)
    stats.sort_stats('ncalls')
    stats.print_stats(20)
    
    # Output to file
    if output_file:
        stats.dump_stats(str(output_file))
        print(f"\nProfiling results saved to: {output_file}")
        print("  To view: python -m pstats", output_file)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Performance profiling for CPG generation"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Python file to profile"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for profiling results (.prof format)"
    )
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    
    return profile_cpg_generation(args.file, args.output)


if __name__ == "__main__":
    sys.exit(main())
