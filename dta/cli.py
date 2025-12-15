#!/usr/bin/env python3
"""
CLI tool for running code with taint tracking and outputting JSONL logs.

Usage:
    python -m dta.cli <script.py> [--output taint_log.jsonl] [--verbose]
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from dta.tracker import get_tracker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Python code with taint tracking enabled and output JSONL logs"
    )
    parser.add_argument("script", help="Python script to execute")
    parser.add_argument(
        "--output",
        "-o",
        default="taint_log.jsonl",
        help="Output JSONL file for taint records",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    tracker = get_tracker()
    tracker.enable()

    # Execute the script
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script not found: {args.script}", file=sys.stderr)
        return 1

    if not script_path.is_file():
        print(f"Error: Not a file: {args.script}", file=sys.stderr)
        return 1

    # Ensure output directory exists
    output_path = Path(args.output)
    if output_path.parent != Path(".") and not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: Cannot create output directory {output_path.parent}: {e}", file=sys.stderr)
            return 1

    try:
        # Read script content
        try:
            script_content = script_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"Error: Script is not UTF-8 encoded: {e}", file=sys.stderr)
            return 1

        # Import and execute the script
        # Note: This is simplified - in production, you'd want proper module loading
        exec(script_content, {"__name__": "__main__", "__file__": str(script_path)})

    except SyntaxError as e:
        print(f"Error: Syntax error in {args.script}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Error: Execution failed: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    # Write taint records to JSONL
    records = tracker.get_records()
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Error: Failed to write output file {args.output}: {e}", file=sys.stderr)
        return 1

    print(f"[OK] Recorded {len(records)} taint flow(s) to {args.output}", file=sys.stderr)
    if args.verbose and records:
        print(f"[DEBUG] Taint records:", file=sys.stderr)
        for i, record in enumerate(records, 1):
            print(f"  {i}. {record.get('source')} -> {record.get('sink')} ({record.get('meta', {}).get('sink_type', 'unknown')})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

