#!/usr/bin/env python3
"""
Convert taint logs from various DTA tools into NNAST canonical format.

Usage:
    python scripts/convert_taint_log.py <input.jsonl> <output.jsonl>

The script automatically detects and normalizes common formats:
- tainted format: {"taint_source": {...}, "taint_sink": {...}, "trace": [...]}
- Generic format: {"from": {...}, "to": {...}, "via": [...]}
- Already canonical format: {"source": {...}, "sink": {...}, "path": [...]}
"""
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import ir modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ir.taint_merge import normalize_taint_record


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                skipped += 1
                continue

            if not isinstance(raw, dict):
                print(f"[WARN] Line {line_num}: Expected dict, got {type(raw).__name__}", file=sys.stderr)
                skipped += 1
                continue

            normalized = normalize_taint_record(raw)
            if normalized is None:
                print(f"[WARN] Line {line_num}: Could not normalize record", file=sys.stderr)
                skipped += 1
                continue

            outfile.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            converted += 1

    print(f"[OK] Converted {converted} records, skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()

