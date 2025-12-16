import json
import pathlib
from typing import Dict, Iterable, Iterator, Any, Optional

from ir.taint_merge import normalize_taint_record


def write_graph_jsonl(path: str, graph_dict: Dict[str, Any]) -> None:
    """Append a single CPG graph (as a dict) to a JSONL file.
    
    Raises:
        OSError: If the file cannot be written (e.g., permission denied).
    """
    path_obj = pathlib.Path(path)
    # Ensure parent directory exists
    if path_obj.parent != pathlib.Path(".") and not path_obj.parent.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(graph_dict, ensure_ascii=False) + "\n")
    except OSError as e:
        raise OSError(f"Failed to write to {path}: {e}") from e


def iter_taint_records(path: str, normalize: bool = True) -> Iterator[Dict[str, Any]]:
    """Stream DTA/taint records from a JSONL file.

    Expected format per line (our canonical NNAST taint record schema):

        {
          "source": {"file": "...", "line": 10, "col": 5},
          "sink":   {"file": "...", "line": 42, "col": 17},
          "path": [
            {"file": "...", "line": 20, "col": 3},
            ...
          ],
          "meta": {
            "taint_kind": "...",
            "sink_type": "...",
            "...": "..."
          }
        }

    If normalize=True (default), also attempts to convert common DTA tool formats
    (e.g., tainted's "taint_source"/"taint_sink" format) into the canonical format.

    Extra keys are ignored. Lines that fail to parse as JSON are skipped.
    Records that cannot be normalized (if normalize=True) are also skipped.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read (e.g., permission denied).
    """
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Taint log file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip malformed JSON lines silently (as documented)
                    continue
                if not isinstance(record, dict):
                    continue

                if normalize:
                    normalized = normalize_taint_record(record)
                    if normalized is not None:
                        yield normalized
                else:
                    yield record
    except OSError as e:
        raise OSError(f"Failed to read taint log {path}: {e}") from e
