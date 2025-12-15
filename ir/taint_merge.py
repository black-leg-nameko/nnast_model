from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

from ir.schema import CPGGraph, CPGEdge, CPGNode


Position = Dict[str, Any]  # expects at least {"file": str, "line": int, "col": Optional[int]}


def _span_covers(span: Tuple[int, int, int, int], line: int, col: Optional[int]) -> bool:
    sl, sc, el, ec = span
    if line < sl or line > el:
        return False
    if col is None:
        return True
    if line == sl and col < sc:
        return False
    if line == el and col > ec:
        return False
    return True


def _span_size(span: Tuple[int, int, int, int]) -> Tuple[int, int]:
    sl, sc, el, ec = span
    return (el - sl, ec - sc)


def _select_best_node(nodes: Sequence[CPGNode], line: int, col: Optional[int], role: str) -> Optional[CPGNode]:
    """
    Select the best-matching CPGNode for a given (line, col) and role.

    role: "source", "sink", or "path"
    """
    covered = [n for n in nodes if _span_covers(n.span, line, col)]
    if not covered:
        return None

    # Prioritize by kind depending on role.
    if role == "sink":
        primary_kinds = {"Call", "Attribute", "Subscript", "Await"}
        secondary_kinds = {"Name", "Literal", "ListComp", "DictComp", "Generator", "AsyncFor"}
    elif role == "source":
        primary_kinds = {"Name", "Attribute", "Subscript"}
        secondary_kinds = {"Call", "Literal", "ListComp", "DictComp", "Generator"}
    else:  # path
        primary_kinds = {"Call", "Attribute", "Subscript", "Await", "Name"}
        secondary_kinds = {"Literal", "ListComp", "DictComp", "Generator"}

    def kind_priority(node: CPGNode) -> int:
        if node.kind in primary_kinds:
            return 0
        if node.kind in secondary_kinds:
            return 1
        # Prefer non-generic statements over generic ones.
        if node.kind == "Stmt":
            return 3
        return 2

    # Sort by (priority, span_size)
    best = min(
        covered,
        key=lambda n: (kind_priority(n), _span_size(n.span)),
    )
    return best


def map_position_to_node_id(graph: CPGGraph, pos: Position, role: str) -> Optional[int]:
    """
    Map a single (file, line, col) position to a CPG node id within the graph.

    role: "source", "sink", or "path"
    """
    file = pos.get("file")
    line = pos.get("line")
    col = pos.get("col")
    if file is None or line is None:
        return None
    # Graph is per-file, but we still check for robustness.
    if graph.file != file:
        return None

    node = _select_best_node(graph.nodes, int(line), int(col) if col is not None else None, role)
    return node.id if node is not None else None


def map_taint_record(
    graph: CPGGraph, record: Dict[str, Any]
) -> Tuple[Optional[int], Optional[int], List[int], Dict[str, str]]:
    """
    Map a single taint record to (source_id, sink_id, path_ids, meta_attrs).

    The record is expected to have:
      - "source": {"file": str, "line": int, "col": Optional[int]}
      - "sink": {"file": str, "line": int, "col": Optional[int]}
      - optional "path": [ {...}, ... ]
      - optional "meta": {str: str}
    """
    src_pos = record.get("source") or {}
    sink_pos = record.get("sink") or {}
    path_pos = record.get("path") or []
    meta = record.get("meta") or {}

    src_id = map_position_to_node_id(graph, src_pos, role="source")
    sink_id = map_position_to_node_id(graph, sink_pos, role="sink")
    path_ids: List[int] = []
    for p in path_pos:
        nid = map_position_to_node_id(graph, p, role="path")
        if nid is not None:
            path_ids.append(nid)

    # attrs for DDFG edges: cast values to str for safety.
    meta_attrs = {str(k): str(v) for k, v in meta.items()}
    return src_id, sink_id, path_ids, meta_attrs


def add_ddfg_from_record(graph: CPGGraph, record: Dict[str, Any]) -> None:
    """
    Add DDFG edges to the given graph based on one taint record.

    The edges are added along the sequence:
        src -> path[0] -> ... -> path[-1] -> sink
    """
    src_id, sink_id, path_ids, meta_attrs = map_taint_record(graph, record)
    if src_id is None or sink_id is None:
        # Cannot reliably map, skip this record.
        return

    seq: List[int] = [src_id] + path_ids + [sink_id]
    if len(seq) < 2:
        return

    existing = {(e.src, e.dst, e.kind) for e in graph.edges}
    for a, b in zip(seq, seq[1:]):
        key = (a, b, "DDFG")
        if key in existing:
            continue
        graph.edges.append(CPGEdge(src=a, dst=b, kind="DDFG", attrs=meta_attrs or None))
        existing.add(key)


def add_ddfg_from_records(graph: CPGGraph, records: Iterable[Dict[str, Any]]) -> None:
    """Add DDFG edges for all taint records in an iterable."""
    for rec in records:
        add_ddfg_from_record(graph, rec)


def normalize_taint_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a raw taint record from various DTA tools into our canonical format.

    Supports common formats:
    - Our canonical format (pass-through)
    - tainted-like format with "taint_source" / "taint_sink" / "trace"
    - Generic format with "from" / "to" / "via"

    Returns None if the record cannot be normalized.
    """
    # Already in canonical format?
    if "source" in raw and "sink" in raw:
        src_pos = _extract_position(raw["source"])
        sink_pos = _extract_position(raw["sink"])
        if src_pos is not None and sink_pos is not None:
            return raw
        # If source/sink exist but aren't valid positions, fall through to try other formats

    result: Dict[str, Any] = {}

    # Try tainted-like format: {"taint_source": {...}, "taint_sink": {...}, "trace": [...]}
    if "taint_source" in raw and "taint_sink" in raw:
        src_raw = raw["taint_source"]
        sink_raw = raw["taint_sink"]
        result["source"] = _extract_position(src_raw)
        result["sink"] = _extract_position(sink_raw)
        if "trace" in raw:
            result["path"] = [_extract_position(p) for p in raw["trace"] if _extract_position(p)]
        if "taint_type" in raw:
            result.setdefault("meta", {})["taint_kind"] = str(raw["taint_type"])
        if "sink_type" in raw:
            result.setdefault("meta", {})["sink_type"] = str(raw["sink_type"])
        return result if result.get("source") and result.get("sink") else None

    # Try generic format: {"from": {...}, "to": {...}, "via": [...]}
    if "from" in raw and "to" in raw:
        result["source"] = _extract_position(raw["from"])
        result["sink"] = _extract_position(raw["to"])
        if "via" in raw:
            result["path"] = [_extract_position(p) for p in raw["via"] if _extract_position(p)]
        # Copy any other keys as meta
        meta = {k: str(v) for k, v in raw.items() if k not in ("from", "to", "via")}
        if meta:
            result["meta"] = meta
        return result if result.get("source") and result.get("sink") else None

    return None


def _extract_position(obj: Any) -> Optional[Position]:
    """Extract a Position dict from various formats."""
    if not isinstance(obj, dict):
        return None

    # Already in {"file": ..., "line": ..., "col": ...} format?
    if "file" in obj and "line" in obj:
        pos: Position = {"file": str(obj["file"]), "line": int(obj["line"])}
        if "col" in obj:
            pos["col"] = int(obj["col"])
        return pos

    # Try {"filename": ..., "lineno": ..., "column": ...}
    if "filename" in obj and "lineno" in obj:
        pos = {"file": str(obj["filename"]), "line": int(obj["lineno"])}
        if "column" in obj:
            pos["col"] = int(obj["column"])
        return pos

    # Try {"path": ..., "line": ..., "column": ...}
    if "path" in obj and "line" in obj:
        pos = {"file": str(obj["path"]), "line": int(obj["line"])}
        if "column" in obj:
            pos["col"] = int(obj["column"])
        return pos

    return None