from typing import Any, Dict, List, Optional, Sequence, Tuple

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