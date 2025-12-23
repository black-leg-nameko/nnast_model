from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any

@dataclass
class CPGNode:
    id: int
    kind: str
    file: str
    span: Tuple[int, int, int, int]
    code: Optional[str] = None
    symbol: Optional[str] = None
    type_hint: Optional[str] = None
    flags: Optional[List[str]] = None
    attrs: Optional[Dict[str, str]] = None


@dataclass
class CPGEdge:
    src: int
    dst: int
    kind: str
    attrs: Optional[Dict[str, str]] = None


@dataclass
class CPGGraph:
    file: str
    nodes: List[CPGNode]
    edges: List[CPGEdge]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self):
        result = {
            "file": self.file,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges]
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
