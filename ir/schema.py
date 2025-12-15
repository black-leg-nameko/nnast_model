from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

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

    def to_dict(self):
        return {
            "file": self.file,
            "nodes": [asdict(n) for n in self.nodes]
            "edges": [asdict(e) for e in self.edges]
        }
