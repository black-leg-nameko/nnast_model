import ast
from typing import Optional


def get_span(node, total_lines: int, last_line_len: int) -> Optional[tuple]:
    """Return (sl, sc, el, ec) span or None when node has no lineno."""
    # Python_version >= 3.8
    if isinstance(node, ast.Module):
        # Cover the full file for the module node
        end_col = last_line_len if total_lines > 0 else 0
        return (1, 0, max(total_lines, 1), end_col)

    sl = getattr(node, "lineno", None)
    sc = getattr(node, "col_offset", None)
    el = getattr(node, "end_lineno", sl)
    ec = getattr(node, "end_col_offset", sc)
    if sl is None:
        return None
    return (sl, sc or 0, el or sl, ec or 0)


def extract_code(source_lines, span):
    if span is None:
        return None
    sl, sc, el, ec = span
    if sl <= 0 or sl > len(source_lines):
        return None
    if sl == el:
        return source_lines[sl - 1][sc:ec]
    # Multi-line snippet: take from start line start col to its end for now
    return source_lines[sl - 1][sc:]


def parse_source(source: str) -> ast.AST:
    """Parse Python source into an AST."""
    return ast.parse(source)
