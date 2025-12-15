import ast


def get_span(node):
    # Python_version >= 3.8
    sl = getattr(node, "lineno", None)
    sc = getattr(node, "col_offset", None)
    el = getattr(node, "end_lineno", sl)
    ec = getattr(node, "end_col_offset", sc)
    if sl is None:
        return (0, 0, 0, 0)
    return (sl, sc or 0, el or sl, ec or 0)


def extract_code(source_lines, span):
    sl, sc, el, ec = span
    if sl == 0:
        return None
    if sl == el:
        return source_lines[sl - 1][sc:ec]
    return source_lines[sl - 1][sc:]
