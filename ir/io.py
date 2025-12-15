import json

def write_graph_json1(path, graph_dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(graph_dict, ensure_ascii=False) + "\n")
