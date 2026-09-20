"""Convert the portable genealogy JSON to Cytoscape and GraphML formats."""

import argparse
import json
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=ROOT / "final_genealogy_graph.json")
parser.add_argument("--output-prefix", type=Path, default=ROOT / "final_genealogy")
args = parser.parse_args()
source = json.loads(args.input.read_text())
nodes = {node["id"]: node for node in source["nodes"]}

# Validate direction and references before exporting to tools that may silently
# drop invalid links or infer an undirected graph.
assert len(nodes) == len(source["nodes"])
assert all(link["source"] in nodes and link["target"] in nodes for link in source["links"])
assert all(nodes[link["source"]]["created_utc"] < nodes[link["target"]]["created_utc"]
           for link in source["links"])

cytoscape = {"data": {"name": "ZeitgeistLM temporal similarity graph"},
             "elements": {"nodes": [{"data": node} for node in source["nodes"]],
                          "edges": [{"data": {
                              "id": f"{link['source']}->{link['target']}", **link}}
                                    for link in source["links"]]}}
(args.output_prefix.with_name(args.output_prefix.name + "_cytoscape.json")).write_text(
    json.dumps(cytoscape, ensure_ascii=False, indent=2) + "\n")

graph = nx.DiGraph()
for node in source["nodes"]:
    graph.add_node(node["id"], **{key: value for key, value in node.items() if key != "id"})
for link in source["links"]:
    graph.add_edge(link["source"], link["target"],
                   **{key: value for key, value in link.items()
                      if key not in {"source", "target"}})
assert nx.is_directed_acyclic_graph(graph)
nx.write_graphml(graph, args.output_prefix.with_suffix(".graphml"))
print(f"Exported {graph.number_of_nodes()} nodes and {graph.number_of_edges()} directed links")
