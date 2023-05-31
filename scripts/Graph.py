from typing import List, Dict
from Node import Node, TerminalNode, Union
from Edge import Edge
from ucca_graph_utils import parse_tree, set_terminal_span

class Graph:
    def __init__(self):

        self.nodes: Dict[int: Union[Node, TerminalNode]] = {}
        self.edges: Dict[str]
        self.id: str = ""
        self.input: str = ""
        self.top = None # top Node

    def create_from_json(self, json_str):
        self.set_nodes_n_edges(json_str)
        self.set_graph_id(json_str)
        self.set_input(json_str)
        self.set_top(json_str)

    def set_graph_id(self, json_to_parse):
        self.id = json_to_parse["id"]

    def set_input(self, json_to_parse):
        self.input = json_to_parse["input"]

    def set_top(self, json_to_parse):
        assert len(json_to_parse["tops"]) == 1
        top_node_id = json_to_parse["tops"][0]
        self.top = self.nodes[top_node_id]

    def set_nodes_n_edges(self, json_to_parse):
        nodes, edges = parse_tree(json_to_parse)
        self.nodes = nodes
        self.edges = edges
        self.nodes = set_terminal_span(self.nodes, self.edges)



