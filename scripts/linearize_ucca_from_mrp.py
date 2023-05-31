import json
from Graph import Graph
import itertools
from Node import Node, TerminalNode
from typing import Dict, Tuple
from pretty_print import pretty_print
from visualize_from_graph import draw_graph
from collections import defaultdict

class EdgeTags:
    """Layer 1 Edge tags."""
    Unanalyzable = 'UNA'
    Uncertain = 'UNC'
    ParallelScene = 'H'
    Participant = 'A'
    Process = 'P'
    State = 'S'
    Adverbial = 'D'
    Ground = 'G'
    Center = 'C'
    Elaborator = 'E'
    Function = 'F'
    Connector = 'N'
    Relator = 'R'
    Time = 'T'
    Quantifier = 'Q'
    Linker = 'L'
    Punctuation = 'U'
    LinkRelation = 'LR'
    LinkArgument = 'LA'
    Terminal = 'Terminal'
    __init__ = None


class UccaLinearizer:

    def __init__(self, graph):
        self.graph = graph
        self.tags_count = []
        self.visited = {"1.1": "root_0"}  # TODO : empty
        self.labeldict = {}                # TODO : remove
        self.visited_parents = []
        self.output_str = ""
        self.dfs_linearize(self.graph.top) # traverse from top node

    def get_edge_label(self, edge):

        edge_label = edge.label
        if "remote" in edge.properties:
            edge_label += "*"

        return edge_label

    def get_node_name(self, edge_label):

        edge_label = edge_label.replace('*', '')
        edge_label = edge_label.replace('?', '')

        label_count = self.tags_count.count(edge_label)
        self.tags_count.append(edge_label)

        return "{}_{}".format(edge_label, label_count)

    def add_to_visited(self, node_id, label_id):
        self.visited[node_id] = label_id

    def dfs_linearize(self, node):

        def start(node): # sort by children's terminal span
            if isinstance(node, TerminalNode):
                return [node.anchor[0]]
            else:
                return sorted([terminal_node.anchor[0] for terminal_node in node.terminal_spans])

        sorted_children = sorted(node.children, key=start)

        for parent_node, child_node in zip(itertools.repeat(node), sorted_children):

            # If not terminal node, do this recursively
            if not isinstance(child_node, TerminalNode):
                edge_id = "{}->{}".format(parent_node.id, child_node.id)
                curr_edge = self.graph.edges[edge_id]
                edge_label = self.get_edge_label(curr_edge)

                # In case of re-entrancy, replace the rest of node
                if child_node.id in self.visited:
                    self.output_str += "{} [ <{}> ".format(edge_label, self.visited[child_node.id])
                    # self.dfs_linearize(child_node)

                else: # when node is not visited
                    node_name = self.get_node_name(edge_label)
                    self.add_to_visited(child_node.id, node_name)
                    self.output_str += "{} [ <{}> ".format(edge_label, node_name)
                    self.dfs_linearize(child_node)

            # If terminal node, add this to the string
            else:
                self.output_str += "Z [ {} ".format(child_node.token)

            self.output_str += "] "


if __name__ == '__main__':

    # [0] Read the whole file
    # file = "/home/getalp/kangje/Projects/PycharmProjects/SGL/data/UCCA/mrp/2019/training/ucca/fixed.ewt.mrp"
    # with open(file) as f:
    #     line = f.readlines()
    #     for l in line:
    #         ex = json.loads(l)
    #         graph = Graph()
    #         graph.create_from_json(ex)
    #         linearized = UccaLinearizer(graph)
    #         pretty_print(linearized.output_str)
    #         draw_graph(graph)

    ## [1] Read one example
    l = '{"id": "103610", "flavor": 1, "framework": "ucca", "version": 1.0, "time": "2019-06-25", "input": "I will draw you a railing to put around your flower.", "tops": [13], "nodes": [{"id": 0, "anchors": [{"from": 0, "to": 1}]}, {"id": 1, "anchors": [{"from": 2, "to": 6}]}, {"id": 2, "anchors": [{"from": 7, "to": 11}]}, {"id": 3, "anchors": [{"from": 12, "to": 15}]}, {"id": 4, "anchors": [{"from": 16, "to": 17}]}, {"id": 5, "anchors": [{"from": 18, "to": 25}]}, {"id": 6, "anchors": [{"from": 26, "to": 28}]}, {"id": 7, "anchors": [{"from": 29, "to": 32}]}, {"id": 8, "anchors": [{"from": 33, "to": 39}]}, {"id": 9, "anchors": [{"from": 40, "to": 44}]}, {"id": 10, "anchors": [{"from": 45, "to": 51}]}, {"id": 11, "anchors": [{"from": 51, "to": 52}]}, {"id": 12}, {"id": 13}, {"id": 14}, {"id": 15}, {"id": 16}, {"id": 17}], "edges": [{"source": 14, "target": 5, "label": "C"}, {"source": 17, "target": 9, "label": "S"}, {"source": 16, "target": 17, "label": "E"}, {"source": 16, "target": 8, "label": "R"}, {"source": 13, "target": 6, "label": "L"}, {"source": 15, "target": 5, "label": "A", "attributes": ["remote"], "values": [true]}, {"source": 17, "target": 9, "label": "A"}, {"source": 15, "target": 3, "label": "A", "attributes": ["remote"], "values": [true]}, {"source": 16, "target": 11, "label": "U"}, {"source": 13, "target": 12, "label": "H"}, {"source": 12, "target": 0, "label": "A"}, {"source": 14, "target": 4, "label": "F"}, {"source": 15, "target": 16, "label": "A"}, {"source": 12, "target": 1, "label": "F"}, {"source": 16, "target": 10, "label": "C"}, {"source": 12, "target": 2, "label": "P"}, {"source": 12, "target": 3, "label": "A"}, {"source": 12, "target": 14, "label": "A"}, {"source": 17, "target": 10, "label": "A", "attributes": ["remote"], "values": [true]}, {"source": 13, "target": 15, "label": "H"}, {"source": 15, "target": 7, "label": "P"}]}'
    ex = json.loads(l)
    graph = Graph()
    graph.create_from_json(ex)
    linearized = UccaLinearizer(graph)
    pretty_print(linearized.output_str)
    draw_graph(graph)



