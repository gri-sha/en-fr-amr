# reformat ucca for mtool evaluation
from delinearize_ucca import UccaParser
from itertools import chain
import uuid
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from datetime import datetime


class UccaMRPConverter:
    """
    Convrt Linearized UCCA to a mrp format for evaluation
    IMPORTANT: Terminal nodes should be 'Z' and the root node should be 'root_O'
    """

    def __init__(self, ucca, src_text, graph_id):
        self.ucca = ucca
        self.src_text_temp = src_text
        self.src_text = src_text
        self.visited_nodes = {}
        self.nodes = []
        self.edges = []
        self.reformatted = []
        self.root_id = None
        self.graph_id = graph_id

    def get_triplets(self) -> List[Tuple]:

        output = []
        triplets = chain(*self.ucca.values())
        # reorder according to the length of the child (more for terminal) nodes
        triplets = sorted(triplets, reverse=True, key=lambda node: len(node[2]))


        for triplet in triplets:
            output.append(tuple(triplet))

        return output


    @staticmethod
    def reverse_dict(dictionary) -> Dict:
        # return a reversed dict (value: key) only with unique values
        values = list(dictionary.values())
        reversed_dict = {}

        for k, v in dictionary.items():
            if values.count(v) == 1:
                reversed_dict[v] = k

        return reversed_dict

    def get_node_id(self, node) -> int:

        if not self.visited_nodes:
            # if the dict is empty, return default index
            return 0

        else:
            # if node exists already, get the index
            if node in self.visited_nodes.values():
                reversed_dict = UccaMRPConverter.reverse_dict(self.visited_nodes)
                return reversed_dict[node]

            else:
                # if node does not exist, return the next big integer
                nid = sorted(self.visited_nodes.keys()).pop() + 1
                return nid

    def get_terminal_node_id(self, node) -> int:

        nid = sorted(self.visited_nodes.keys()).pop() + 1
        self.visited_nodes[nid] = node
        return nid

    def update_edges(self, p_id, label, ch_id):

        new_edge = {"source": p_id,
                    "target": ch_id,
                    "label": label}

        self.edges.append(new_edge)

    def update_nodes(self, nid, node, is_terminal=False):
        if is_terminal:
            start_idx, end_idx = self.get_anchor_positions(node)
            if not isinstance(start_idx, int):
                print("Warning: Terminal node not found in the original text, return 0, 0 ")
                start_idx, end_idx = 0, 0
            self.nodes.append(
                {
                    "id": nid,
                    "anchors": [{"from": start_idx, "to": end_idx}]
                })
            self.visited_nodes[nid] = node
        else:
            if nid not in self.visited_nodes:
                self.nodes.append({"id": nid})
                self.visited_nodes[nid] = node

    def get_anchor_positions(self, t_node) -> Tuple[Optional[int], Optional[int]]:

        start_idx = None
        end_idx = None

        if t_node in self.src_text_temp:
            start_idx = self.src_text_temp.index(t_node)
            end_idx = start_idx + len(t_node)
            self.src_text_temp = self.src_text_temp[:start_idx] + "_" * len(t_node) + self.src_text_temp[end_idx:]

        return start_idx, end_idx

    def update_nodes_edges(self):
        triplets = self.get_triplets()

        for p_node, label, ch_node in triplets:
            # 1. update parent node / node_id
            p_id = self.get_node_id(p_node)
            self.update_nodes(p_id, p_node)
            if p_node == "root_0":
                self.root_id = p_id

            # 2. update child node / node_id
            if label != "Z":
                ch_id = self.get_node_id(ch_node)
                self.update_nodes(ch_id, ch_node)
            else:
                ch_id = self.get_terminal_node_id(ch_node)
                self.update_nodes(ch_id, ch_node, is_terminal=True)

            # 3. update edge information between the two nodes
            self.update_edges(p_id, label, ch_id)

    def convert_tree_to_json(self):

        output = {}

        self.update_nodes_edges()
        output["id"] = self.graph_id
        output["flavor"] = 1
        output["framework"] = "ucca"
        output["time"] = datetime.now().strftime("%Y-%m-%d (%H:%M)")
        output["version"] = 0.9
        output["tops"] = [self.root_id]
        output["input"] = self.src_text
        output["nodes"] = self.nodes
        output["edges"] = self.edges

        return output


def write_file(save_to: Path, data):
    with open(save_to, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def main(graph_file: Path, sent_file: Path, save_to: Path):
    mrps = []

    with open(graph_file, 'r') as g, open(sent_file, 'r') as s:
        graphs = g.readlines()
        sents = s.readlines()

    for i, (graph, sent) in enumerate(zip(graphs, sents)):
        parser = UccaParser()
        tree = parser.parse_ucca(graph)
        converter = UccaMRPConverter(tree, sent, i)
        converted = converter.convert_tree_to_json()

        mrps.append(converted)

    write_file(save_to, mrps)


if __name__ == '__main__':
    CURR = Path().cwd()  # SGL/scripts/UCCA
    ROOT = CURR.parent.parent  # SGL
    DATA = ROOT / "data"
    UCCA_DATA = DATA / "UCCA"

    # main(graph_file=UCCA_DATA / "pred.test.tf",
    #      sent_file=UCCA_DATA / "test.sent",
    #      save_to=UCCA_DATA / "Eval" / "lpp.pred.mrp")

    main(graph_file=UCCA_DATA / "test.tf",
         sent_file=UCCA_DATA / "test.sent",
         save_to=UCCA_DATA / "Eval" / "lpp.gold.mrp")
