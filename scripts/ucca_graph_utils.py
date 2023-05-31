import json
from Node import Node, TerminalNode
from Edge import Edge
from typing import List, Union, Tuple, Dict


def get_anchor_node_ids(json_to_parse) -> List[int]:
    # return a list of ids of nodes having anchors (pre-terminal nodes)
    terminal_ids = []

    for node in json_to_parse["nodes"]:
        if "anchors" in node:
            terminal_ids.append(node["id"])

    return terminal_ids


def get_anchor_positions(curr_node: Node, json_to_parse)-> List[Tuple]:

    anchor_positions = []
    for node in json_to_parse["nodes"]:
        if node["id"] == curr_node.id:
            for anchor in node["anchors"]:
                start, end = anchor["from"], anchor["to"]
                anchor_positions.append((start, end))
    return anchor_positions


def make_new_terminal_nodes(initial_node_ids, parent_node, json_to_parse):
    # Create new (a) terminal node(s) and return the new node(s)
    t_nodes = {}
    anchor_positions = get_anchor_positions(parent_node, json_to_parse)

    for start, end in anchor_positions:
        # Assign an incremental id to terminal node and create a new terminal node
        tnode_id = max(initial_node_ids) + 1
        tnode = TerminalNode(tnode_id)
        initial_node_ids.append(tnode_id)  # update an id list
        t_nodes[tnode_id] = tnode

        # Set attributes - anchor positions, tokens
        tnode.set_anchor((start, end))
        token = json_to_parse["input"][start:end]
        tnode.set_token(token)

    return t_nodes, initial_node_ids


def make_new_node(node_id):
        node = Node(node_id)
        return {node_id: node}


def make_new_edge(p_node: Node, c_node: Union[Node, TerminalNode], label: str, edges, edge_properties=None):

    # Add edge to edges
    # Edges = [{edge_id : Edge}, {edge_id : Edge} ... ]
    edge = Edge(p_node, c_node, label)

    # Set edge property
    if edge_properties is not None:
        edge.set_property(edge_properties)

    # Set edge label (+ handle multi-labeled edges)
    edge_id = "{}->{}".format(p_node.id, c_node.id)

    if edge_id not in edges:
        edge_id = edge_id
    else:
        new_label = edges[edge_id].label + "|{}".format(label)
        edge.label = new_label

    return {edge_id: edge}


def parse_tree(json_to_parse) -> Tuple[Dict, Dict]:
    nodes = {}
    edges = {}
    initial_node_ids = [node["id"] for node in json_to_parse["nodes"]] # necessary to create new terminal token id
    pre_terminal_nodes = get_anchor_node_ids(json_to_parse)

    for edge in json_to_parse["edges"]:

        sourceID = edge["source"]
        targetID = edge["target"]
        label = edge["label"]
        edge_properties = edge["properties"] if "properties" in edge.keys() else None
        # edge_properties = edge["attributes"] if "attributes" in edge.keys() else None #TODO

        # Add source and target to nodes
        # Nodes = [{node_id : Node}, {node_id : Node} ... ]

        if sourceID not in nodes:
            nodes.update(make_new_node(sourceID))
        if targetID not in nodes:
            nodes.update(make_new_node(targetID))

            if targetID in pre_terminal_nodes:
                t_nodes, initial_node_ids = make_new_terminal_nodes(initial_node_ids, nodes[targetID], json_to_parse)
                nodes.update(t_nodes)

                for t_id, t_node in t_nodes.items():
                    nodes[targetID].add_child(t_node)
                    t_node.add_parent(nodes[targetID])
                    edges.update(make_new_edge(nodes[targetID], t_node, "Terminal", edges))

        nodes[sourceID].add_child(nodes[targetID])
        nodes[targetID].add_parent(nodes[sourceID])
        edges.update(make_new_edge(nodes[sourceID], nodes[targetID], label, edges, edge_properties))

    return nodes, edges


def set_terminal_span(nodes, edges):
    for node_id in nodes:
        if not isinstance(nodes[node_id], TerminalNode):
            terminal_nodes = get_my_terminal_nodes(nodes[node_id], nodes, edges)
            nodes[node_id].terminal_spans = [node for node in terminal_nodes if isinstance(node, TerminalNode)]
    return nodes


def get_my_terminal_nodes(curr_node, nodes, edges, terminal_nodes=None): #TODO: how to not return an intermediate list..?
    # Added by Pycharm
    if terminal_nodes is None:
        terminal_nodes = list()

    if isinstance(curr_node, TerminalNode):
        return curr_node

    # consider only a direct child (not a remote)
    for child in curr_node.children:
        edge = edges["{}->{}".format(curr_node.id, child.id)]
        if "remote" not in edge.properties:
            terminal_nodes.append(get_my_terminal_nodes(child, nodes, edges, terminal_nodes))

    return terminal_nodes



if __name__ == '__main__':
    mrp_ex = json.loads('{"id": "187266-0003", "flavor": 1, "framework": "ucca", "version": 0.9, "time": "2019-05-18 (06:33)", "input": "SHE KNOWS GREAT FOOD AND DINING EXPERIENCES.", "tops": [8], "nodes": [{"id": 0, "anchors": [{"from": 0, "to": 3}]}, {"id": 1, "anchors": [{"from": 4, "to": 9}]}, {"id": 2, "anchors": [{"from": 10, "to": 15}]}, {"id": 3, "anchors": [{"from": 16, "to": 20}]}, {"id": 4, "anchors": [{"from": 21, "to": 24}]}, {"id": 5, "anchors": [{"from": 25, "to": 31}]}, {"id": 6, "anchors": [{"from": 32, "to": 43}]}, {"id": 7, "anchors": [{"from": 43, "to": 44}]}, {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12}, {"id": 13}], "edges": [{"source": 8, "target": 9, "label": "H"}, {"source": 11, "target": 4, "label": "L"}, {"source": 10, "target": 2, "label": "S"}, {"source": 12, "target": 13, "label": "P"}, {"source": 9, "target": 1, "label": "S"}, {"source": 11, "target": 12, "label": "H"}, {"source": 13, "target": 7, "label": "U"}, {"source": 9, "target": 11, "label": "A"}, {"source": 13, "target": 5, "label": "C"}, {"source": 10, "target": 3, "label": "A"}, {"source": 11, "target": 10, "label": "H"}, {"source": 13, "target": 6, "label": "E"}, {"source": 9, "target": 0, "label": "A"}]}')
    nodes, edges = parse_tree(mrp_ex)
    nodes = set_terminal_span(nodes)
    print(nodes)
    print(edges)