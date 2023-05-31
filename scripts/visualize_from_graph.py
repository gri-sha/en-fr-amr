
import warnings
from collections import defaultdict
from Graph import Graph
from Node import TerminalNode
import networkx as nx
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib

def draw_graph(graph: Graph, node_ids=False):

    matplotlib.use('TkAgg')
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
    warnings.filterwarnings("ignore", category=UserWarning)

    g = nx.DiGraph()
    terminal_nodes = [node for id, node in graph.nodes.items() if isinstance(node, TerminalNode)]
    non_terminal_nodes = [node for id, node in graph.nodes.items() if not isinstance(node, TerminalNode)]
    edges = [edge for id, edge in graph.edges.items()]

    g.add_nodes_from([(n.id, {"label": n.token, "color": "white", "id": n.id}) for n in terminal_nodes])
    # g.add_nodes_from([(n.ID, {"label": "IMPLICIT", "color": "white"}) for n in passage.layer(layer1.LAYER_ID).all
    #                   if n.attrib.get("implicit")])
    g.add_nodes_from([(n.id, {"label": "", "color": "black"}) for n in non_terminal_nodes])

    g.add_edges_from([(e.source.id, e.target.id, {"label": e.label,
                                                  "style": "dashed" if "remote" in e.properties else "solid"})
                      for e in edges])

    # pos = graphviz_layout(g, prog="dot")
    pos = graph_topological_layout(graph)
    # plt.figure(1)
    # nx.draw(g, pos, with_labels=True)
    figure = plt.figure("mrp : {}".format(graph.id))
    nx.draw(g, pos, arrows=False, font_size=10,
            node_color=[d["color"] for _, d in g.nodes(data=True)],
            labels={n: d["label"] for n, d in g.nodes(data=True) if d["label"]},
            style=[d["style"] for _, _, d in g.edges(data=True)])
    nx.draw_networkx_edge_labels(g, pos, font_size=8,
                                 edge_labels={(u, v): d["label"] for u, v, d in g.edges(data=True)})
    plt.show()
    return figure, g

def graph_topological_layout(graph):

    def start(node):  # sort by children's terminal span
        if isinstance(node, TerminalNode):
            return [node.anchor[0]]
        else:
            return sorted([terminal_node.anchor[0] for terminal_node in node.terminal_spans])


    visited = defaultdict(set)
    pos = {}
    all_nodes = [node for id, node in graph.nodes.items()]
    terminals = [node for id, node in graph.nodes.items() if isinstance(node, TerminalNode)]
    non_terminals = [node for id, node in graph.nodes.items() if not isinstance(node, TerminalNode)]

    if terminals:
        implicit_offset = list(range(0, 1 + max(n.anchor[0] for n in terminals)))
        leaves = sorted([node for node in non_terminals], key=start)

        for node in all_nodes:  # draw leaves first to establish ordering
            if isinstance(node, TerminalNode):  # terminal
                x = node.anchor[0]
                pos[node.id] = (x + sum(implicit_offset[:x + 1]), 0)

            # elif node.fparent:  # implicit
            #     implicit_offset[node.fparent.end_position] += 1
    else:
        implicit_offset = [0]

    remaining = [n for n in non_terminals]
    implicits = []
    while remaining:  # draw non-terminals
        node = remaining.pop()
        if node.id in pos:  # done already
            continue
        children = [c for c in node.children if c.id not in pos and c not in visited[node.id]]
        if children:
            visited[node.id].update(children)  # to avoid cycles
            remaining += [node] + children
            continue
        if node.children:
            xs, ys = zip(*(pos[c.id] for c in node.children))
            pos[node.id] = sum(xs) / len(xs), 1 + max(ys)  # done with children
        else:
            implicits.append(node)

    # for node in implicits:
    #     fparent = node.fparent or passage.layer(layer1.LAYER_ID).heads[0]
    #     x = fparent.end_position
    #     x += sum(implicit_offset[:x + 1])
    #     _, y = pos.get(fparent.ID, (0, 0))
    #     pos[node.ID] = (x, y - 1)
    pos = {i: (x, y ** 1.01)for i, (x, y) in pos.items()}  # stretch up to avoid over cluttering
    return pos