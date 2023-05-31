from ucca import layer0
import ucca.convert as cv
from settings import UCCA_DATA
from visualization import topological_layout
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from visualization import draw
import matplotlib.pyplot as plt
import matplotlib
from pretty_print import pretty_print

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


class SentenceLinearizer:

    def __init__(self, passage):
        self.passage = passage
        self.tags_count = []
        self.visited = {"1.1": "root_0"}  # TODO : empty
        self.labeldict = {}                # TODO : remove
        self.visited_parents = []
        self.output_str = ""
        self.G = nx.DiGraph()

        # pass the root node to start linearization
        self.dfs_linearize(passage.nodes['1.1'])

    def get_edge_label(self, edge):

        edge_label = "|".join(sorted(edge.tags))
        is_uncertain = edge.attrib.get("uncertain")
        is_remote = edge.attrib.get("remote")

        if is_remote:
            edge_label += "*"
        if is_uncertain:
            edge_label += "?"

        return edge_label

    def get_label_id(self, label):

        label = label.replace('*', '')
        label = label.replace('?', '')

        label_count = self.tags_count.count(label)
        self.tags_count.append(label)

        return "{}_{}".format(label, label_count)

    def add_to_visited(self, node_id, label_id):
        self.visited[node_id] = label_id

    def dfs_linearize(self, fnode):

        def start(e):
             return e.child.position if e.child.layer.ID == layer0.LAYER_ID else e.child.start_position

        sorted_edges = sorted(fnode, key=start)

        for edge, next_edge in zip(sorted_edges, sorted_edges[1:] + [None]):

            # if the node is not pre-terminal node, do it recursively
            if edge.child.layer.ID != layer0.LAYER_ID:
                node = edge.child
                edge_label = self.get_edge_label(edge)

                # if node is implicit, ignore the node (following mrp 2019)
                if node.attrib.get("implicit"):
                    continue

                # if node is already visited, get its id and add it to an output
                elif node.ID in self.visited:
                    self.output_str += "{} [ <{}> ".format(edge_label, self.visited[node.ID])

                    ####### add node to graph ###### -> only for debugging purpose
                    self.labeldict[edge.parent.ID] = self.visited[edge.parent.ID]
                    self.G.add_edge(edge.parent.ID, node.ID, label=edge_label)
                    ##### end #####

                # if new node, add it to output and continue to child node recursively
                else:
                    label_id = self.get_label_id(edge_label)
                    self.add_to_visited(node.ID, label_id)
                    self.output_str += "{} [ <{}> ".format(edge_label, label_id)

                    ####### add node to graph ######  -> only for debugging purpose
                    self.labeldict[edge.parent.ID] = self.visited[edge.parent.ID]
                    self.labeldict[node.ID] = self.visited[node.ID]
                    self.G.add_edge(edge.parent.ID, node.ID, label=edge_label)
                    ##### end #####

                    self.dfs_linearize(node)

            # if node is pre-terminal node, add it to linearized output
            else:
                node = edge.child
                self.output_str += "Z [ {} ".format(node.text)


            # reaching the end of recursive, add closing bracket
            self.output_str += "] "

if __name__ == '__main__':

    matplotlib.use('TkAgg')
    file = UCCA_DATA / "UCCA_English-EWT" / "xml" / "001325.xml"
    # file = UCCA_DATA / "UCCA_English-EWT/xml/009775.xml"
    dir_ = UCCA_DATA / "UCCA_French-20K-master" / "xml"

    for file in dir_.iterdir():
        passage = cv.xml2passage(str(file))
        for i, sentence_ps in enumerate(cv.split2sentences(passage)):
            print("=========passage {}============".format(i))
            linearized = SentenceLinearizer(sentence_ps)
            pretty_print(linearized.output_str)
            draw(sentence_ps)
            # try:
            # pos = graphviz_layout(linearized.G, prog="dot")
            # plt.figure(3)
            # nx.draw(linearized.G, labels=linearized.labeldict, with_labels=True, pos=pos, arrows=True, font_size=10)
            # nx.draw_networkx_edge_labels(linearized.G, pos, font_size=10, edge_labels={(u, v): d["label"] for u, v, d in linearized.G.edges(data=True)})
            # draw(sentence_ps)
            # plt.show()
            # except:
            #     plt.clf()
            #     continue