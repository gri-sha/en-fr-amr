from typing import List

class Edge:
    def __init__(self, source_node, target_node, label):

        self.source = source_node
        self.target = target_node
        self.label = label
        self.properties = []
        self.id = "{}->{}".format(self.source.id, self.target.id)

    def set_property(self, properties: List):
        self.properties = properties
