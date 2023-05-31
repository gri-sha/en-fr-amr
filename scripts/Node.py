from __future__ import annotations
from typing import Union, List, Tuple


class Node:
    def __init__(self, id: int):
        self.id = id
        self.parents = set()
        self.children = set()
        self.terminal_spans = None # List of terminal nodes ?

    def add_parent(self, node: Node):
        self.parents.add(node)

    def add_child(self, node: Union[Node, TerminalNode]):
        self.children.add(node)

class TerminalNode:
    def __init__(self, id: int):
        self.id = id
        self.parents = []
        self.anchor = None # Tuple (from, start)
        self.token = ""

    def add_parent(self, node: Node):
        self.parents.append(node)

    def set_anchor(self, anchor: Tuple):
        self.anchor = anchor

    def set_token(self, token):
        self.token = token