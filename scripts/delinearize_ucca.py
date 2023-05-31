import json
import re
from ucca_utils import preprocess_str_ucca

class UccaParser:

    def __init__(self):
        self.non_terminal_node = re.compile('^[A-Z](\*)*(\|[A-Z](\*)*)*_[0-9]{1,2}$')  # 'A_0' or 'A|S_0' or 'A*_0'
        self.visited = set()
        self.tree = dict()

    def update_tree(self, depth, curr_triplet):
        if depth in self.tree:
            self.tree[depth].append(curr_triplet)
        else:
            self.tree[depth] = [curr_triplet]

        self.visited.add(curr_triplet[0])
        self.visited.add(curr_triplet[1])

    def add_starting_node(self, curr_depth):
        # add starting node to the triplet:
        updated_triplet = [self.tree[curr_depth][-1][0]]  # 'from' in curr_triplet = "root_0"
        return updated_triplet

    def update_curr_status(self, curr_triplet, curr_node, depth):
        # update some variables
        is_node = False
        if curr_node != "":
            curr_triplet.append(curr_node)
        curr_node = ""

        # update curr status only when the triplet is complete
        if len(curr_triplet) == 3:
            self.update_tree(depth, curr_triplet)
            if self.is_terminal(curr_triplet):
                curr_triplet = [] # if it was a terminal node, empty the triplet
            else:
                curr_triplet = [self.tree[depth][-1][-1]]  # if it was not a terminal node, continue to a child node

        return is_node, curr_node, curr_triplet

    def is_terminal(self, curr_triplet):
        if curr_triplet[1] == 'Z' :               # condition 1) terminal node edge Z
            return True
        elif curr_triplet[2] in self.visited:     # condition 2) re-entrant node
            return True
        else:
            return False


    def parse_ucca(self, lin_graph):
        preprocessed = preprocess_str_ucca(lin_graph)
        queue = list(preprocessed.replace(" ", ""))
        curr_depth = -1
        tree = dict()
        is_node = False
        curr_node = ""
        temp_edge = ""
        curr_triplet = []
        curr_edge = ""

        while queue:
            char = queue.pop(0)

            if char == "[":
                curr_depth += 1   # root level = 0

                # update curr_edge and add it to curr_triplet by the beginning of new bracket
                if temp_edge != "":
                    curr_edge = temp_edge

                    # if edge is detected while triplet is empty, add starting node from previous nodes
                    # only when ....
                    if len(curr_triplet) == 0 :
                        curr_triplet = self.add_starting_node(curr_depth)
                        curr_node = ""

                    curr_triplet.append(curr_edge)
                    temp_edge = ""

            elif char == "<":
                # beginning of node
                is_node = True

            elif char == ">":
                # end of node, update the status or not (depending on starting node, ending node)
                is_node, curr_node, curr_triplet = self.update_curr_status(curr_triplet, curr_node, curr_depth)

            elif char == "]":
                # end of triplet, update the status (or not) if the current triplet is yet to be added to the tree (this is a depth first search tree)
                is_node, curr_node, curr_triplet = self.update_curr_status(curr_triplet, curr_node, curr_depth)
                curr_depth -= 1
                if curr_depth < 0:
                    return self.tree

            elif is_node:
                curr_node += char

            elif not is_node:
                temp_edge += char

                if temp_edge == "Z":
                    is_node = True

        return self.tree


if __name__ == '__main__':
    lin_ucca = '[ <root_0> H [ <H_0> D [ <D_0> T [ Highly ] ] S [ <S_0> T [ recommended ] ] ] ]'
    doulbe_rel = '[ <root_0> H [ <H_0> D [ <D_0> T [ Highly ] ] S|A [ <S|A_0> T [ recommended ] ] ] ]'
    new = '[ <root_0> H [ <H_0> A [ <A_0>  Z [ That ] ] D [ <D_0>  Z [ would ] ] F [ <F_0>  Z [ be ] ] D [ <D_1>  Z [ too ] ] S [ <S_0>  Z [ complicated ] ] U [ <U_0>  Z [ . ] ] ] ]'
    new2 = '[ <root_0> H [ <H_0> A [ <A_0> F [ <F_0>  Z [ The ] ] E [ <E_0> S [ <S_0>  Z [ little ] ] A* [ <A_1> S|A [ <S|A_0>  Z [ prince ] ] ] ] C [ <A_1> ] ] S [ <S_1>  Z [ flushed ] ] D [ <D_0>  Z [ once ]  Z [ more ] ] U [ <U_0>  Z [ . ] ] ] ]'
    test ='[ <root_0> H [ <H_0> P [ <P_0>  Z [ Want ] ] A [ <A_0> F [ <F_0>  Z [ a ] ] E [ <E_0> S [ <S_0>  Z [ great ] ] A* [ <A_1>  Z [ burger ] ] ] C [ <A_1> ] ] ] U [ <U_0>  Z [ ? ] ] ]'
    parser = UccaParser()
    tree = parser.parse_ucca(lin_ucca)
    print(json.dumps(tree, sort_keys=False, indent=4))




