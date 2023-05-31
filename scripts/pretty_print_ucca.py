def pretty_print(lin_graph):
    num_indent = 0
    pretty = ""
    strings = [*lin_graph]

    for i, (previous_char, curr_char) in enumerate(zip([None, None] + strings[:-3], strings)):

        if curr_char == ">":
            pretty += curr_char
            pretty += "\n"
            tabs = '\t' * num_indent
            pretty += tabs

        elif curr_char == "Z" and previous_char == ">":
            pretty = pretty.rstrip('\t\n ')
            pretty += " "
            pretty += curr_char

        elif curr_char == "[":
            num_indent += 1
            pretty += curr_char

        elif curr_char == "]" and previous_char == "]":
            num_indent -= 1
            pretty += curr_char
            pretty += "\n"
            tabs = '\t' * num_indent
            pretty += tabs

        elif curr_char == "]" and previous_char ==">":  # re-entrancy node
            pretty = pretty.rstrip('\t\n ')
            pretty += " "
            pretty += curr_char
            pretty += "\n"
            num_indent -= 1
            tabs = '\t' * num_indent
            pretty += tabs

        elif curr_char == "]":
            num_indent -= 1
            pretty += curr_char

        else:
            pretty += curr_char

    print(pretty)


if __name__ == '__main__':
    ucca_linearized = "[ <root_0> H [ <H_0> F [ <F_0> Z [ there ] ] D [ <D_0> Z [ might ] ] F [ <F_1> Z [ be ] ] S [ <S_0> Z [ bigger ] ] A [ <A_0> E [ <E_0> Z [ bagel ] ] C [ <C_0> Z [ places ] ] ] L [ <L_0> Z [ and ] ] H [ <H_1> D [ <D_1> Z [ more ] ] S [ <S_1> Z [ well ] ] A [ <A_1> Z [ known ] ] A [ <A_2> R [ <R_0> Z [ in ] ] F [ <F_2> Z [ the ] ] C [ <C_1> Z [ area ] ] ] ] L [ <L_1> Z [ but ] ] H [ <H_2> A [ <A_3> Z [ Family ] Z [ Bagels ] ] F [ <F_3> Z [ are ] ] S [ <S_2> Z [ nice ] ] A [ <A_4> C [ <C_2> Z [ people ] ] U [ <U_0> Z [, ] ] E [ <E_1> S [ <S_3> Z [ small ] ] A* [ <A_5> Z [ shop ] ] ] C [ <A_5> ] ] L [ <L_2> Z [ and ] ] H [ <H_3> A* [ <A_3> ] D [ <D_2> Z [ incredibly ] ] S [ <S_4> Z [ friendly ] ] U [ <U_1> Z [. ] ] ] ]"
    pty = pretty_print(ucca_linearized)
    print(pty)
    print("\n===========================\n")