import ucca.ioutil as util
import ucca.convert as cv
import itertools
from argparse import ArgumentParser
from Graph import Graph
from linearize_ucca_from_mrp import UccaLinearizer
from pathlib import Path
import re
import random
import json
from tqdm import tqdm
from typing import List, Dict
from linearize_ucca import SentenceLinearizer

from settings import UCCA_DATA

def segment_to_tree(passage):
    for node in passage.nodes:
        nd = passage.nodes[node]
        if nd.extra:
            if isinstance(nd.extra["tree_id"], int):
                sub_psg = nd

def linearize_from_xml(path):

    passages = util.get_passages(str(path))

    ucca_corpus = []
    print("file to list..")
    sentence_id = 0
    for i, passage in tqdm(enumerate(passages)):
        # segment_to_tree(passage)
        sentences = cv.split2sentences(passage)

        for sentence in sentences:
            linearizer = SentenceLinearizer(sentence)
            linearized = linearizer.output_str
            rooted = "[ <root_0> " + linearized + "]"
            # ucca_corpus.append((cv.to_text(sentence)[0], str(sentence.root)))
            ucca_corpus.append((cv.to_text(sentence)[0], str(rooted)))

            sentence_id += 1

    return ucca_corpus

def node_label(node):
    return re.sub("[^(]*\((.*)\)", "\\1", node.attrib.get("label", ""))


def split(data_list):

    dev = data_list[:500]
    train = data_list[500:]
    print("dev data: {}, train_data:{} ".format(len(dev), len(train)))

    return train, dev


def write_file(ucca, data_dir, split_name):

    text_filename = str(split_name) + ".sent"
    graph_filename = str(split_name) + ".tf"
    text_file = data_dir / text_filename
    graph_file = data_dir / graph_filename

    print("writing files to {} and {}".format(str(text_file), str(graph_file)))
    with open(text_file, 'w') as t, open(graph_file, 'w') as g:
        for (text, graph) in ucca:
            t.write(text + '\n')
            g.write(graph + '\n')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--mrp_dir", type=str, default= UCCA_DATA / "mrp" / "2019" / "training" / "ucca",
                        help="directory containing mrp files (training and dev data)")
    parser.add_argument("--xml_dir", type=str, default= UCCA_DATA / "UCCA_English-LPP-main"/ "xml",
                        help="directory containing lpp xml files (test data)")

    # end of args
    args = parser.parse_args()
    ucca_mrp = Path(args.mrp_dir)
    test_xml = Path(args.xml_dir)

    print("linearizing for training and dev..")
    linearized_graphs = [] # [(text_str, graph_str), () ..]
    for mrp in ucca_mrp.iterdir():
        with open(mrp) as f:
            lines = f.readlines()

            for line in lines:
                mrp_instance = json.loads(line)
                graph = Graph()
                graph.create_from_json(mrp_instance)
                linearized = UccaLinearizer(graph)
                rooted = "[ <root_0> " + linearized.output_str + "]"
                text = graph.input
                linearized_graphs.append((text, rooted))


    random_seed = 42
    random.Random(random_seed).shuffle(linearized_graphs)
    ucca_dev = linearized_graphs[:500]
    ucca_train = linearized_graphs[500:]

    write_file(ucca_dev, UCCA_DATA, 'dev')
    write_file(ucca_train, UCCA_DATA, 'train')

    ucca_test = linearize_from_xml(test_xml)
    write_file(ucca_test, UCCA_DATA, 'test')

