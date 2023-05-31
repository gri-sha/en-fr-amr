
def ucca_to_dict(dir_path, split):  # split should be one of ["train", "dev", "test]

    sentences = open(dir_path / '{}.sent'.format(split))
    graphs = open(dir_path / '{}.tf'.format(split))
    bitexts = map(lambda zipped: {'en': zipped[0].rstrip(), 'ucca': zipped[1].rstrip()},
                  zip(sentences.readlines(), graphs.readlines()))

    sentences.close()
    graphs.close()

    return list(bitexts)


def amr_to_dict(dir_path, split):  # split should be one of ["train", "dev", "test"]

    def get_texts(amr_file):
        f = open(amr_file, 'r')
        sentences = f.readlines()
        f.close()
        return sentences

    texts = get_texts(dir_path / "{}.txt.sent".format(split))
    with open(dir_path / "{}.txt.tf".format(split)) as g:
        graphs = g.readlines()

    bitexts = map(lambda zipped: {'en': zipped[0].rstrip(), 'amr': zipped[1].rstrip()},
                  zip(texts, graphs))

    return list(bitexts)