
def preprocess_str_ucca(str):
    wo_crossing_marker = remove_unused_markers(str)
    return wo_crossing_marker


def remove_unused_markers(str):
    str = str.replace("] ... ", "] ")
    str = str.replace("*", "")
    return str