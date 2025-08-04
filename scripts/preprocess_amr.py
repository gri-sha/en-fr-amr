import os
import sys
import re
import importlib.util
from pathlib import Path
import argparse

"""
The structure if the output is the following:
- output_folder/
    - train.txt.graph  -> Structured NOT var-free AMR graphs, delimited with a blank line
    - train.txt.sent   -> Sentences corresponding to the AMR graphs
    - train.txt.tf     -> Linearized var-free AMR graph
"""


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        required=True,
        type=str,
        help="The folder to read the amr .txt files",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=str,
        help="The folder to to store the output files: .graph, .sent, .tf",
    )
    args = parser.parse_args()
    return args

def parse_lines(file_path):
    """
    Analog funtion to 'single_line_convert' from AMR, but it does not convert graphs to a single line.
    Returns 
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    amr_lines, sent_lines = [], []

    for num, line in enumerate(lines):
        # Check if line is empty, but considering possible duplicated empty lines
        if (not line.strip()) and amr_lines and amr_lines[-1].strip():
            amr_lines.append('\n')
        if line.startswith('# ::snt') or line.startswith('# ::tok'):
            # Save sentences 
            sent = re.sub('(^# ::(tok|snt))', '', line).strip() #remove # ::snt or # ::tok
            sent_lines.append(sent+"\n")
        if not line.startswith('#') and line.strip():
            amr_lines.append(line)

    return amr_lines, sent_lines

if __name__ == "__main__":

    # Add the AMR folder (or its parent) to sys.path
    sys.path.insert(0, str(Path(__file__).parent.parent / "AMR"))

    # Load module
    module_path = Path(__file__).parent.parent / "AMR" / "var_free_amrs.py"
    spec = importlib.util.spec_from_file_location("var_free_amrs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    var_free_amrs = module.var_free_amrs
    delete_wiki = module.delete_wiki
    delete_amr_variables = module.delete_amr_variables
    single_line_convert = module.single_line_convert

    args = create_args_parser()

    os.makedirs(args.output_folder, exist_ok=True)

    dir_before_last = os.path.basename(os.path.dirname(args.output_folder))

    # Prepare output file paths
    amrs_file = os.path.join(args.output_folder, f"{dir_before_last}.txt.graph")
    sentences_file = os.path.join(args.output_folder, f"{dir_before_last}.txt.sent")
    lin_amrs_file = os.path.join(args.output_folder, f"{dir_before_last}.txt.tf")

    # Open both output files
    with open(amrs_file, "w") as amrs_out, open(sentences_file, "w") as sentences_out, open(lin_amrs_file, "w") as lin_amrs_out:
        for root, dirs, files in os.walk(args.input_folder):
            for f in files:
                if f.endswith(".txt"):
                    file_path = os.path.join(root, f)
                    print(f"Processing file: {file_path}")
                    amrs, sentences = parse_lines(file_path)

                    amr_no_wiki = delete_wiki(file_path)
                    del_amrs = delete_amr_variables(amr_no_wiki)
                    single_amrs, _ = single_line_convert(del_amrs, '')

                    for amr in amrs:
                        amrs_out.write(amr)

                    for sentence in sentences:
                        sentences_out.write(sentence)

                    for amr in single_amrs:
                        lin_amrs_out.write(amr+"\n")
