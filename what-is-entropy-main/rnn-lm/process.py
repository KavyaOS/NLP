import os
import re

def remove_special_tokens(input_path, output_path):
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            line = re.sub(r'<\d+>', '', line).strip()
            results.append(line)
    with open(output_path, 'w') as f:
        for line in results:
            f.write(line + '\n')


def test_exp1():
    root_folder = 'data/position'
    input_files = [ 'cleaned_data_final.txt',\
                    'labeled.test.txt', \
                    'labeled.train.txt', \
                    'labeled.valid.txt']
    output_files = ['cleaned_data_final_unlabeled.txt',\
                    'unlabeled.test.txt', \
                    'unlabeled.train.txt', \
                    'unlabeled.valid.txt']
    for in_file, out_file in zip(input_files, output_files):
        in_path = os.path.join(root_folder, in_file)
        out_path = os.path.join(root_folder, out_file)
        remove_special_tokens(in_path, out_path)


if __name__ == '__main__':
    test_exp1()