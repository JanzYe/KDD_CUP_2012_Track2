# -*- ecoding: utf-8 -*-

import pandas as pd
from sklearn.utils import shuffle
from constants import *

input_path = DIR_PATH + 'training.txt'
output_path = DIR_PATH + 'shuffled_training.txt'


def shuffle_big_file(path_in, path_out):
    print('loading ......')
    data = pd.read_csv(path_in, delimiter='\t', header=None, dtype=str)
    print('shuffling ......')
    data = shuffle(data)
    print('writing ......')
    fout = open(path_out, 'w')
    for index, row in data.iterrows():
        to_write = '\t'.join(row)+'\n'
        fout.write(to_write)
        if index % 100000 == 0:
            print(index)
            print(to_write)
    fout.close()

def check(input_path, output_path):
    fin = open(input_path)
    print(fin.readline())
    fout = open(output_path)
    print(fout.readline())
    print(fout.readline())
    print(fout.readline())
    print(fout.readline())
    fin.close()
    fout.close()

if __name__ == '__main__':
#     check(input_path, output_path)
    shuffle_big_file(input_path, output_path)
