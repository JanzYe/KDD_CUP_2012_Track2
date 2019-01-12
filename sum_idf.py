
from constants import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import sp
import numpy as np

paths_input = [PATH_VEC_TITLE, PATH_VEC_KEYWORD, PATH_VEC_QUERY, PATH_VEC_DESCRIPTION, ]
paths_output = [PATH_SUM_TITLE, PATH_SUM_KEYWORD, PATH_SUM_QUERY, PATH_SUM_DESCRIPTION, ]

for i in range(len(paths_input)):
    print('loading '+ paths_input[i])
    tokens_vector = sp.load_npz(paths_input[i])
    print(tokens_vector.shape)
    print('summing ......')
    sum_ = tokens_vector.sum(axis=1)
    sum_ = np.array(sum_ / max(sum_))
    print('writing to ' + paths_output[i])
    with open(paths_output[i], 'w') as fw:
        print(sum_.shape)
        for j in range(len(sum_)):
            val = float(sum_[j][0])
            to_write = str(val) + '\n'
            if j % 1000000 == 0:
                print(to_write)
            fw.write(to_write)
        print(j)
        fw.close()

