# -*- coding:utf-8 -*-

from constants import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import sp

min_max = pd.read_csv(PATH_MIN_MAX)
features_min_max = {key: min_max[key].values for key in [

    'Depth', 'Position', 'Gender', 'Age',
    #
    #     #  14            15                  16               17           18                 19               20
    'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description',
    'group_User',

    'num_Depth', 'num_Position', 'num_RPosition',

    'num_Query', 'num_Keyword', 'num_Title', 'num_Description',

]}


def get_multi_val_feature(max_header, sparseM, path_multi_val, path_valid_len):

    # rows.nonzero()[1][rows.nonzero()[0]==0]
    cols = features_min_max[max_header][1] + 1
    rows = sparseM.shape[0]
    sparseV = sp.dok_matrix((rows, cols), dtype=np.int)
    for i in range(rows):
        pos = sparseM[i].nonzero()[1]
        if len(pos) < 1:
            # many id don't have relative tokens
            # print('empty tokens')
            continue
        sparseV[i, 0] = len(pos)
        for j in range(len(pos)):
            sparseV[i, j+1] = pos[j]

        if (i+1) % 10000 == 0:
            print(max_header+': '+str(i+1))
            # break

    sp.save_npz(path_multi_val, sparseV.tocsr())

def get_multi_val_features():
    multi_vals = []
    valid_lens = []
    print('combining tokens_vector_query')
    get_multi_val_feature('num_Query', tokens_vector_query, PATH_MUL_QUERY, PATH_LEN_QUERY)


    print('combining tokens_vector_keyword')
    get_multi_val_feature('num_Keyword', tokens_vector_keyword, PATH_MUL_KEYWORD, PATH_LEN_KEYWORD)

    print('combining tokens_vector_title')
    get_multi_val_feature('num_Title', tokens_vector_title, PATH_MUL_TITLE, PATH_LEN_TITLE)

    print('combining tokens_vector_description')
    get_multi_val_feature('num_Description', tokens_vector_description, PATH_MUL_DESCRIPTION, PATH_LEN_DESCRIPTION)



print('loading tokens_vector ......')
# tokens_multi_val_query = sp.load_npz(multi_val_query)
# tokens_multi_val_query[[1,3,999]].toarray()
tokens_vector_title = sp.load_npz(PATH_VEC_TITLE)
tokens_vector_query = sp.load_npz(PATH_VEC_QUERY)
tokens_vector_keyword = sp.load_npz(PATH_VEC_KEYWORD)
tokens_vector_description = sp.load_npz(PATH_VEC_DESCRIPTION)

get_multi_val_features()