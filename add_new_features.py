# -*- coding:utf-8 -*=

import pickle as pkl
from constants import *

original = "data/combined_mapped_training.txt"
# original = "/data/feature_mapped_combined_test.data"
added = 'data/combined_mapped_training_cli_imp.txt'

features_in = DATA_TRAINING
# feature_mapping = "data/features_mapping_combined.pkl"
# feature_statistic = "data/numerical.pkl"

class FeatureStatistic:
    def __init__(self,name):
        self.name = name
        self.statistic = {}

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

features = {}
headers = ['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']

if __name__ == '__main__':

    # with open(feature_statistic, 'rb') as f:
    #     print('feature_statistic loading ... ')
    #     feat_stat = pkl.load(f)
    #     print('feature_statistic loaded')
    #
    # with open(feature_mapping, 'rb') as f:
    #     print('features mapping loading ... ')
    #     features = pkl.load(f)
    #     print('features mapping loaded')

    with open(original, 'r') as fo, open(features_in, 'r') as ff, open(added, 'w') as fa:
        index = 0
        for line_o in fo:
            line_f = ff.readline()
            record = line_f.strip().split('\t')
            to_write = record[0] + ',' + record[1] + ',' + line_o
            fa.write(to_write)
            index += 1
            if index % 1000000 == 0:
                print(to_write)
                break

        fo.close()
        ff.close()
        fa.close()