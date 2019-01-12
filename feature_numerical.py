'''
This scripts should run on raw training files.
for each feature, counts the times for each different value that occurs
'''

import pickle as pkl
from constants import *

# input = "/home/yezhizi/Documents/python/2018DM_Project/track2/training.txt"
# feature_mapping = "data/features_mapping_fn.pkl"
input = DIR_PATH + 'training.txt'
feature_mapping = DATA_PATH + "numerical.pkl"

features = {}  # structure: { headers[i]: { str(val): num } }
headers = ['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'PositionDepth']

with open(input) as f:
    for idx,line in enumerate(f):
        records = line.strip().split("\t")
        # the 0, 1 is click, impression
        depth = '0'
        pos = '0'
        for i in range(len(headers)):
            words = records[i]
            keywords = headers[i]
            if keywords not in features:
                features[keywords] = FeatureNumerical(keywords)

            if words not in features[keywords].statistic:
                features[keywords].statistic[words] = 0
            else:
                features[keywords].statistic[words] += 1
            
            if keywords == 'Depth':
                depth = words
                
            if keywords == 'Position':
                pos = words
                records.append(pos+','+depth)
                

        if (idx+1)%100000==0:
            print(idx+1)
            # break

    print(idx + 1)

with open(feature_mapping,'wb') as f:
    pkl.dump(features, f)


