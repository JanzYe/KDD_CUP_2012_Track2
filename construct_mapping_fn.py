'''
This scripts should run on training combined files.
only mapping sparse features
'''

import pickle as pkl
from constants import *
import joblib
import shelve

input = DIR_PATH + "training.txt"
feature_mapping = DATA_PATH + "features_mapping_combined.pkl"
features_statistic = DATA_PATH + "features_statistic.pkl"

class FeatureNumberical:
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
                       'TitleID', 'DescriptionID','UserID', 'PositionDepth']

headers_add = ['Gender', 'Age']
features[headers_add[0]]= Feature(headers_add[0])
for val in range(3):
    features[headers_add[0]].values.update([str(val)])
    #features[headers_add[0]].values.update([val])
features[headers_add[1]] = Feature(headers_add[1])
for val in range(7):
    features[headers_add[1]].values.update([str(val)])
    #features[headers_add[1]].values.update([val])

print('features_statistic loading ... ')
#statistics = joblib.load(statistic_path)
statistics_db = shelve.open(DATA_PATH + "features_mapping_combined.db")
feat_stat = statistics_db['features_statistic']
#statistics_db.close()
print('features_statistic loaded')

queryID20 = 0
for id in feat_stat[headers[7]].statistic:
    if feat_stat[headers[7]].statistic[id][2] > 20:
        queryID20 += 1
print('queryid occurs > 20: %d' % queryID20)

with open(input) as f:
    for idx,line in enumerate(f):
        records = line.strip().split("\t")
        # the 0, 1 is click, impression
        depth = '0'
        pos = '0'
        for i in range(2, len(headers)):
            words = records[i]
            keywords = headers[i]
            if keywords not in features:
                features[keywords] = Feature(keywords)
                # add zero to mapping values not in training data which indicates as unknown
                features[keywords].values.update(['0'])

            if (keywords == headers[7]) and (feat_stat[keywords].statistic[words][2] <= 20):
                # only use queryID which occurs more than 20 times
                continue
                
            if (keywords == headers[11]) and (feat_stat[keywords].statistic[words][2] <= 20):
                # only use queryID which occurs more than 20 times
                continue
                
            if keywords == 'Depth':
                depth = words
                
            if keywords == 'Position':
                pos = words
                records.append(pos+','+depth)

            features[keywords].values.update([words])
        if (idx+1)%100000==0:
            print(len(features['QueryID'].values))
#             break
    print(idx + 1)

for feature in features.values():
    feature.values = sorted(list(feature.values))
    feature.mapping = {val:idx for idx,val in enumerate(feature.values)}
    feature.values = None
    print(len(feature.mapping))

with open(feature_mapping,'wb') as f:
    print('saving ......')
    pkl.dump(features, f)
    print('saved')

statistics_db.close()
