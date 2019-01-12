'''
This scripts should run on raw user feature files.
'''

input = "data/sample/training_no_combine.txt"
feature_mapping = "data/sample/features_mapping.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

features = {}
headers = ['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']

with open(input) as f:
    for idx,line in enumerate(f):
        records = line.strip().split("\t")
        # the 0, 1 is click, impression
        for i in range(2, len(records)):
            words = records[i]
            keywords = headers[i]
            values = [int(words)]
            if keywords not in features:
                features[keywords] = Feature(keywords)
            features[keywords].values.update(values)
        if (idx+1)%100000==0:
            print(idx+1)

for feature in features.values():
    feature.values = sorted(list(feature.values))
    feature.mapping = {val:idx for idx,val in enumerate(feature.values)}

with open(feature_mapping,'wb') as f:
    pkl.dump(features,f)


