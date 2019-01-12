'''
This scripts should run on raw user feature files.
'''

# total number of records in training.txt is almost 150000000, exactly 149639105
# divide the training.txt to feature_mapped_train and feature_mapped_valid
# feature_mapped_train(120000000), feature_mapped_valid(30000000)

#  data_descriptionid, data_purchasedkeywordid, data_queryid, data_titleid, gender, age are not need to mapping, because every records are distinct,
#  and their id can consider as the value after mapping



input = "/home/yezhizi/Documents/python/2018DM_Project/track2/test_combined.txt"
mapped = "data/feature_mapped_combined_test.data"

features_mapping = "data/features_mapping_combined.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping,'rb') as f:
    print('features mapping loading ... ')
    features = pkl.load(f)
    print('features mapping loaded')

feat_names = [
    'DisplayURL',
    'AdID',
    'AdvertiserID',
    'Depth',
    'Position',
    'QueryID',
    'KeywordID',
    'TitleID',
    'DescriptionID',
    'Gender',
    'Age',
]

user_offsets = {}
offset = 0
with open(input) as fr, open(mapped, 'w') as fw:
    for idx, line in enumerate(fr):
        records = line.strip().split("\t")
        to_write = []
        for i in range(len(records)):
            val = int(records[i])
            keywords = feat_names[i]
            if val not in features[keywords].mapping:
                val = 0

            values = [str(features[keywords].mapping[val])]
            #record = keywords+" "+" ".join(values)
            to_write.append(values[0])

        to_write = ','.join(to_write) + "\n"

        fw.write(to_write)

        if (idx+1) % 200000 == 0:
            print(idx+1)
            print(to_write)
            #break

    fr.close()
    fw.close()
