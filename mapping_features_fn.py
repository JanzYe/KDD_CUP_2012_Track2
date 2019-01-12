'''
This scripts should run on raw user feature files.
'''

# total number of records in training.txt is almost 150000000, exactly 149639105
# divide the training.txt to feature_mapped_train and feature_mapped_valid
# feature_mapped_train(120000000), feature_mapped_valid(30000000)

#  data_descriptionid, data_purchasedkeywordid, data_queryid, data_titleid, gender, age are not need to mapping, because every records are distinct,
#  and their id can consider as the value after mapping

# input = "/home/yezhizi/Documents/python/2018DM_Project/track2/training.txt"
# new_train = "data/feature_mapped_train.data"
# new_valid = "data/feature_mapped_valid.data"
input = "/home/yezhizi/Documents/python/2018DM_Project/track2/training_combined.txt"
mapped = "data/feature_mapped_combined_training.data"

# input = "/home/yezhizi/Documents/python/2018DM_Project/track2/test_combined.txt"
# mapped = "data/feature_mapped_combined_test.data"

# features_mapping = "data/features_mapping_fn.pkl"
features_mapping = "data/features_mapping_combined.pkl"
# number_of_training = 120000000
# number_of_training = 120000

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
        # the 0, 1 are click and impression
        ctr = 1.0 * int(records[0]) / int(records[1])
        to_write = str(ctr)
        for i in range(2, len(records)):
            val = int(records[i])
            keywords = feat_names[i-2]
            if val not in features[keywords].mapping:
                val = 0

            values = [str(features[keywords].mapping[val])]
            #record = keywords+" "+" ".join(values)
            to_write = to_write + ',' + str(values[0])

        to_write = to_write + "\n"

        fw.write(to_write)

        if (idx+1) % 200000 == 0:
            print(idx+1)
            print(to_write)
            # break

    fr.close()
    fw.close()
