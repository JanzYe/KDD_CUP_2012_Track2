'''
This scripts should run on raw user feature files.
'''

# input = "data/sample/training_no_combine.txt"
# new_input = "data/sample/feature_mapped.data"

input = "data/sample/validation_no_combine.txt"
new_input = "data/sample/feature_mapped_valid.data"

input_offfet = "data/sample/feature_mapped_offsets.pkl"
features_mapping = "data/sample/features_mapping.pkl"

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping,'rb') as f:
    features = pkl.load(f)

feat_names = [
    'DisplayURL',
    'AdID',
    'AdvertiserID',
    'Depth',
    'Position',
    'QueryID',
    'KeywordID',
    'TitleID',
    'DescriptionID'
]

user_offsets = {}
offset = 0
with open(input) as fr, open(new_input, 'w') as fw:
    for idx, line in enumerate(fr):
        records = line.strip().split("\t")
        # the 0, 1 are click and impression, the last one is uid
        ctr = 1.0 * int(records[0]) / int(records[1])
        to_write = str(ctr)
        for i in range(2, len(records)-1):
            val = int(records[i])
            keywords = feat_names[i-2]
            values = [str(features[keywords].mapping[val])]
            #record = keywords+" "+" ".join(values)
            to_write = to_write + ',' + str(values[0])

        to_write = to_write + "\n"
        fw.write(to_write)
        # user_offsets[int(ctr)] = offset
        # offset += len(to_write)
        if (idx) % 100000 == 0:
            print(idx)
            print(to_write)

# with open(user_offsets_fn, 'wb') as f:
#     pkl.dump(user_offsets,f)