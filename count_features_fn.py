
# features_mapping_fn = "data/features_mapping_fn.pkl"
# features_infos = 'data/features_infos_fn.txt'
features_mapping_fn = "data/features_mapping_combined.pkl"
features_infos = 'data/features_infos_combined.txt'

import pickle as pkl

class Feature:
    def __init__(self,name):
        self.name = name
        self.values = set()
        self.mapping = None

with open(features_mapping_fn,'rb') as f:
    feature_mappings = pkl.load(f)


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
    'UserID',
    'Gender',
    'Age',
    'PositionDepth'
]


with open(features_infos,'w') as f:
    for k in feat_names:
        to_write = k+':'+str(len(feature_mappings[k].mapping)) + '\n'
        f.write(to_write)

