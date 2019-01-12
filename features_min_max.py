# -*- coding:utf-8 -*-

from constants import headers, SELF, TOKENS_LEN, IMPRESSION
import joblib
import shelve

statistic_path = "data/features_statistic.pkl"
min_max_path = 'data/features_min_max.csv'
print('statistic loading....')
#statistics = joblib.load(statistic_path)
statistics_db = shelve.open(DATA_PATH + "features_mapping_combined.db")
statistics = statistics_db['features_statistic']
#statistics_db.close()
print('statistic loaded ')

feats_keys = ['DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID', 'UserID',
         'Depth', 'Position', 'Gender', 'Age']

feats_num_occurs = ['Depth', 'Position', 'RelativePosition',]
feats_len_tokens = ['QueryID', 'KeywordID', 'TitleID', 'DescriptionID',]
feats_num_imp = ['AdID', 'AdvertiserID', 'Depth', 'Position', 'RelativePosition',]

min_max_headers = ['DisplayURL', 'AdID', 'AdvertiserID',  'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'UserID', 'Depth', 'Position', 'Gender', 'Age',

           'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',

           'num_Depth', 'num_Position', 'num_RPosition',
           'num_Query', 'num_Keyword', 'num_Title', 'num_Description',
           'num_Imp__Ad', 'num_Imp__Advertiser', 'num_Imp_Depth', 'num_Imp_Position', 'num_Imp_RPosition'

           ]

min_is_zero = ['Age']  # some user id has no record, so it is set to 0

def keys_min_max(feats):

    mins = []
    maxs = []
    for feat in feats:
        print('keys_min_max: %s' % feat)
        vals = [int(key) for key in statistics[feat].statistic.keys()]
        if feat in min_is_zero:
            min_val = 0
        else:
            min_val = min(vals)
        max_val = max(vals)
        mins.append(str(min_val))
        maxs.append(str(max_val))
    return mins, maxs

def num_min_max(feats, num):
    mins = []
    maxs = []
    for feat in feats:
        print('occurs_min_max: %s' % feat)
        vals = set()
        for key in statistics[feat].statistic.keys():
            vals.update([statistics[feat].statistic[key][num]])
        vals = list(vals)
        min_val = min(vals)
        max_val = max(vals)
        mins.append(str(min_val))
        maxs.append(str(max_val))
    return mins, maxs

if __name__ == '__main__':
    mins_whole = []
    maxs_whole = []

    mins, maxs = keys_min_max(feats_keys)
    mins_whole.extend(mins)
    maxs_whole.extend(maxs)

    mins = ['0'] * 7
    maxs = ['10000'] * 7
    mins_whole.extend(mins)
    maxs_whole.extend(maxs)

    mins, maxs = num_min_max(feats_num_occurs, 2)
    mins_whole.extend(mins)
    maxs_whole.extend(maxs)

    mins, maxs = num_min_max(feats_len_tokens, 3)
    mins_whole.extend(mins)
    maxs_whole.extend(maxs)

    mins, maxs = num_min_max(feats_num_imp, 1)
    mins_whole.extend(mins)
    maxs_whole.extend(maxs)

    with open(min_max_path, 'w') as f:
        f.write(','.join(min_max_headers) + '\n')
        f.write(','.join(mins_whole) + '\n')
        f.write(','.join(maxs_whole) + '\n')
        f.close()

    statistics_db.close()

