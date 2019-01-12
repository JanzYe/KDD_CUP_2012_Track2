# -*- coding:utf-8 -*-

import load_user_profile
import pickle as pkl
from constants import *
import sys
import shelve

dir_path = DIR_PATH
statistic_path = DATA_PATH + "features_statistic.pkl"
numerical_path = DATA_PATH + "numerical.pkl"
features_mapping = "data/features_mapping_combined.pkl"

mode = TRAIN  # test train
memory_enough = True

# origin: both sparse and dense feature?
# group: IDs are divided into 10000 groups by their values :dense feature?
# aCTR: average CTR for each value of each features  :dense feature?
# pCTR: pseudo CTR for each value of each features, alpha=0.05, beta=75, (click+a*b)/(impression+b) :dense feature?
# num: numerical value of  :dense feature?
# num: length of tokens in :dense feature?
# num_Imp: number of impression for  :dense feature?
# headers = ['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
#                        'TitleID', 'DescriptionID', 'UserID', 'Gender', 'Age', 'RelativePosition',

#            'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',

#            'aCTR_Ad', 'aCTR_Advertiser', 'aCTR_Depth', 'aCTR_Position', 'aCTR_RPosition',

#            'pCTR_Url', 'pCTR_Ad', 'pCTR_Advertiser', 'pCTR_Query', 'pCTR_Keyword', 'pCTR_Title', 'pCTR_Description', 'pCTR_User',
#             'pCTR_Gender', 'pCTR_Age', 'pCTR_RPosition',

#            'num_Depth', 'num_Position', 'num_RPosition',
#            'num_Query', 'num_Keyword', 'num_Title', 'num_Description',
#            'num_Imp__Ad', 'num_Imp__Advertiser', 'num_Imp_Depth', 'num_Imp_Position', 'num_Imp_RPosition'

#            ]
# RelativePosition = (depth - position) / depth

with open(features_mapping,'rb') as f:
    print('features mapping loading ... ')
    mapping = pkl.load(f)
    print('features mapping loaded')
print(mapping['Age'].mapping.keys())

print('statistic loading....')
#statistics = joblib.load(statistic_path)
statistics_db = shelve.open(DATA_PATH + "features_mapping_combined.db")
statistics = statistics_db['features_statistic']
#statistics_db.close()
print('statistic loaded ')



groups = 10000.0

def group_statistic(feats):

    mins = []
    maxs = []
    intervals = []
    for feat in feats:
        print('grouping: %s' % feat)
        vals = [int(key) for key in statistics[feat].statistic.keys()]
        min_val = min(vals)
        max_val = max(vals)
        mins.append(min_val)
        maxs.append(max_val)
        intervals.append((max_val - min_val) / groups)
    return mins, maxs, intervals

#'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',
mins, maxs, intervals = group_statistic(['AdID', 'AdvertiserID', 'QueryID',
                                         'KeywordID', 'TitleID', 'DescriptionID', 'UserID'])

def group_mapping(ids):

    mapped = []
    for i in range(len(ids)):
        map = int((ids[i] - mins[i]) / intervals[i])
        if map < 0:
            map = 0
        elif map >= groups:
            map = groups - 1
        mapped.append(str(map))
    return mapped

def aCTR_mapping(ids, feats):
    mapped = []

    for i in range(len(ids)):
        if str(ids[i]) not in statistics[feats[i]].statistic:
            data = [0, 0, 0, 0]
        else:
            data = statistics[feats[i]].statistic[str(ids[i])]
        if data[1] == 0:
            mapped.append(str(0))
        else:
            aCTR = 1.0 * data[0] / data[1]
            mapped.append(str('%.3f' % aCTR))
    return mapped

def pCTR_mapping(ids, feats):
    mapped = []
    alpha = 0.05
    beta = 75
    for i in range(len(ids)):
        if str(ids[i]) not in statistics[feats[i]].statistic:
            data = [0, 0, 0, 0]
        else:
            data = statistics[feats[i]].statistic[str(ids[i])]
        pCTR = (data[0] + alpha * beta) / (data[1] + beta)
        mapped.append(str('%.3f' % pCTR))
    return mapped

def num_occurs_mapping(ids, feats):
    mapped = []
    for i in range(len(ids)):
        if str(ids[i]) not in statistics[feats[i]].statistic:
            data = [0, 0, 0, 0]
        else:
            data = statistics[feats[i]].statistic[str(ids[i])]
        mapped.append(str(data[2]))
    return mapped

def len_tokens_mapping(ids, feats):
    mapped = []
    for i in range(len(ids)):
        if str(ids[i]) not in statistics[feats[i]].statistic:
            data = [0, 0, 0, 0]
        else:
            data = statistics[feats[i]].statistic[str(ids[i])]
        mapped.append(str(data[3]))
    return mapped

def num_imp_mapping(ids, feats):
    mapped = []
    for i in range(len(ids)):
        if str(ids[i]) not in statistics[feats[i]].statistic:
            data = [0, 0, 0, 0]
        else:
            data = statistics[feats[i]].statistic[str(ids[i])]
        mapped.append(str(data[1]))
    return mapped

def sparse_mapping(ids, feats):
    mapped = []
    for i in range(len(ids)):
        key = feats[i]
        #int_set = ['Age', 'Gender']
        int_set = []
        if key not in int_set:
            val = str(ids[i])
        else:
            val = int(ids[i])
            
        if val not in mapping[key].mapping:
            if key not in int_set:
                val = '0'
            else:
                val = int(0)

        values = [str(mapping[key].mapping[val])]
        mapped.append(str(values[0]))
    return mapped


def combine(profile, mode):

    if mode == TRAIN:
        file = open(dir_path + "shuffled_training.txt")
        combined = open('data/combined_mapped_shuffled_training.txt', 'w')
        combined.write(','.join(headers) + '\n')
    elif mode == TEST:
        file = open(dir_path + "test.txt")
        combined = open(dir_path + "combined_mapped_test.txt", 'w')
        combined.write(','.join(headers[2:]) + '\n')

    # not_exist_user_id = open(dir_path+"not_exist_user_id.txt", 'a')
#     if memory_enough == 'True':
 

    i = 1
    for line in file:
        # file_train.write(line)
        to_write = []
        record = line.strip().split('\t')
        if mode == TRAIN:
            # click = int(record[0])
            # impression = int(record[1])
            # ctr = 1.0 * click / impression
            # to_write.append(str('%.3f' % ctr))
            # to_write.extend(record[2:])
            to_write.extend(record)
        elif mode == TEST:
            to_write.extend(record)

        if record[-1] not in profile:
            print('no userid: %s' % record[-1])
            # not_exist_user_id.write(record[-1]+'\n')
            # 0 denotes unknown
            # gender； 0, 1, 2
            # age: 0, 1, 2, 3, 4, 5, 6
            profile[record[-1]] = [0, 0]

        # gender age
        gender = profile[record[-1]][0]
        age = profile[record[-1]][1]
        to_write.append(str(gender))
        to_write.append(str(age))

        # relative position
        depth = int(record[-7])
        position = int(record[-6])
        rPosition = 1.0 * (depth - position) / depth
        to_write.append(str('%.3f' % rPosition))

        # 'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',
        adID = int(record[-9])
        advertiserID = int(record[-8])
        userID = int(record[-1])
        descriptionID = int(record[-2])
        titleID = int(record[-3])
        keywordID = int(record[-4])
        queryID = int(record[-5])
        displayUrl = int(record[-10])
        feats = ['AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID', 'UserID']
        to_write.extend(group_mapping([adID, advertiserID, queryID, keywordID, titleID, descriptionID, userID]))

        # 'aCTR_Ad', 'aCTR_Advertiser', 'aCTR_Depth', 'aCTR_Position', 'aCTR_RPosition',
        feats = ['AdID', 'AdvertiserID', 'Depth', 'Position', 'RelativePosition']
        to_write.extend(aCTR_mapping([adID, advertiserID, depth, position, rPosition], feats))

        # 'pCTR_Url', 'pCTR_Ad', 'pCTR_Advertiser', 'pCTR_Query', 'pCTR_Keyword', 'pCTR_Title', 'pCTR_Description', 'pCTR_User',
        # 'pCTR_Gender', 'pCTR_Age', 'pCTR_RPosition',
        feats = ['DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID', 'UserID',
                 'Gender', 'Age', 'RelativePosition']
        to_write.extend(pCTR_mapping([displayUrl, adID, advertiserID, queryID, keywordID, titleID, descriptionID, userID,
                                     gender, age, rPosition], feats))

        # 'num_Depth', 'num_Position', 'num_RPosition',     num_of_occurs
        feats = ['Depth', 'Position', 'RelativePosition']
        to_write.extend(num_occurs_mapping([depth, position, rPosition], feats))

        # 'num_Query', 'num_Keyword', 'num_Title', 'num_Description',     len of tokens
        feats = ['QueryID', 'KeywordID', 'TitleID', 'DescriptionID']
        to_write.extend(len_tokens_mapping([queryID, keywordID, titleID, descriptionID], feats))

        # 'num_Imp__Ad', 'num_Imp__Advertiser', 'num_Imp_Depth', 'num_Imp_Position', 'num_Imp_RPosition'  num_of_impression
        feats = ['AdID', 'AdvertiserID', 'Depth', 'Position', 'RelativePosition']
        to_write.extend(num_imp_mapping([adID, advertiserID, depth, position, rPosition], feats))

        # sparse feature
        feats = ['DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'UserID', 'Gender', 'Age', 'PositionDepth']
        to_write.extend(sparse_mapping([displayUrl, adID, advertiserID, depth, position, queryID, keywordID, 
                                        titleID, descriptionID, userID, gender, age, str(position)+','+str(depth)], feats))

        to_write = ','.join(to_write)
        if i % 100000 == 0:
            print(i)
            print(to_write)
#             break
        to_write = to_write + '\n'
        combined.write(to_write)

        i = i + 1

    file.close()
    combined.close()


if __name__ == '__main__':
    mode = str(sys.argv[1])
    
    print('Starting ...')
    profile = load_user_profile.load()
    # profile = {'490234': '1,2'}
    
    combine(profile, mode)
    statistics_db.close()
    print('End')
