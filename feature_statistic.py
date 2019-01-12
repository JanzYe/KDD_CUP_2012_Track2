'''
This scripts should run on raw training files.
for each feature, counts the times for each different value that being clicked, showed(impression),
    itself occurs and the tokens len
'''

import pickle as pkl
import load_user_profile
import joblib
import shelve
from constants import *
import sys

input = DIR_PATH + 'training.txt'
feature_statistic = DATA_PATH + "features_statistic.pkl"

statistics = {}  # structure: { headers[i]: { str(val): [num_click, num_impression, num_self, num_tokens_len] } }
headers = ['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'UserID', 'Gender', 'Age']

feats = ['DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID', 'UserID',
         'Depth', 'Position', 'RelativePosition', 'Gender', 'Age']
# RelativePosition = (depth - position) / depth

stat = ['Click', 'Impression', 'Self' 'TokensLen']

for keywords in feats:
    statistics[keywords] = FeatureStatistic(keywords)

# statistics = joblib.load(feature_statistic)
# profile = load_user_profile()

# f = open(input)
# line = f.readline()
# use this way to compute the length of tokens(how many words a tokens) for every id

# training.txt
with open(input) as f:
    for idx, line in enumerate(f):
        records = line.strip().split("\t")
        # the 0, 1 is click, impression
        click = int(records[0])
        impression = int(records[1])
        depth = int(records[5])
        position = int(records[6])
        relative_position = str(1.0 * (depth - position) / depth)
        for i in range(2, len(records)):
            words = records[i]
            keywords = headers[i]

            if words not in statistics[keywords].statistic:
                statistics[keywords].statistic[words] = [int(0), int(0), int(0), int(0)]

            statistics[keywords].statistic[words][0] += click
            statistics[keywords].statistic[words][1] += impression
            statistics[keywords].statistic[words][2] += 1

        if relative_position not in statistics[RELATIVE_POSITION].statistic:
            statistics[RELATIVE_POSITION].statistic[relative_position] = [int(0), int(0), int(0), int(0)]

        statistics[RELATIVE_POSITION].statistic[relative_position][0] += click
        statistics[RELATIVE_POSITION].statistic[relative_position][1] += impression
        statistics[RELATIVE_POSITION].statistic[relative_position][2] += 1

        if (idx+1)%100000==0:
            print('training: %d' % (idx + 1))
            # break

    print(idx + 1)
f.close()


with open(PATH_USER) as f:
    for idx, line in enumerate(f):
        records = line.strip().split("\t")
        # userid gender age
        if records[0] not in statistics[USER_ID].statistic:
            continue

        user_click = statistics[USER_ID].statistic[records[0]][0]
        user_impression = statistics[USER_ID].statistic[records[0]][1]
        user_self = statistics[USER_ID].statistic[records[0]][2]

        if records[1] not in statistics[GENDER].statistic:
            statistics[GENDER].statistic[records[1]] = [int(0), int(0), int(0), int(0)]
        statistics[GENDER].statistic[records[1]][0] += user_click
        statistics[GENDER].statistic[records[1]][1] += user_impression
        statistics[GENDER].statistic[records[1]][2] += user_self

        if records[2] not in statistics[AGE].statistic:
            statistics[AGE].statistic[records[2]] = [int(0), int(0), int(0), int(0)]
        statistics[AGE].statistic[records[2]][0] += user_click
        statistics[AGE].statistic[records[2]][1] += user_impression
        statistics[AGE].statistic[records[2]][2] += user_self

        if (idx+1)%100000==0:
            print('data_user: %d' % (idx+1))
            # break

    print(idx + 1)
f.close()


def combine_tokens_len(dir, feat):
    with open(dir) as f:
        for idx, line in enumerate(f):
            records = line.strip().split("\t")
            # userid gender age
            if records[0] not in statistics[feat].statistic:
                continue

            words = records[1].strip().split('|')
            words_len = len(words)

            statistics[feat].statistic[records[0]][3] = words_len

            if (idx + 1) % 100000 == 0:
                print('%s: %d' % (feat, idx + 1))
                # break

        print(idx + 1)
    f.close()

combine_tokens_len(PATH_DESCRIPTION_ID, DESCRIPTION_ID)
combine_tokens_len(PATH_KEYWORD_ID, KEYWORD_ID)
combine_tokens_len(PATH_QUERY_ID, QUERY_ID)
combine_tokens_len(PATH_TITLE_ID, TITLE_ID)

print('saving .....')
# joblib.dump(statistics, features_statistic, compress=1)
statistics_db = shelve.open(DATA_PATH + "features_mapping_combined.db", flag='c', writeback=True)
statistics_db['features_statistic'] = statistics
statistics_db.close
print('saved .....')
# for keywords in feats:
#     path = 'data/'+keywords+'_statistic.pkl'
#     print(path)
#     print(sys.getsizeof(statistics[keywords]))
    # with open(path, 'wb') as f:
    #     pkl.dump(statistics[keywords], f)
    #     f.close()

