# -*- coding = utf-8 -*-
from constants import *
dir_path = DIR_PATH
data_user = dir_path + 'userid_profile.txt'

def load():
    fr = open(data_user)
    print('Loading user profile ...')
    profile = {}
    for idx, line in enumerate(fr):
        if idx % 100000 == 0:
            print(idx)
        records = line.strip().split('\t')
        # the 0 is id
        if idx == 11184020:
            print(records)
        # profile[records[0]] = str(records[1]+'\t'+records[2])
        profile[records[0]] = [int(records[1]), int(records[2])]
    return profile
