# -*- coding:utf-8 -*-

import load_user_profile

dir_path = '/home/yezhizi/Documents/2018DM_Project/track2/'

# 采样
def sample(profile):
    file_train = open(dir_path + "training.txt")
    file_test = open(dir_path + "test.txt")
    sample_train = open("data/sample/training.txt", 'w')
    sample_valid = open("data/sample/validation.txt", 'w')
    sample_test = open("data/sample/test.txt", 'w')
    header = 'CTR,DisplayURL,AdID,AdvertiserID,Depth,Position,QueryID,KeywordID,TitleID,DescriptionID,Gender,Age\n'
    sample_train.write(header)
    sample_valid.write(header)
    sample_test.write(header)

    i = 0
    for line in file_train:
        # file_train.write(line)
        record = line.strip().split('\t')
        ctr = float(record[0]) / float(record[1])
        to_write = str(ctr) + ',' + (','.join((record[2:-1])))
        if record[-1] not in profile:
            continue
        to_write = to_write + ',' + profile[record[-1]] + '\n'

        if i % 100000 == 0 :
            print(i)
            print(to_write)
        if i < 10000000:
            sample_train.write(to_write)
        elif i >= 10000000 and i < 13000000:
            sample_valid.write(to_write)
        else:
            sample_test.write(to_write)
        if i > 15000000 :    # 多了一行数据
            file_train.close()
            sample_train.close()
            sample_valid.close()
            sample_test.close()
            break
        i = i + 1


if __name__ == '__main__':
    profile = load_user_profile.load()
    # profile = {'490234': '1,2'}
    sample(profile)