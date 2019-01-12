# -*- coding:utf-8 -*-

import load_user_profile

dir_path = '/home/yezhizi/Documents/2018DM_Project/track2/'

# 采样
def sample():
    file_train = open(dir_path + "training.txt")
    file_test = open(dir_path + "test.txt")
    sample_train = open("data/sample/training_no_combine.txt", 'w')
    sample_valid = open("data/sample/validation_no_combine.txt", 'w')

    i = 0
    for line in file_train:
        # file_train.write(line)
        to_write = line
        if i % 100000 == 0 :
            print(i)
            print(to_write)
        if i < 10000000:
            sample_train.write(to_write)
        elif i >= 10000000 and i < 13000000:
            sample_valid.write(to_write)

        if i > 13000000 :    # 多了一行数据
            file_train.close()
            sample_train.close()
            sample_valid.close()
            break
        i = i + 1


if __name__ == '__main__':
    sample()