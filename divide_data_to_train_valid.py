# -*- coding:utf-8 -*-

import random
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from constants import headers, NUM_TRAINING
import  time

# dir_path = 'data/sample/feature_mapped.data'
# dir_path = 'data/mapped_combined_training.data'
# dir_path = 'data/combined_mapped_training.txt'
dir_path = 'data/combined_mapped_shuffled_training.txt'
new_train = "data/feature_mapped_combined_train.data"
new_valid = "data/feature_mapped_combined_valid.data"

# randomly select about 1/11 data to be valid data
# shuffle the whole training file by linux cmd shuf
# shuf /home/yezhizi/Documents/python/2018DM_Project/DeepCTR/CTR/data/combined_mapped_training.txt
# -o /home/yezhizi/Documents/python/2018DM_Project/DeepCTR/CTR/data/combined_mapped_shuffled_training.txt

def divide_and_shuffle_by_readlines():
    print('loading ......')
    training = open(dir_path, 'r')
    sample_train = open(new_train, 'w')
    sample_valid = open(new_valid, 'w')
    sample_train.write(','.join(headers) + '\n')
    sample_valid.write(','.join(headers) + '\n')

    #lines = training.readlines(50000000)
    print('random ......')
    total = NUM_TRAINING
#     total = 100000
    print(total)
    population = set(np.arange(total))
    valid_len = int(total / 11)
    valid_indexs = random.sample(population, valid_len)
    #train_indexs = population.difference(valid_indexs)

    valid_indexs = list(valid_indexs)
    print('sorting ......')
    valid_indexs = sorted(valid_indexs)
    # train_indexs = list(train_indexs)

    index_train = 0
    index_valid = 0

    print('sampling ......')
    for index, line in enumerate(training):
        if index == 0:
            continue
        if (index_valid < valid_len) and (index-1 == valid_indexs[index_valid]):
            sample_valid.write(line)
            index_valid += 1

        else:
            sample_train.write(line)
            index_train += 1

        if (index+1) % 100000 == 0:
            print('total: %d, train: %d, valid: %d' %(index, index_train, index_valid))
            print(line)
            
#             break

    sample_train.close()
    sample_valid.close()

if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    divide_and_shuffle_by_readlines()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

