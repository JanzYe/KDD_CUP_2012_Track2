# -*- coding:utf-8 -*-

# merge mapped features and combined features

from constants import DIR_PATH, headers, TRAIN, TEST

mode = TEST  # train test

if mode == TRAIN:
    mapped = 'data/feature_mapped_combined_training.data'
    combined = DIR_PATH + 'training_combined.txt'
    output = 'data/combined_mapped_training.txt'
elif mode == TEST:
    mapped = 'data/feature_mapped_combined_test.data'
    combined = DIR_PATH + 'test_combined.txt'
    output = 'data/feature_combined_mapped_test.txt'

f_mapped = open(mapped, 'r')
f_combined = open(combined, 'r')
f_output = open(output, 'w')

line_mapped = f_mapped.readline()
line_combined = f_combined.readline()

if mode == TEST:
    f_output.write(','.join(headers[1:]) + '\n')

idx = 0
while line_mapped:
    idx += 1

    if mode == TRAIN:
        words_mapped = line_mapped.strip().split(',')
        # remove first ctr, line_combined already had
        to_write = line_combined.strip('\n') + ',' + ','.join(words_mapped[1:]) + '\n'

    elif mode == TEST:
        to_write = line_combined.strip('\n') + ',' + line_mapped

    f_output.write(to_write)

    line_mapped = f_mapped.readline()
    line_combined = f_combined.readline()

    if idx % 100000 == 0:
        print(idx)
        print(to_write)

