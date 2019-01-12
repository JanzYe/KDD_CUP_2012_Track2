# -*- coding:utf-8 -*-

import load_user_profile

dir_path = '/home/yezhizi/Documents/python/2018DM_Project/track2/'

def sample(profile):
    # file = open(dir_path + "training.txt")
    # combine = open(dir_path+"training_combined.txt", 'w')

    file = open(dir_path + "test.txt")
    combine = open(dir_path + "test_combined.txt", 'w')

    not_exist_user_id = open(dir_path+"not_exist_user_id.txt", 'a')


    i = 1
    for line in file:
        # file_train.write(line)
        record = line.strip().split('\t')
        to_write = ('\t'.join((record[0:-1])))
        if record[-1] not in profile:
            print('no userid: %s' % record[-1])
            not_exist_user_id.write(record[-1]+'\n')
            # 0 denotes unknown
            # gender； 0, 1, 2
            # age: 0, 1, 2, 3, 4, 5, 6
            profile[record[-1]] = str('0\t0')
        to_write = to_write + '\t' + profile[record[-1]] + '\n'

        if i % 100000 == 0:
            print(i)
            print(to_write)
        combine.write(to_write)

        i = i + 1

    file.close()
    combine.close()


if __name__ == '__main__':
    profile = load_user_profile.load()
    # profile = {'490234': '1,2'}
    sample(profile)