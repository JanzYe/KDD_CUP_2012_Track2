# -*- coding:utf-8 -*-

#  data_descriptionid, data_purchasedkeywordid, data_queryid, data_titleid are not need to mapping, because every records are distinct,
#  and their id can consider as the value after mapping

#

dir_path = '/home/yezhizi/Documents/2018DM_Project/track2/'
data_descriptionid = dir_path + 'descriptionid_tokensid.txt'
data_purchasedkeywordid = dir_path + 'purchasedkeywordid_tokensid.txt'
data_queryid = dir_path + 'queryid_tokensid.txt'
data_titleid = dir_path + 'titleid_tokensid.txt'

out_descriptionid = 'data/descriptionid_mapped.txt'
out_purchasedkeywordid = 'data/purchasedkeywordid_mapped.txt'
out_queryid = 'data/queryid_mapped.txt'
out_titleid = 'data/titleid_mapped.txt'



def mapping(fr, fw):
    features = {}
    feature_val = 1
    for idx, line in enumerate(fr):
        if idx % 100000 == 0:
            print(idx)
        records = line.strip().split("\t")
        # the 0 is id
        if records[1] not in features:
            features[records[1]] = str(feature_val)
            feature_val = feature_val + 1
        to_write = records[0] + ',' + features[records[1]] + '\n'
        fw.write(to_write)

    fr.close()
    fw.close()


with open(data_titleid) as fr, open(out_titleid, 'w') as fw:
    mapping(fr, fw)

with open(data_descriptionid) as fr, open(out_descriptionid, 'w') as fw:
    mapping(fr, fw)

with open(data_purchasedkeywordid) as fr, open(out_purchasedkeywordid, 'w') as fw:
    mapping(fr, fw)

with open(data_queryid) as fr, open(out_queryid, 'w') as fw:
    mapping(fr, fw)