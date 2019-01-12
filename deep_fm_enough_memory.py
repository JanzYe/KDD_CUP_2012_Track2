# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import sp
from deepctr.models import DeepFM
from deepctr.models import MDFM
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import auc
import numpy as np
import operator
import time
from constants import *
from tensorflow.keras import metrics

# total number of records in training.txt is almost 149600000, exactly 149639105

mode = TRAIN  # train valid test

num_train = int(NUM_TRAINING * (10.0 / 11))
# num_train=300000
num_valid = int(NUM_TRAINING * (1.0 / 11))

# chunk_size = 10000000
chunk_size = 360000000
chunk_size_test = 36000000

seed = 1024

# 隐藏层单元数不是越高越好，中间有一个临界值达到最优.
# Dropout在数据量本来就很稀疏的情况下尽量不用，不同的数据集dropout表现差距比较大

embedding_size = 8  # embedding cost most of the memory, if OOM, reduce this
hidden_size = [512, 256, 256, 256, 128]
lr = 0.0005
batch_size = 18000
l2_reg_linear = 0.000001
l2_reg_embedding = 0.000001
l2_reg_deep = 0.001

def process_line(line):
    records = line.strip().split(',')
    y = (float(records[0]))
    x = np.zeros((len(sparse_features)), dtype=np.int)
    i = 0
    for word in records[1:]:
        x[i] = (int(word))
        i += 1
    return x, y


def get_dense_need_combine_feature(data, header, sum_):
    ids = data[header].values
    dense_need_combine_vals = sum_[ids]
    # for i, id in enumerate(ids):
    #     if id >= sparseM.shape[0]:
    #         continue
    #     else:
    #         row = sparseM[int(id)]
    #         if row.count_nonzero() > 0:
    #             dense_need_combine_vals[i] = row.sum() / row.count_nonzero()
    return np.array(dense_need_combine_vals)

def get_dense_need_combine_features(data):
    dense_need_combine_vals = []
    dense_need_combine_vals.append(get_dense_need_combine_feature(data, QUERY_ID, sum_query))
    dense_need_combine_vals.append(get_dense_need_combine_feature(data, KEYWORD_ID, sum_keyword))
    dense_need_combine_vals.append(get_dense_need_combine_feature(data, TITLE_ID, sum_title))
    dense_need_combine_vals.append(get_dense_need_combine_feature(data, DESCRIPTION_ID, sum_description))

    return dense_need_combine_vals

def get_multi_val_feature(data, header, sparseV):
    # rows.nonzero()[1][rows.nonzero()[0]==0]

    # ids = data[header].values
    # cols = features_min_max[max_header][1]
    # rows = len(ids)
    # multi_vals = np.zeros((rows, cols), dtype='int')
    # valid_len = np.ones((rows), dtype='int')
    # for i, id in enumerate(ids):
    #     pos = sparseM[int(id)].nonzero()[1]
    #     if len(pos) < 1:
    #         # many id don't have relative tokens
    #         # print('empty tokens')
    #         continue
    #     valid_len[i] = len(pos)
    #     multi_vals[i, 0:valid_len[i]] = pos

    ids = data[header].values

    valid_len_multi_val = sparseV[ids].toarray()
    valid_len = valid_len_multi_val[:, 0]
    valid_len[valid_len == 0] = 1
    return valid_len_multi_val[:, 1:], valid_len

def get_multi_val_features(data):
    multi_vals = []
    valid_lens = []
    print('combining tokens_vector_query')
    vals, len = get_multi_val_feature(data, QUERY_ID, tokens_multi_val_query)
    multi_vals.append(vals)
    valid_lens.append(len)

    print('combining tokens_vector_keyword')
    vals, len = get_multi_val_feature(data, KEYWORD_ID, tokens_multi_val_keyword)
    multi_vals.append(vals)
    valid_lens.append(len)

    print('combining tokens_vector_title')
    vals, len = get_multi_val_feature(data, TITLE_ID, tokens_multi_val_title)
    multi_vals.append(vals)
    valid_lens.append(len)

    print('combining tokens_vector_description')
    vals, len = get_multi_val_feature(data, DESCRIPTION_ID, tokens_multi_val_description)
    multi_vals.append(vals)
    valid_lens.append(len)

    return multi_vals, valid_lens

def generate_arrays_from_file(path_combined_mapped):
    
    reader = pd.read_csv(path_combined_mapped, chunksize=chunk_size)

    print('generating start')
    for df in reader:
        print('\ndf size: %d' % df.shape[0])
        df = shuffle(df, random_state=seed)
        len_df=df.shape[0]

        if len(multi_val_features_dim) > 0:
            print('combining multi_val features ......')
            multi_vals, valid_lens = get_multi_val_features(df)
        else:
            multi_vals = []
            valid_lens = []

        if len(dense_need_combine_features) > 0:
            print('combining get_dense_need_combine_features ......')
            # sum of idf
            dense_need_combine_val = get_dense_need_combine_features(df)
            # mean of idf
            if len(valid_lens) > 0:
                dense_need_combine_val.extend([dense_need_combine_val[i] / valid_lens[i]
                                               for i in range(len(valid_lens))])
        else:
            dense_need_combine_val = []

        print('scaling features ...')
        for key in features_min_max:
            min_val = features_min_max[key][0]
            max_val = features_min_max[key][1]
            df[key] = (df[key] - min_val) / (max_val - min_val)
            print(df[key].values[0])

        # print(multi_vals)
        x_train = [df[key].values[0:num_train] for key in sparse_features] + \
            [df[key].values[0:num_train] for key in dense_features] + \
            [feat[0:num_train] for feat in dense_need_combine_val] + \
            [vec[0:num_train] for vec in multi_vals] + [val_len[0:num_train] for val_len in valid_lens]
        y_train = [df[key].values[0:num_train] for key in target]
        x_valid = [df[key].values[num_train:] for key in sparse_features] + \
                  [df[key].values[num_train:] for key in dense_features] + \
                  [feat[num_train:] for feat in dense_need_combine_val] + \
                  [vec[num_train:] for vec in multi_vals] + [val_len[num_train:] for val_len in valid_lens]
        y_valid = [df[key].values[num_train:] for key in target]
        return x_train, y_train, x_valid, y_valid
        print('generating complete')


class AucCallback(Callback):
    def __init__(self, clicks, imps, datas, batch_size):
        self.clicks = clicks
        self.imps = imps
        self.datas = datas
        self.batch_size = batch_size
        self.auc = []

    def on_epoch_end(self, epoch, logs=None):
        print('predicting valid ......')

        preds = self.model.predict(self.datas, self.batch_size, verbose=1)

        print('calculating auc ......')
        Auc = auc.scoreClickAUC(self.clicks, self.imps, preds)
        self.auc.append(str(Auc))
        print('scoreClickAUC: %s' % (','.join(self.auc)))

def step_decay(epoch):
        initial_lrate = lr
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate
        
        
if __name__ == "__main__":

    # dense_features = [
    #     #     1         2          3             4        5           6            7
    #     'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
    #     # 8            9             10        12          13
    #     'TitleID', 'DescriptionID', 'UserID', 'Gender', 'Age', 'RelativePosition',
    #
    #     #  14            15                  16               17           18                 19               20
    #     'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',
    #
    #     #  21            22             23              24              25
    #     'aCTR_Ad', 'aCTR_Advertiser', 'aCTR_Depth', 'aCTR_Position', 'aCTR_RPosition',
    #
    #     # 26           27             28             29           30                 31               32
    #     'pCTR_Url', 'pCTR_Ad', 'pCTR_Advertiser', 'pCTR_Query', 'pCTR_Keyword', 'pCTR_Title', 'pCTR_Description',
    #     #  33            34            35               36
    #     'pCTR_User', 'pCTR_Gender', 'pCTR_Age', 'pCTR_RPosition',
    #
    #     #  37            38                39
    #     'num_Depth', 'num_Position', 'num_RPosition',
    #     #  40            41              42            43
    #     'num_Query', 'num_Keyword', 'num_Title', 'num_Description',
    #     # 44                     45                  46               47               48
    #     'num_Imp__Ad', 'num_Imp__Advertiser', 'num_Imp_Depth', 'num_Imp_Position', 'num_Imp_RPosition'
    # ]
    dense_features = [

        'DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID',

        'Depth', 'Position', 'Gender', 'Age', 'RelativePosition',
    #
    #     #  14            15                  16               17           18                 19               20
        'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description', 'group_User',
    #
    #     #  21            22             23              24
    #
    #     # 26           27             28             29           30                 31               32
        'pCTR_Url', 'pCTR_Ad', 'pCTR_Advertiser', 'pCTR_Query', 'pCTR_Keyword', 'pCTR_Title', 'pCTR_Description',
    #     #  33            34            35               36
        'pCTR_User', 'pCTR_Gender', 'pCTR_Age', 'pCTR_RPosition',

        'num_Depth', 'num_Position', 'num_RPosition',

        'num_Query', 'num_Keyword', 'num_Title', 'num_Description',

    ]

    dense_need_combine_features = [
        'sum_Query', 'sum_Keyword', 'sum_Title', 'sum_Description',  # sum of idf
        'mean_Query', 'mean_Keyword', 'mean_Title', 'mean_Description',  # mean of idf
    ]
    # dense_need_combine_features = []

    dense_features_complete = []
    dense_features_complete.extend(dense_features)
    dense_features_complete.extend(dense_need_combine_features)

    min_max = pd.read_csv(PATH_MIN_MAX, dtype=float)
    features_min_max = {key: min_max[key].values for key in [

        'DisplayURL', 'AdID', 'AdvertiserID', 'QueryID', 'KeywordID', 'TitleID', 'DescriptionID',

        'Depth', 'Position', 'Gender', 'Age',
        #
        #     #  14            15                  16               17           18                 19               20
        'group_Ad', 'group_Advertiser', 'group_Query', 'group_Keyword', 'group_Title', 'group_Description',
        'group_User',

        'num_Depth', 'num_Position', 'num_RPosition',

        'num_Query', 'num_Keyword', 'num_Title', 'num_Description',

    ]}

    sparse_features = [
        # 49                50                51             52               53
        'sparse_Url', 'sparse_Ad', 'sparse_Advertiser', 'sparse_Depth', 'sparse_Position',
        # 54                55                   56             57                   58              59
        'sparse_Query', 'sparse_Keyword', 'sparse_Title', 'sparse_Description', 'sparse_UserID', 'sparse_Gender', 'sparse_Age',
        'sparse_PosDepth']

    print('summing tokens_vector_title ......')
    sum_title = pd.read_csv(PATH_SUM_TITLE, header=None, dtype=np.float)[0].values
    print(sum_title.shape)
    print('summing tokens_vector_query ......')
    sum_query = pd.read_csv(PATH_SUM_QUERY, header=None, dtype=np.float)[0].values
    print(sum_query.shape)
    print('summing tokens_vector_keyword ......')
    sum_keyword = pd.read_csv(PATH_SUM_KEYWORD, header=None, dtype=np.float)[0].values
    print('summing tokens_vector_description ......')
    sum_description = pd.read_csv(PATH_SUM_DESCRIPTION, header=None, dtype=np.float)[0].values
    print(sum_description.shape)

    multi_val_features_dim = {
        'vec_Query': [int(features_min_max['num_Query'][1]), N_WORDS_QUERY],
        'vec_Keyword': [int(features_min_max['num_Keyword'][1]), N_WORDS_KEYWORD],
        'vec_Title': [int(features_min_max['num_Title'][1]), N_WORDS_TITLE],
        'vec_Description': [int(features_min_max['num_Description'][1]), N_WORDS_DESCRIPTION],
    }
    # multi_val_features_dim = {}

    print('loading tokens_multi_val ......')
    tokens_multi_val_query = sp.load_npz(PATH_MUL_QUERY)
    tokens_multi_val_title = sp.load_npz(PATH_MUL_TITLE)
    tokens_multi_val_keyword = sp.load_npz(PATH_MUL_KEYWORD)
    tokens_multi_val_description = sp.load_npz(PATH_MUL_DESCRIPTION)

    exclude = ['']

    target = ['Click', 'Impression']


    # 2.count #unique features for each sparse field
    sparse_feature_dim = {}
    with open('./data/features_infos_combined.txt') as fr:
    # with open('./data/sample/features_infos.txt') as fr:
        for line in fr:
            records = line.strip().split(':')
            if records[0] in exclude:
                continue
            sparse_feature_dim[records[0]] = int(records[1])
        fr.close()

    # 4.Define Model,compile and train
    # model = DeepFM({"sparse": sparse_feature_dim, "dense": dense_features}, embedding_size=embedding_size,
    #                hidden_size=hidden_size,
    #                final_activation='sigmoid')

    model = MDFM({"sparse": sparse_feature_dim, "dense": dense_features_complete,
                  'multi_val': multi_val_features_dim},
                embedding_size=embedding_size,
                  hidden_size=hidden_size,
                 final_activation='sigmoid',
                 l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, l2_reg_deep=l2_reg_deep)


    if mode == TRAIN:

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_crossentropy])

        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        filepath = 'model_save/deep_fm_combined-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-bs' + str(batch_size)\
                   + '-ee' + str(embedding_size) + '-hz' + str(hidden_size) \
                   + '-l2l' + str(l2_reg_linear) + '-l2e' + str(l2_reg_embedding) + '-l2d' + str(l2_reg_deep) \
                   + '-t' + now_time + '.h5'
        
        x_train, y_train, x_valid, y_valid = generate_arrays_from_file(PATH_TRAIN)
        
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=True)
        
        auc_eval = AucCallback(y_valid[0], y_valid[1], x_valid, batch_size)
        print((y_valid[0] / y_valid[1]))
        #lrate = LearningRateScheduler(step_decay)

        history = model.fit(x=x_train, y=y_train[0] / y_train[1],
                            steps_per_epoch=int(np.ceil(num_train/batch_size)),
                            callbacks=[checkpoint, auc_eval], epochs=100, verbose=1,
                            validation_data=(x_valid, y_valid[0] / y_valid[1]),
                            validation_steps=int(np.ceil(num_valid/batch_size)), shuffle=True)

    elif mode == VALID:
        # model.load_weights('model_save/deep_fm_fn-ep002-loss0.148-val_loss0.174.h5')  # auc: 0.718467 batch_size=6000
        #model.load_weights('model_save/deep_fm_fn-ep001-loss0.149-val_loss0.175.h5')  # auc: 0.714243  batch_size = 2048
        # model.load_weights('model_save/deep_fm_fn-ep005-loss0.147-val_loss0.173.h5')  # auc: 0.722535  batch_size = 10000
        # model.load_weights('model_save/deep_fm_fn_bs10000-ep001-loss0.155-val_loss0.153.h5')  # auc: 0.738023
        #model.load_weights('model_save/deep_fm_fn_bs15000-ep001-loss0.156-val_loss0.152.h5')  # auc: 0.739935
        #model.load_weights('model_save/deep_fm_fn-ep002-loss0.154-val_loss0.154-bs15000-ee20-hz[128, 128].h5')  # auc: 0.741590
        # model.load_weights('model_save/deep_fm_fn-ep020-loss0.153-val_loss0.153-bs15000-ee20-hz[5, 600].h5')  # auc: 0.742558
        #model.load_weights('model_save/deep_fm_combined-ep001-loss3.077-val_loss0.627-bs6000-ee8-hz[128, 128].h5')  # auc: 0.49
        # model.load_weights('model_save/deep_fm_combined-ep009-loss0.134-val_loss0.134-bs15000-ee20-hz[3, 600].h5')  # auc: 0.876005

        # add pctr, group, len of token, age, depth, position, rposition, num_imp
        #model.load_weights('model_save/deep_fm_combined-ep004-loss0.141-val_loss0.141-bs15000-ee20-hz[3, 600]-t2018-12-18 18:35:59.h5')  # auc: 0.834878

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector
        # model.load_weights('model_save/deep_fm_combined-ep007-loss0.140-val_loss0.140-bs18000-ee20-hz[3, 500]-t2018-12-25 18:44:01.h5')  # auc: 0.837464
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-t2018-12-26 21:32:47.h5')  # auc:838491

        #model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.0001-t2018-12-28 08:39:49.h5')  # auc: 0.839753

        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 20:18:42.h5'  # auc: 0.869702 0.

        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.869080 0.
        model_name = 'deep_fm_combined-ep010-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.874691 0.784256

        model.load_weights('model_save/' + model_name)

        _, _, x_valid, y_valid = generate_arrays_from_file(PATH_TRAIN)

        preds = model.predict(x_valid, batch_size, verbose=1)

        print('calculating auc ......')
        AUC = auc.scoreClickAUC(y_valid[0], y_valid[1], preds)
        print('scoreClickAUC: %f' % AUC)

    elif mode == TEST:
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(now_time)

        # if the loss changes less than 0.001, stops training, does not update model

        # model.load_weights('model_save/deep_fm_fn_bs10000-ep001-loss0.155-val_loss0.153.h5')  # auc: 0.714774
        #model.load_weights('model_save/deep_fm_fn_bs15000-ep001-loss0.156-val_loss0.152.h5')  # auc: 0.717083
        #model.load_weights('model_save/deep_fm_fn-ep002-loss0.154-val_loss0.154-bs15000-ee20-hz[128, 128].h5')  # auc: 0.718581
        #model.load_weights('model_save/deep_fm_fn-ep020-loss0.153-val_loss0.153-bs15000-ee20-hz[5, 600].h5')  # auc: 0.719317
        #model.load_weights('model_save/deep_fm_fn-ep043-loss0.152-val_loss0.152-bs15000-ee20-hz[3, 600].h5')  # auc: 0.722419

        # add dense feature pCTR: over fitting
        # model.load_weights('model_save/deep_fm_combined-ep009-loss0.134-val_loss0.134-bs15000-ee20-hz[3, 600].h5')  # auc: 0.733984
        # model.load_weights('model_save/deep_fm_combined-ep001-loss0.147-val_loss0.139-bs15000-ee20-hz[3, 600].h5')  # auc: 0.744694
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.135-val_loss0.135-bs15000-ee20-hz[3, 600].h5')  # auc: 0.733826
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.132-val_loss0.132-bs15000-ee8-hz[3, 600].h5')  # auc: 0.735597
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.144-val_loss0.135-bs15000-ee8-hz[3, 600].h5')  # auc: 0.743677

        # add pCTR and aCTR
        #model.load_weights('model_save/deep_fm_combined-ep011-loss0.135-val_loss0.135-bs15000-ee20-hz[3, 600].h5')  # auc: 0.738687
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.150-val_loss0.141-bs15000-ee20-hz[3, 600].h5')  # auc: 0.737412
        #model.load_weights('model_save/deep_fm_combined-ep002-loss0.140-val_loss0.139-bs15000-ee20-hz[3, 600].h5')  # auc: 0.736510

        # add pCTR and group
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600].h5')  # auc: 0.746002
        #model.load_weights('model_save/deep_fm_combined-ep001-loss0.165-val_loss0.146-bs15000-ee20-hz[3, 600].h5')  # auc: 0.739382

        # add pCTR and group and len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600].h5')  # auc: 0.748758

        # add pCTR and len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep012-loss0.134-val_loss0.134-bs15000-ee20-hz[3, 600]-t2018-12-17 17:09:17.h5')  # auc: 0.717313
        # model.load_weights('model_save/deep_fm_combined-ep001-loss0.151-val_loss0.141-bs15000-ee20-hz[3, 600]-t2018-12-17 17:09:17.h5')  # auc: 0.739057

        # add len of tokens
        # model.load_weights('model_save/deep_fm_combined-ep013-loss0.153-val_loss0.153-bs15000-ee20-hz[3, 600]-t2018-12-18 00:59:06.h5')  # auc: 0.72

        # add pctr, group, len of token, age, depth, position, rposition
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600]-t2018-12-18 12:48:01.h5')  # auc: 0.749360

        # add pctr, group, len of token, age, depth, position, rposition, num_imp
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.141-val_loss0.141-bs15000-ee20-hz[3, 600]-t2018-12-18 18:35:59.h5')  # auc: 0.739763

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.142-bs15000-ee20-hz[3, 600]-t2018-12-18 23:26:58.h5')  # auc: 0.750127
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.141-val_loss0.141-bs18000-ee20-hz[3, 600]-t2018-12-19 11:20:59.h5')  # auc: 0.753482
        # model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.141-bs18000-ee20-hz[3, 500]-t2018-12-19 16:12:38.h5')  # auc: 0.755629
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.141-val_loss0.141-bs18000-ee20-hz[3, 300]-t2018-12-19 20:46:10.h5')  # auc: 0.751999
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.140-val_loss0.141-bs18000-ee20-hz[3, 400]-t2018-12-20 10:14:14.h5')  # auc: 0.752596
        # model.load_weights('model_save/deep_fm_combined-ep010-loss0.140-val_loss0.140-bs18000-ee25-hz[3, 500]-t2018-12-20 16:05:34.h5')  # auc: 0.753794

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, sum_idf
        # model.load_weights('model_save/deep_fm_combined-ep006-loss0.140-val_loss0.140-bs18000-ee20-hz[3, 500]-t2018-12-21 17:17:20.h5')  # auc: 0.753753

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.142-val_loss0.141-bs18000-ee20-hz[3, 500]-t2018-12-22 23:35:54.h5')  # auc: 0.761872
        # model.load_weights('model_save/deep_fm_combined-ep007-loss0.140-val_loss0.140-bs18000-ee20-hz[3, 500]-t2018-12-25 18:44:01.h5')  # auc: 0.762638

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector, id_val
        #model.load_weights('model_save/deep_fm_combined-ep004-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-t2018-12-26 21:32:47.h5')  # auc: 0.769375 778695
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 128]-l2l1e-05-l2e1e-05-l2d0-t2018-12-27 13:41:15.h5')  # auc: 0.768785 778826
        #model.load_weights('model_save/deep_fm_combined-ep006-loss0.138-val_loss0.138-bs30000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.0-t2018-12-27 17:20:56.h5')  # auc: 0.769178 778902
        #model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d1e-05-t2018-12-27 21:42:03.h5')  # auc: 0.769245 780172
        # model.load_weights('model_save/deep_fm_combined-ep005-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.0001-t2018-12-28 08:39:49.h5')  # auc: 0.771686 780086
        # model.load_weights('model_save/deep_fm_combined-ep004-loss0.139-val_loss0.139-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-05-l2d0.001-t2018-12-28 13:39:25.h5')  # auc: 0.769395 780504
        #model.load_weights('model_save/deep_fm_combined-ep003-loss0.141-val_loss0.141-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e0.0001-l2d0.001-t2018-12-28 19:11:03.h5')  # auc: 0.758987 0.770800
        # model.load_weights('model_save/deep_fm_combined-ep002-loss0.134-val_loss0.138-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e0.0-l2d0.001-t2018-12-28 22:40:46.h5')  # auc: 0.759393 0.770754
        #model.load_weights('model_save/deep_fm_combined-ep006-loss0.137-val_loss0.137-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-05-l2e1e-06-l2d0.001-t2018-12-29 07:15:55.h5')  # auc: 0.775155 0.781973
        #model.load_weights('model_save/deep_fm_combined-ep013-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-29 12:56:43.h5')  # auc: 0.774949 0.783688
        # model.load_weights('model_save/deep_fm_combined-ep009-loss0.117-val_loss0.124-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l0.0-l2e1e-06-l2d0.001-t2018-12-30 00:45:54.h5')  # auc: 0.763515 0.772781
        # model.load_weights('model_save/deep_fm_combined-ep014-loss0.122-val_loss0.123-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-07-l2e1e-06-l2d0.001-t2018-12-30 09:43:53.h5')  # auc: 0.769962 0.778210

        # add group, len of token, age, depth, position, rposition, num_occurs, tokens_vector
        #model.load_weights('model_save/deep_fm_combined-ep004-loss0.150-val_loss0.151-bs18000-ee20-hz[3, 500]-t2018-12-26 10:14:01.h5')  # auc: 0.742817

        # NEW DATA
        # model_name = 'deep_fm_combined-ep013-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 00:27:56.h5'  # auc: 0.773103 0.782441
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774968 0.785623
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774931 0.785993
        # model_name = 'deep_fm_combined-ep005-loss0.129-val_loss0.128-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5'  # auc: 0.774381 0.785257
        # new data
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 22:51:48.h5'  # auc: 0.773548 0.783240
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 22:51:48.h5'  # auc: 0.774124 0.784411
        # new data
        # model_name = 'deep_fm_combined-ep009-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 11:52:39.h5'  # auc: 0.772808 0.783461
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 11:52:39.h5'  # auc: 0.773789 0.784705
        # new data
        # model_name = 'deep_fm_combined-ep011-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772387 0.781246
        # model_name = 'deep_fm_combined-ep008-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772418 0.782084
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5'  # auc: 0.772749 0.782517
        # new data
        # model_name = 'deep_fm_combined-ep010-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 10:21:56.h5'  # auc: 0.774124 0.785176
        # model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 10:21:56.h5'  # auc: 0.775915 0.786094
        # new data
        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 20:18:42.h5'  # auc: 0.773678 0.784004
        # whole data, stop at epoch 6 base on exp
        # model_name = '.h5'  # auc: 0. 0.

        # add pctr, group, len of token, age, depth, position, rposition, num_occurs, tokens_vector, id_val, sum_mean_idf
        model_name = 'deep_fm_combined-ep010-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.774472 0.784256
        #model_name = 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-03 03:45:50.h5'  # auc: 0.773730 0.784448


        model.load_weights('model_save/' + model_name)

        ctr = []
        click = []
        imp = []
        reader = pd.read_csv(PATH_SOLUTION, chunksize=chunk_size_test)
        for df in reader:
            print('df size: %d' % df.shape[0])

            cnt = 0
            while cnt < df.shape[0]:
                end = cnt + batch_size
                if end > df.shape[0]:
                    end = df.shape[0]
                batch = df[cnt:end]

                # Y = np.array(batch['clicks'].values, dtype=float) / np.array(batch['impressions'].values, dtype=float)
                # ctr.extend(Y)
                click.extend(batch['clicks'].values)
                imp.extend(batch['impressions'].values)

                cnt += batch_size

                if cnt % (batch_size * 100):
                    print(click[0])

        preds = []
        # labels = ctr
        reader = pd.read_csv(PATH_TEST, chunksize=chunk_size_test)
        idf = 0
        for df in reader:
            len_df = df.shape[0]
            print('\ndf size: %d' % len_df)
            ctr_chunk = ctr[idf:idf+len_df]

            # 这样打乱测试集没有提升效果
            #random_state = random.randint(0, chunk_size)
            #ctr_chunk = shuffle(ctr_chunk, random_state=random_state)
            #data = shuffle(df, random_state=random_state)
            data = df
            #labels.extend(ctr_chunk)
            idf += len_df

            cnt = 0

            if len(multi_val_features_dim) > 0:
                print('combining multi_val features ......')
                multi_vals, valid_lens = get_multi_val_features(df)
            else:
                multi_vals = []
                valid_lens = []

            if len(dense_need_combine_features) > 0:
                print('combining get_dense_need_combine_features ......')
                # sum of idf
                dense_need_combine_val = get_dense_need_combine_features(df)
                # mean of idf
                if len(valid_lens) > 0:
                    dense_need_combine_val.extend([dense_need_combine_val[i] / valid_lens[i]
                                                   for i in range(len(valid_lens))])
            else:
                dense_need_combine_val = []


            print('scaling features ...')
            for key in features_min_max:
                min_val = features_min_max[key][0]
                max_val = features_min_max[key][1]
                data[key] = np.clip((data[key] - min_val) / (max_val - min_val), 0, 1)
                print(data[key].values[0])

            while cnt < len_df:
                end = cnt + batch_size
                if end > len_df:
                    end = len_df
                batch = data[cnt:end]

                X = [batch[key].values for key in sparse_features] + \
                    [batch[key].values for key in dense_features] + \
                    [feat[cnt: end] for feat in dense_need_combine_val] + \
                    [vec[cnt: end] for vec in multi_vals] + [len[cnt: end] for len in valid_lens]

                pred = model.predict_on_batch(X)
                preds.extend(pred.flatten().tolist())

                cnt += batch_size

                if cnt % (batch_size * 100):
                    print(pred[0])

        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(now_time)

        print('calculating auc ......')

        print('labels； %d' % len(click))
        print('preds: %d' % len(preds))
        AUC = auc.auc(np.array(click, dtype=np.float) / np.array(imp, dtype=np.float), preds)
        print('auc: %f' % AUC)
        AUC = auc.scoreClickAUC(click, imp, preds)
        print('scoreClickAUC: %f' % AUC)

        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(now_time)

        # writing preds to csv
        with open('data/' + model_name + '.csv', 'w') as fw:
            for i in range(len(preds)):
                if i % 1000000 == 0:
                    print('label: %f, pred: %f' % (click[i]/imp[i], preds[i]))
                to_write = str(preds[i]) + '\n'
                fw.write(to_write)
            fw.close()

        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(now_time)

    print("demo done")
