import pandas as pd
from sklearn.utils import shuffle
from deepctr.models import DeepFM
from tensorflow.keras.callbacks import ModelCheckpoint
import auc
import numpy as np
import operator
import random

# total number of records in training.txt is almost 149600000, exactly 149639105

mode = 'pred'  # train pred test

num_train = 149639105 * (10.0 / 11)
num_valid = 149639105 * (1.0 / 11)

chunk_size = 10000000

# 隐藏层单元数不是越高越好，中间有一个临界值达到最优.
# Dropout在数据量本来就很稀疏的情况下尽量不用，不同的数据集dropout表现差距比较大
batch_size = 15000
embedding_size = 20  # 8
hidden_size = [3, 600]

def process_line(line):
    records = line.strip().split(',')
    y = (float(records[0]))
    x = np.zeros((len(sparse_features)), dtype=np.int)
    i = 0
    for word in records[1:]:
        x[i] = (int(word))
        i += 1
    return x, y

#
# def generate_arrays_from_file(path, batch_size):
#     f = open(path)
#     cnt = 0
#     X = np.zeros((batch_size, len(sparse_features)), dtype=np.int)
#     Y = np.zeros((batch_size))
#     while 1:
#
#         for line in f:
#             # create Numpy arrays of input data
#             # and labels, from each line in the file
#             x, y = process_line(line)
#             X[cnt] = (x)
#             Y[cnt] = (y)
#             cnt += 1
#             if cnt == batch_size:
#                 cnt = 0
#                 X = X.T
#                 # yield X, Y
#                 #yield ([X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]], Y)
#                 yield ([X[0], X[1], X[2], X[3], X[4], X[6], X[7], X[8]], Y)
#                 X = np.zeros((batch_size, len(sparse_features)), dtype=np.int)
#                 Y = np.zeros((batch_size))
#     f.close()


def generate_arrays_from_file(path, batch_size):
    while 1:
        reader = pd.read_csv(path, header=None, chunksize=chunk_size)

        print('epoch start')
        for df in reader:
            print('\ndf size: %d' % df.shape[0])
            df = shuffle(df)
            len_df=df.shape[0]
            cnt = 0
            while cnt < len_df:
                end = cnt+batch_size
                if end > len_df:
                    end = len_df
                    cnt = end - batch_size
                batch = df[cnt: end]
                # X = [batch[key+1].values for key in range(len(sparse_features))]
                X = [batch[key+1].values for key in np.hstack((np.arange(0, 6), np.arange(6, len(sparse_feature_dim))))]
                Y = batch[0].values
                yield (X, Y)
                cnt += batch_size
        print('epoch complete')



if __name__ == "__main__":

    headers = ['CTR', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']

    sparse_features = ['DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']
    exclude = ['']

    target = ['CTR']


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
    model = DeepFM({"sparse": sparse_feature_dim, "dense": []}, embedding_size=embedding_size,
                   hidden_size=hidden_size,
                   final_activation='sigmoid')

    if mode == 'train':

        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

        filepath = 'model_save/deep_fm_fn-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-bs' + str(batch_size)\
                   + '-ee' + str(embedding_size) + '-hz' + str(hidden_size) + '.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=True)

        #history = model.fit_generator(generate_arrays_from_file('./data/sample/feature_mapped_valid.data', batch_size=batch_size),
        #            steps_per_epoch=int(np.ceil(num_train/batch_size)), callbacks=[checkpoint], epochs=50, verbose=1,
        #          validation_data=generate_arrays_from_file('./data/sample/feature_mapped_valid.data', batch_size=batch_size),
        #                             validation_steps=int(np.ceil(num_valid/batch_size)))
        history = model.fit_generator(generate_arrays_from_file('./data/feature_mapped_combined_train.data', batch_size=batch_size),
                   steps_per_epoch=int(np.ceil(num_train/batch_size)), callbacks=[checkpoint], epochs=50, verbose=1,
                 validation_data=generate_arrays_from_file('./data/feature_mapped_combined_valid.data', batch_size=batch_size),
                                     validation_steps=int(np.ceil(num_valid/batch_size)))

    elif mode == 'test':
        # model.load_weights('model_save/deep_fm_fn-ep002-loss0.148-val_loss0.174.h5')  # auc: 0.718467 batch_size=6000
        #model.load_weights('model_save/deep_fm_fn-ep001-loss0.149-val_loss0.175.h5')  # auc: 0.714243  batch_size = 2048
        # model.load_weights('model_save/deep_fm_fn-ep005-loss0.147-val_loss0.173.h5')  # auc: 0.722535  batch_size = 10000
        # model.load_weights('model_save/deep_fm_fn_bs10000-ep001-loss0.155-val_loss0.153.h5')  # auc: 0.738023
        #model.load_weights('model_save/deep_fm_fn_bs15000-ep001-loss0.156-val_loss0.152.h5')  # auc: 0.739935
        #model.load_weights('model_save/deep_fm_fn-ep002-loss0.154-val_loss0.154-bs15000-ee20-hz[128, 128].h5')  # auc: 0.741590
        model.load_weights('model_save/deep_fm_fn-ep020-loss0.153-val_loss0.153-bs15000-ee20-hz[5, 600].h5')  # auc: 0.742558

        labels = []
        preds = []

        reader = pd.read_csv("./data/feature_mapped_combined_valid.data", header=None, chunksize=chunk_size)

        for df in reader:
            print('df size: %d' % df.shape[0])
            df = shuffle(df)
            cnt = 0
            while cnt < df.shape[0]:
                end = cnt + batch_size
                if end > df.shape[0]:
                    end = df.shape[0]
                batch = df[cnt:end]
                X = [batch[key + 1].values for key in np.hstack((np.arange(0, 6), np.arange(6, len(sparse_feature_dim))))]
                labels.extend(batch[0].values)

                pred = model.predict_on_batch(X)
                preds.extend(pred.flatten().tolist())

                cnt += batch_size

                if cnt % (batch_size * 100):
                    print(pred[0])

        print('calculating auc ......')
        AUC = auc.auc(labels, preds)
        print('auc: %f' % AUC)

    elif mode == 'pred':
        # model.load_weights('model_save/deep_fm_fn_bs10000-ep001-loss0.155-val_loss0.153.h5')  # auc: 0.714774
        #model.load_weights('model_save/deep_fm_fn_bs15000-ep001-loss0.156-val_loss0.152.h5')  # auc: 0.717083
        #model.load_weights('model_save/deep_fm_fn-ep002-loss0.154-val_loss0.154-bs15000-ee20-hz[128, 128].h5')  # auc: 0.718581
        #model.load_weights('model_save/deep_fm_fn-ep020-loss0.153-val_loss0.153-bs15000-ee20-hz[5, 600].h5')  # auc: 0.719317
        model.load_weights('model_save/deep_fm_fn-ep043-loss0.152-val_loss0.152-bs15000-ee20-hz[3, 600].h5')  # auc: 0.722419

        ctr = []
        reader = pd.read_csv('/home/yezhizi/Documents/python/2018DM_Project/track2/KDD_Track2_solution.csv',
                             chunksize=chunk_size)
        for df in reader:
            print('df size: %d' % df.shape[0])

            cnt = 0
            while cnt < df.shape[0]:
                end = cnt + batch_size
                if end > df.shape[0]:
                    end = df.shape[0]
                batch = df[cnt:end]

                Y = np.array(batch['clicks'].values, dtype=float) / np.array(batch['impressions'].values, dtype=float)

                ctr.extend(Y)

                cnt += batch_size

                if cnt % (batch_size * 100):
                    print(Y[0])

        preds = []
        labels = []
        reader = pd.read_csv("./data/feature_mapped_combined_test.data", header=None, chunksize=chunk_size)
        idf = 0
        for df in reader:
            len_df = df.shape[0]
            print('\ndf size: %d' % len_df)
            ctr_chunk = ctr[idf:idf+len_df]

            # 这样打乱测试集没有提升效果
            random_state = random.randint(0, chunk_size)
            ctr_chunk = shuffle(ctr_chunk, random_state=random_state)
            data = shuffle(df, random_state=random_state)
            labels.extend(ctr_chunk)
            idf += len_df

            cnt = 0
            while cnt < len_df:
                end = cnt + batch_size
                if end > len_df:
                    end = len_df
                batch = data[cnt:end]
                X = [batch[key].values for key in np.hstack((np.arange(0, 6), np.arange(6, len(sparse_feature_dim))))]

                pred = model.predict_on_batch(X)
                preds.extend(pred.flatten().tolist())

                cnt += batch_size

                if cnt % (batch_size * 100):
                    print(pred[0])


            # with open('data/pctr', 'w') as fw:
            #     for i in range(len(preds)):
            #         if i % 10000 == 0:
            #             print('label: %f, pred: %f' % (labels[i], preds[i]))
            #         to_write = str(i + 1) + ',' + str(labels[i]) + ',' + str(preds[i]) + '\n'
            #         fw.write(to_write)
            #     fw.close()

        print('calculating auc ......')

        print('labels； %d' % len(labels))
        print('preds: %d' % len(preds))
        AUC = auc.auc(labels, preds)
        print('auc: %f' % AUC)

    print("demo done")
