import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import auc

mode = 'test'  # train pred test
batch_size = 2048

if __name__ == "__main__":

    headers = ['CTR', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID']

    sparse_features = ['DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID']
    target = ['CTR']


    # 2.count #unique features for each sparse field
    sparse_feature_dim = {}
    with open('./data/sample/features_infos.txt') as fr:
        for line in fr:
            records = line.strip().split(':')
            sparse_feature_dim[records[0]] = int(records[1])
        fr.close()

    # 3.generate input data for model
    # label = []
    # model_input = []
    # with open('./data/sample/feature_mapped.data') as fr:
    #     for line in fr:
    #         records = line.strip().split(',')
    #         label.append(float(records[0]))
    #         values = []
    #         for word in records[1:]:
    #             values.append(int(word))
    #         model_input.append(values)
    #     fr.close()

    data = pd.read_csv("./data/sample/feature_mapped.data", header=None)
    label = data[0].values
    model_input = [data[feat+1].values for feat in range(len(sparse_feature_dim))]

    if mode == 'train':
        # 4.Define Model,compile and train
        model = DeepFM({"sparse": sparse_feature_dim, "dense": []},
                       final_activation='sigmoid')

        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

        filepath = 'model_save/deep_fm_sample-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(model_input, label, callbacks=[checkpoint],
                            batch_size=batch_size, epochs=50, verbose=1, validation_split=0.2,)

    elif mode == 'test':
        model = DeepFM({"sparse": sparse_feature_dim, "dense": []},
                       final_activation='sigmoid')
        model.load_weights('model_save/deep_fm_sample-ep002-loss0.175-val_loss0.171.h5')

        # model = load_model('model_save/deep_fm_sample-ep001-loss0.192-val_loss0.176.h5')

        data = pd.read_csv("./data/sample/feature_mapped.data", header=None)
        label = data[0].values
        model_input = [data[feat + 1].values for feat in range(len(sparse_feature_dim))]

        pred = model.predict(model_input, batch_size, 1)
        label = label.flatten().tolist()
        pred = pred.flatten().tolist()
        with open('data/pctr', 'w') as fw:
            for i in range(len(pred)):
                if i % 10000 == 0:
                    print('label: %f, pred: %f' % (label[i], pred[i]))
                to_write = str(i+1)+','+str(label[i])+','+str(pred[i])+'\n'
                fw.write(to_write)
            fw.close()
        AUC = auc.auc(label, pred)
        print('auc: %f' % AUC)

    print("demo done")
