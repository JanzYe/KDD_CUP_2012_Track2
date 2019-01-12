import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.models import WDL
from tensorflow.keras.callbacks import ModelCheckpoint
import auc

mode = 'train'  # train pred test
batch_size = 2048

if __name__ == "__main__":

    data = pd.read_csv("./data/sample/training.txt")

    headers = ['CTR', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']

    sparse_features = ['DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position', 'QueryID', 'KeywordID',
                       'TitleID', 'DescriptionID', 'Gender', 'Age']
    target = ['CTR']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    sparse_feature_dim = {feat: data[feat].nunique()
                          for feat in sparse_features}
    # 3.generate input data for model
    model_input = [data[feat].values for feat in sparse_feature_dim]

    if mode == 'train':
        # 4.Define Model,compile and train
        model = WDL({"sparse": sparse_feature_dim, "dense": []},
                       final_activation='sigmoid')

        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

        filepath = 'model_save/wdl_sample-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(model_input, data[target].values, callbacks=[checkpoint],
                            batch_size=batch_size, epochs=50, verbose=1, validation_split=0.2,)

    elif mode == 'test':
        model = WDL({"sparse": sparse_feature_dim, "dense": []},
                       final_activation='sigmoid')
        model.load_weights('model_save/wdl_sample-ep001-loss0.184-val_loss0.172.h5')

        # model = load_model('model_save/deep_fm_sample-ep001-loss0.192-val_loss0.176.h5')

        data = pd.read_csv("./data/sample/validation.txt")

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        # 2.count #unique features for each sparse field
        sparse_feature_dim = {feat: data[feat].nunique()
                              for feat in sparse_features}
        # 3.generate input data for model
        model_input = [data[feat].values for feat in sparse_feature_dim]

        pred = model.predict(model_input, batch_size, 1)
        label = data[target].values.flatten().tolist()
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
