# The Data Mining Project
## PREDICT THE CLICK-THROUGH RATE(CTR) OF ADS GIVEN THE QUERY AND USER INFORMATION
#### Data: KDD CUP 2012 Track2

# Running Steps

#### Pre-processing
(1) python feature_statistic.py

(2) python features_min_max.py // for dense features

(3) python construct_mapping_fn.py  // for sparse features

(4) python count_features_fn.py  // for sparse features

(5) python tokens_vector.py

(6) python sum_idf.py

(7) python construct_tokens_vectors.py 

(8) python shuffle_big_file.py or python shuffle_file_enough_memory.py  

(9) python combine_data.py train/test

(10) python divide_data_to_train_valid.py

#### Training

(11) python deep_fm_combined.py or python deep_fm_enough_memory.py

##### changing the parameters in file to deal with training and test data, and training more models

(12) python auc.py

##### more details: see the documents in directory "doc"

####Model
Modified the deepfm which is implemented by Weichen Shen,wcshen1994@163.com in DeepCTR
##### more details: see the contents in directory "deepctr"
