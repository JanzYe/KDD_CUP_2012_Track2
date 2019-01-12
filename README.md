# The Data Mining Project
#### data: KDD CUP 2012 Track2

# Running Steps

#### pre-processing
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

(11) python deep_fm_combined.py or python deep_fm_enough_memory.py

##### changing the parameters in file to deal with training and test data, and training more models

(12) python auc.py

## Models List

|Model|Paper|
|:--:|:--|
|Factorization-supported Neural Network|[ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)|
|Product-based Neural Network|[ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)|
|Wide & Deep|[arxiv 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)|
|DeepFM|[IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)|
|Piece-wise Linear Model|[arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)|
|Deep & Cross Network|[ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)|
|Attentional Factorization Machine|[IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435)|
|Neural Factorization Machine|[SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)|
|Deep Interest Network|[KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)|