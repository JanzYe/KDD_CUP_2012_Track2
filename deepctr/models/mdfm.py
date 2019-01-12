# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction(https://arxiv.org/abs/1703.04247)

Modified by JanzYe

"""


from tensorflow.python.keras.layers import Dense, Embedding, Concatenate, Reshape, Flatten, add, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
import numpy as np
import tensorflow as tf

from ..utils import get_input_multi
from ..layers import PredictionLayer, MLP, FM
from deepctr.sequence import SequencePoolingLayer

def MDFM(feature_dim_dict, embedding_size=8,
           use_fm=True, hidden_size=[128, 128], l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_deep=0,
           init_std=0.0001, seed=1024, keep_prob=1, activation='relu', final_activation='sigmoid', use_bn=False):
    """Instantiates the DeepFM Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field, dense, multi_val field like
            {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5'], 'multi_val': {'field_6':4,'field_7':3} }
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param use_bn: bool. Whether use BatchNormalization before activation or not.in deep net
    :return: A Keras model instance.
    """
    if not isinstance(feature_dim_dict,
                      dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim_dict must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
    if not isinstance(feature_dim_dict["sparse"], dict):
        raise ValueError("feature_dim_dict['sparse'] must be a dict,cur is", type(
            feature_dim_dict['sparse']))
    if not isinstance(feature_dim_dict["dense"], list):
        raise ValueError("feature_dim_dict['dense'] must be a list,cur is", type(
            feature_dim_dict['dense']))

    sparse_input, dense_input, multi_val_input, valid_len_input = get_input_multi(feature_dim_dict, None)
    sparse_embedding, linear_embedding, = get_embeddings(
        feature_dim_dict, 'sparse', embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]
    linear_term = [linear_embedding[i](sparse_input[i])
                   for i in range(len(sparse_input))]
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) > 0:
        linear_term = linear_term[0]
    else:
        linear_term = 0

    if len(dense_input) > 0:
        continuous_embedding_list = list(
            map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding), ),
                dense_input))
        continuous_embedding_list = list(
            map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list

        dense_input_ = dense_input[0] if len(
            dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg_linear))(dense_input_)
        linear_term = add([linear_dense_logit, linear_term])


    if len(multi_val_input) > 0:
        # multi_val_embedding_list = [[(multi_val_input[i]) for i in range(len(multi_val_input))]]

        # multi_val_embedding_list, multi_val_linear_embedding = get_pooling(multi_val_input, feature_dim_dict, embedding_size,
        #                                        init_std, seed, l2_reg_embedding, l2_reg_deep)

        multi_val_embedding, multi_val_linear_embedding, = get_embeddings_multi_val(
            feature_dim_dict, 'multi_val', embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

        multi_val_embedding_list = [multi_val_embedding[i](multi_val_input[i])
                                    for i in range(len(multi_val_input))]
        multi_val_linear_embedding_list = [multi_val_linear_embedding[i](multi_val_input[i])
                                    for i in range(len(multi_val_input))]

        multi_val_pooling = [SequencePoolingLayer(seq_len_max=feature_dim_dict['multi_val'][feat][0], mode='mean',
                                                        name='multi_val_pool_' + str(i) + '-' + feat)
                                  for i, feat in enumerate(feature_dim_dict['multi_val'])]
        multi_val_pooling_list = [multi_val_pooling[i]([multi_val_embedding_list[i], valid_len_input[i]])
                                  for i in range(len(valid_len_input))]

        multi_val_linear_pooling_list = [SequencePoolingLayer(seq_len_max=feature_dim_dict['multi_val'][feat][0], mode='mean',
                                                        name='linear_pool_' + str(i) + '-' + feat)
                                  ([multi_val_linear_embedding_list[i], valid_len_input[i]])
                                  for i, feat in enumerate(feature_dim_dict['multi_val'])]

        embed_list += multi_val_pooling_list

        if len(multi_val_linear_pooling_list) > 1:
            multi_val_linear_pooling_list = add(multi_val_linear_pooling_list)
        elif len(multi_val_linear_pooling_list) > 0:
            multi_val_linear_pooling_list = multi_val_linear_pooling_list[0]
        else:
            multi_val_linear_pooling_list = 0

        linear_term = add([multi_val_linear_pooling_list, linear_term])


    fm_input = Concatenate(axis=1)(embed_list)
    deep_input = Flatten()(fm_input)
    fm_out = FM()(fm_input)
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   use_bn, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)

    if len(hidden_size) == 0 and use_fm == False:  # only linear
        final_logit = linear_term
    elif len(hidden_size) == 0 and use_fm == True:  # linear + FM
        final_logit = add([linear_term, fm_out])
    elif len(hidden_size) > 0 and use_fm == False:  # linear +　Deep
        final_logit = add([linear_term, deep_logit])
    elif len(hidden_size) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = add([linear_term, fm_out, deep_logit])
    else:
        raise NotImplementedError

    # output = PredictionLayer(final_activation)(final_logit)
    output = Lambda(pred)([final_activation, final_logit])
    input_all = sparse_input + dense_input + multi_val_input + valid_len_input
    # for i in range(len(valid_len_input)):
    #     input_all.append(multi_val_input[i])
    #     input_all.append(valid_len_input[i])
    model = Model(inputs=input_all, outputs=output)
    return model


def pred(args):
    output_layer = PredictionLayer(args[0])
    output = output_layer(args[1])
    # output = tf.squeeze(output)
    return output

def get_embeddings(feature_dim_dict, key, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(feature_dim_dict[key][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name=key+'_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict[key])]
    linear_embedding = [Embedding(feature_dim_dict[key][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict[key])]

    return sparse_embedding, linear_embedding


def get_embeddings_multi_val(feature_dim_dict, key, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    # the input dim is the number of different words
    multi_val_embedding = [Embedding(feature_dim_dict[key][feat][1], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name=key+'_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict[key])]
    linear_embedding = [Embedding(feature_dim_dict[key][feat][1], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                  seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict[key])]

    return multi_val_embedding, linear_embedding


def get_pooling(input, feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    # using this will get error:"ValueError: Output tensors to a Model must be the output of a TensorFlow `Layer`
    # (thus holding past layer metadata). Found: Tensor("prediction_layer/Reshape_1:0", shape=(?, 1), dtype=float32)"

    # movie_tags = [[0, 1, 2, 0, 0, 0, 0],  # movie1 具有0，1，2 一共3个标签
    #     [0, 1, 2, 3, 4, 0, 0]]  # movie2 具有0，1，2，3，4 一共5个标签
    # tags_len = [[3], [5]]  # 这里记得输入变长特征的有效长度
    # model_input = [movie_tags, tags_len]
    # 之后我们可以根据输入拿到对应的embeddding矩阵tag_embedding
    # 按照API的要求输入embedding矩阵和长度
    # tags_pooling = SequencePoolingLayer(seq_len_max=7, mode='mean', )([tag_embedding, tags_len_input])
    # 这样就得到了对变长多值特征的一个定长表示，后续可以进行其他操作
    multi_val_embedding = []
    multi_val_linear_embedding = []
    for i, feat in enumerate(feature_dim_dict['multi_val']):
        max_len = feature_dim_dict['multi_val'][feat]
        # embedding_input = [input[i][:, 0:max_len], input[i][:, -1]]
        tag_embedding = Embedding(feature_dim_dict["multi_val"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='multi_val_emb_' + str(i) + '-' + feat)

        tag_len = tf.expand_dims(input[i][:, max_len], -1)

        tag_embedding = Lambda(tag_embedding)(input[i][:, 0:max_len])
        multi_val_pooling = Lambda(SequencePoolingLayer(seq_len_max=max_len, mode='max',
                                                 name='multi_val_emb_' + str(i) + '-' + feat))([tag_embedding, tag_len])
        multi_val_embedding.append(multi_val_pooling)

        linear_embedding = Embedding(feature_dim_dict["multi_val"][feat], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                          seed=seed),
                                      embeddings_regularizer=l2(l2_reg_w),
                                      name='linear_emb_' + str(i) + '-' + feat)

        linear_embedding = Lambda(linear_embedding)((input[i][:, 0:max_len]))
        multi_val_linear_pooling = Lambda(SequencePoolingLayer(seq_len_max=max_len, mode='max',
                                                        name='linear_emb_' + str(i) + '-' + feat))([linear_embedding, tag_len])
        multi_val_linear_embedding.append(multi_val_linear_pooling)

    return multi_val_embedding, multi_val_linear_embedding



def get_multi_val_embedding(input, feature_dim_dict, embedding_size):
    # input is a sparse matrix, but keras can not input a sparse matrix
    multi_val_embedding = []
    for i, feat in enumerate(feature_dim_dict['multi_val']):
        embedding_params = tf.Variable(tf.truncated_normal([feature_dim_dict['multi_val'][feat], embedding_size]))
        embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=input, sp_weights=None)
        multi_val_embedding.append(embedded_tags)

    return multi_val_embedding
