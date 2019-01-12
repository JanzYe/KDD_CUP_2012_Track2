from tensorflow.python.keras.layers import Input
from .activations import *
from .layers import *
from .sequence import *


def get_input(feature_dim_dict, bias_feature_dim_dict=None):
    sparse_input = [Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])]
    dense_input = [Input(shape=(1,), name='dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])]
    if bias_feature_dim_dict is None:
        return sparse_input, dense_input
    else:
        bias_sparse_input = [Input(shape=(1,), name='bias_sparse_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["sparse"])]
        bias_dense_input = [Input(shape=(1,), name='bias_dense_' + str(i) + '-' + feat) for i, feat in
                            enumerate(bias_feature_dim_dict["dense"])]
        return sparse_input, dense_input, bias_sparse_input, bias_dense_input


def get_input_multi(feature_dim_dict, bias_feature_dim_dict=None):
    sparse_input = [Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])]
    dense_input = [Input(shape=(1,), name='dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])]
    # the input shape is the max length of a token
    multi_val_input = [Input(shape=(feature_dim_dict["multi_val"][feat][0],), name='multi_val_' + str(i) + '-' + feat)
                       for i, feat in enumerate(feature_dim_dict["multi_val"])]
    valid_len_input = [Input(shape=(1,), name='valid_len_' + str(i) + '-' + feat)
                        for i, feat in enumerate(feature_dim_dict["multi_val"])]

    if bias_feature_dim_dict is None:
        return sparse_input, dense_input, multi_val_input, valid_len_input
    else:
        bias_sparse_input = [Input(shape=(1,), name='bias_sparse_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["sparse"])]
        bias_dense_input = [Input(shape=(1,), name='bias_dense_' + str(i) + '-' + feat) for i, feat in
                            enumerate(bias_feature_dim_dict["dense"])]
        bias_multi_val_input = [Input(shape=(feature_dim_dict["multi_val"][feat]+1,), name='bias_multi_val_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["multi_val"])]
        return sparse_input, dense_input, bias_sparse_input, bias_dense_input, multi_val_input, bias_multi_val_input


custom_objects = {'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'MLP': MLP,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'AFMLayer': AFMLayer,
                  'CrossNet': CrossNet,
                  'BiInteractionPooling': BiInteractionPooling,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer, }
